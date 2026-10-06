"""Reentrena con coordenadas experimentales y evalúa con etiquetas observadas.

Desde Backend:
    python -m app.flujo_entrenamiento_sidpol.optimizacion.evaluar_escenario_distrito

Las coordenadas copiadas alimentan las variables y las etiquetas de entrenamiento.
La calibración, selección y evaluación retrospectiva usan únicamente la matriz
construida a partir de las coordenadas originales, sin simulaciones.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.flujo_entrenamiento_sidpol.optimizacion.ejecutar import Experiment, write_json
from app.flujo_entrenamiento_sidpol.optimizacion.evaluacion import ProbabilityCalibration, metrics, tune_joint_thresholds
from app.flujo_entrenamiento_sidpol.segmentacion import MONTHS, cargar_matrices


BACKEND = Path(__file__).resolve().parents[3]
CONFIG = BACKEND / "config_experimento_sidpol_escenario_distrito.json"
ORIGINAL = BACKEND / "data" / "procesados_sidpol_v2"
ORIGINAL_COMPARISON = BACKEND / "data" / "experimentos_sidpol" / "ventanas_v1" / "comparacion.json"


class ObservedEvaluation(Experiment):
    """Usa variables del escenario, pero verdad observada fuera de entrenamiento."""

    def __init__(self, config_path: Path):
        self._observed_target = False
        super().__init__(config_path)
        original_ids = pd.read_csv(ORIGINAL / "tramos_osm.csv", usecols=["tramo_id"])["tramo_id"]
        scenario_ids = self.builder.tramos["tramo_id"]
        if not original_ids.equals(scenario_ids):
            raise ValueError("El orden de tramos cambió; no se pueden comparar las etiquetas")
        self.observed_weight = cargar_matrices(ORIGINAL)["weight"]
        if self.observed_weight.shape != self.builder.matrices["weight"].shape:
            raise ValueError("Las matrices original y experimental tienen distinta forma")

    def target(self, month: int, turn: int):
        if not self._observed_target:
            return super().target(month, turn)
        risk = self.observed_weight.getrow(month * 4 + turn).toarray().ravel() / self.builder.factor
        high = self.settings["high_threshold"]
        return np.where(risk >= high, 2, np.where(risk > 0, 1, 0)).astype(np.int8)

    def cache_stage(self, stage: str):
        self._observed_target = True
        try:
            super().cache_stage(stage)
        finally:
            self._observed_target = False

    def cache_training(self, months=None, tag="train", record_in_results=False):
        return super().cache_training(months, tag, record_in_results=False)


def prepare_reference():
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_config = json.loads((BACKEND / settings["source_config"]).read_text(encoding="utf-8"))
    scenario = (BACKEND / source_config["output"]).resolve()
    baseline = json.loads((ORIGINAL / "resumen_entrenamiento.json").read_text(encoding="utf-8"))
    reference = {
        "reference_only": True,
        "source": str(ORIGINAL / "resumen_entrenamiento.json"),
        "splits": baseline["splits"],
        "assessments": baseline["assessments"],
    }
    path = scenario / "resumen_entrenamiento.json"
    if not path.exists():
        write_json(path, reference)


def main():
    prepare_reference()
    experiment = ObservedEvaluation(CONFIG)
    result_path = experiment.output / "evaluacion_observada.json"
    original = json.loads(ORIGINAL_COMPARISON.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
        "scenario": "coordenadas recuperadas de la misma denuncia o copiadas de otro delito del mismo año y distrito",
        "training_labels": "escenario experimental, incluye ubicaciones simuladas",
        "evaluation_labels": "solo coordenadas originales observadas",
        "train_months": [MONTHS[m] for m in experiment.train_months],
        "calibration_months": experiment.settings["calibration_months"],
        "selection_months": experiment.settings["selection_months"],
        "retrospective_months": experiment.settings["retrospective_months"],
        "original_comparison": str(ORIGINAL_COMPARISON),
        "candidates": {},
    }
    for stage in ("calibration", "selection", "retrospective"):
        experiment.cache_stage(stage)
    y_cal = np.load(experiment.cache / "calibration_y.npy", mmap_mode="r")
    y_selection = np.load(experiment.cache / "selection_y.npy", mmap_mode="r")
    y_test = np.load(experiment.cache / "retrospective_y.npy", mmap_mode="r")
    print("Etiquetas observadas por clase:", {stage: np.bincount(y, minlength=3).tolist()
          for stage, y in (("calibration", y_cal), ("selection", y_selection), ("retrospective", y_test))}, flush=True)

    for spec in experiment.settings["candidates"]:
        name = spec["name"]
        if name in result["candidates"]:
            continue
        start = time.monotonic()
        print(f"ENTRENANDO ESCENARIO {name}", flush=True)
        model = experiment.fit(spec)
        joblib.dump(model, experiment.output / f"modelo_{name}.joblib", compress=3)
        p_cal = experiment.probability(model, spec["features"], "calibration")
        p_sel = experiment.probability(model, spec["features"], "selection")
        calibrator = ProbabilityCalibration().fit(y_cal, p_cal, experiment.settings["random_state"])
        joblib.dump(calibrator, experiment.output / f"calibrador_{name}.joblib")
        variants = {}
        for variant in ("raw", "calibrated"):
            calibrated_cal = p_cal if variant == "raw" else calibrator.predict_proba(p_cal)
            calibrated_sel = p_sel if variant == "raw" else calibrator.predict_proba(p_sel)
            cutoffs = tune_joint_thresholds(y_cal, calibrated_cal)
            selection = metrics(y_selection, calibrated_sel, cutoffs)
            variants[variant] = {"cutoffs": cutoffs, "selection": selection}
            print(f"SELECCIÓN {name}/{variant}: F1 medio+alto={selection['f1_medio_alto']:.4f}", flush=True)
            del calibrated_cal, calibrated_sel
            gc.collect()
        selected = max(variants, key=lambda v: variants[v]["selection"]["f1_medio_alto"])
        p_test = experiment.probability(model, spec["features"], "retrospective")
        if selected == "calibrated":
            p_test = calibrator.predict_proba(p_test)
        cutoffs = variants[selected]["cutoffs"]
        retrospective = metrics(y_test, p_test, cutoffs)
        by_month = experiment.per_month("retrospective", p_test, cutoffs)
        previous = original["candidates"][name]["retrospective"]
        result["candidates"][name] = {
            "spec": spec, "variants": variants, "selected_variant": selected,
            "retrospective": retrospective, "retrospective_by_month": by_month,
            "original_retrospective": previous,
            "delta_f1_medio_alto": retrospective["f1_medio_alto"] - previous["f1_medio_alto"],
            "elapsed_seconds": round(time.monotonic() - start, 1),
        }
        write_json(result_path, result)
        print(f"RESULTADO {name}: observado={retrospective['f1_medio_alto']:.4f}; "
              f"original={previous['f1_medio_alto']:.4f}", flush=True)
        del model, calibrator, p_cal, p_sel, p_test
        gc.collect()

    ranking = sorted(result["candidates"], key=lambda n: result["candidates"][n]["variants"]
                     [result["candidates"][n]["selected_variant"]]["selection"]["f1_medio_alto"], reverse=True)
    result["ranking_selection"] = ranking
    result["selected_by_selection"] = ranking[0]
    write_json(result_path, result)
    print(f"RESULTADOS {result_path}", flush=True)


if __name__ == "__main__":
    main()
