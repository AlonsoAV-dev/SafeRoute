"""Aísla el efecto de entrenar el XGBoost experimental también con 2024–2025.

Calibra en enero–febrero 2026, elige variante en marzo 2026 y evalúa
abril–agosto 2026, siempre con etiquetas basadas en ubicaciones originales.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import joblib
import numpy as np

from app.flujo_entrenamiento_sidpol.optimizacion.ejecutar import write_json
from app.flujo_entrenamiento_sidpol.optimizacion.evaluacion import ProbabilityCalibration, metrics, tune_joint_thresholds
from app.flujo_entrenamiento_sidpol.optimizacion.evaluar_escenario_distrito import BACKEND, ObservedEvaluation
from app.flujo_entrenamiento_sidpol.segmentacion import MONTHS


CONFIG = BACKEND / "config_experimento_sidpol_escenario_reciente.json"
OLD_MODEL = BACKEND / "data" / "experimentos_sidpol" / "escenario_distrito" / "modelos" / "modelo_xgb_multi_recent.joblib"


def main():
    experiment = ObservedEvaluation(CONFIG)
    result_path = experiment.output / "comparacion_entrenamiento_2025.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
        "source": str(experiment.config.source),
        "evaluation_labels": "coordenadas originales observadas",
        "calibration_months": experiment.settings["calibration_months"],
        "selection_months": experiment.settings["selection_months"],
        "retrospective_months": experiment.settings["retrospective_months"],
        "candidates": {},
    }
    early_months = experiment.train_months.copy()
    recent_months = [MONTHS.index(m) for m in MONTHS if "2023-06" <= m <= "2025-12"]
    full_months = sorted(set(early_months + recent_months))
    result["training_months_old"] = [MONTHS[m] for m in early_months]
    result["training_months_recent"] = [MONTHS[m] for m in full_months]
    result["training_labels"] = "coordenadas experimentales, incluidas simuladas"
    for stage in ("calibration", "selection", "retrospective"):
        experiment.cache_stage(stage)
    labels = {stage: np.load(experiment.cache / f"{stage}_y.npy", mmap_mode="r")
              for stage in ("calibration", "selection", "retrospective")}
    spec = experiment.settings["candidates"][0]

    for name in ("hasta_2023", "hasta_2025"):
        if name in result["candidates"]:
            continue
        print(f"EVALUANDO {name}", flush=True)
        if name == "hasta_2023":
            model = joblib.load(OLD_MODEL)
        else:
            model = experiment.fit(spec, tag="train_hasta_2025", months=full_months)
            joblib.dump(model, experiment.output / "modelo_xgb_hasta_2025.joblib", compress=3)
        p_cal = experiment.probability(model, "multi", "calibration")
        p_sel = experiment.probability(model, "multi", "selection")
        calibrator = ProbabilityCalibration().fit(labels["calibration"], p_cal, experiment.settings["random_state"])
        variants = {}
        for variant in ("raw", "calibrated"):
            c = p_cal if variant == "raw" else calibrator.predict_proba(p_cal)
            s = p_sel if variant == "raw" else calibrator.predict_proba(p_sel)
            cutoffs = tune_joint_thresholds(labels["calibration"], c)
            selection = metrics(labels["selection"], s, cutoffs)
            variants[variant] = {"cutoffs": cutoffs, "selection": selection}
            print(f"SELECCIÓN {name}/{variant}: {selection['f1_medio_alto']:.4f}", flush=True)
            del c, s
            gc.collect()
        selected = max(variants, key=lambda v: variants[v]["selection"]["f1_medio_alto"])
        p_test = experiment.probability(model, "multi", "retrospective")
        if selected == "calibrated":
            p_test = calibrator.predict_proba(p_test)
        cutoffs = variants[selected]["cutoffs"]
        retrospective = metrics(labels["retrospective"], p_test, cutoffs)
        result["candidates"][name] = {
            "train_months": result["training_months_old"] if name == "hasta_2023" else result["training_months_recent"],
            "selected_variant": selected, "variants": variants,
            "retrospective": retrospective,
            "retrospective_by_month": experiment.per_month("retrospective", p_test, cutoffs),
        }
        write_json(result_path, result)
        print(f"RETROSPECTIVA {name}: {retrospective['f1_medio_alto']:.4f}", flush=True)
        del model, calibrator, p_cal, p_sel, p_test
        gc.collect()
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
