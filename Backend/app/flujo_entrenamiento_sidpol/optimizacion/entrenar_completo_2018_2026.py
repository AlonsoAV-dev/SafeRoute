"""Entrena con todas las coordenadas del escenario como etiquetas ordinarias.

La evaluación temporal retiene mayo–agosto de 2026. Terminada la evaluación,
ajusta versiones finales independientes con enero 2018–agosto 2026 completos.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np

from app.flujo_entrenamiento_sidpol.optimizacion.ejecutar import Experiment, write_json
from app.flujo_entrenamiento_sidpol.optimizacion.evaluacion import ProbabilityCalibration, metrics, tune_joint_thresholds
from app.flujo_entrenamiento_sidpol.segmentacion import MONTHS


BACKEND = Path(__file__).resolve().parents[3]
CONFIG = BACKEND / "config_experimento_sidpol_completo_2018_2026.json"
TRAIN_END = MONTHS.index("2025-12")
FINAL_END = MONTHS.index("2026-08")


def evaluate():
    experiment = Experiment(CONFIG)
    experiment.train_months = list(range(TRAIN_END + 1))
    final_months = list(range(FINAL_END + 1))
    path = experiment.output / "evaluacion_temporal.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "source": str(experiment.config.source),
        "coordinate_policy": "todas las coordenadas del Excel experimental se tratan como datos de entrenamiento y verdad de evaluación",
        "target": "riesgo de un segmento vial por mes y turno; alto si gravedad/longitud >= 3",
        "train_months": [MONTHS[m] for m in experiment.train_months],
        "calibration_months": experiment.settings["calibration_months"],
        "selection_months": experiment.settings["selection_months"],
        "test_months": experiment.settings["retrospective_months"],
        "final_train_months": [MONTHS[m] for m in final_months],
        "test_metrics_apply_to": "modelos entrenados solo hasta diciembre de 2025; no a las versiones finales reajustadas",
        "candidates": {},
    }
    for stage in ("calibration", "selection", "retrospective"):
        experiment.cache_stage(stage)
    y_cal = np.load(experiment.cache / "calibration_y.npy", mmap_mode="r")
    y_sel = np.load(experiment.cache / "selection_y.npy", mmap_mode="r")
    y_test = np.load(experiment.cache / "retrospective_y.npy", mmap_mode="r")
    result["support"] = {"calibration": np.bincount(y_cal, minlength=3).tolist(),
                         "selection": np.bincount(y_sel, minlength=3).tolist(),
                         "test": np.bincount(y_test, minlength=3).tolist()}
    write_json(path, result)
    print("CLASES", result["support"], flush=True)

    # La misma matriz muestreada sirve para todos los candidatos con ventanas.
    experiment.cache_training(months=experiment.train_months, tag="train_2018_2025", record_in_results=False)
    for spec in experiment.settings["candidates"]:
        name = spec["name"]
        if "test" in result["candidates"].get(name, {}):
            continue
        started = time.monotonic()
        model_path = experiment.output / f"modelo_evaluado_{name}.joblib"
        print(f"ENTRENANDO {name}", flush=True)
        model = joblib.load(model_path) if model_path.exists() else experiment.fit(
            spec, tag="train_2018_2025", months=experiment.train_months)
        if not model_path.exists():
            joblib.dump(model, model_path, compress=3)
        p_cal = experiment.probability(model, spec["features"], "calibration")
        p_sel = experiment.probability(model, spec["features"], "selection")
        calibrator = ProbabilityCalibration().fit(y_cal, p_cal, experiment.settings["random_state"])
        joblib.dump(calibrator, experiment.output / f"calibrador_evaluado_{name}.joblib")
        variants = {}
        for variant in ("raw", "calibrated"):
            cal = p_cal if variant == "raw" else calibrator.predict_proba(p_cal)
            sel = p_sel if variant == "raw" else calibrator.predict_proba(p_sel)
            cutoffs = tune_joint_thresholds(y_cal, cal)
            selection = metrics(y_sel, sel, cutoffs)
            variants[variant] = {"cutoffs": cutoffs, "selection": selection}
            print(f"VALIDACIÓN {name}/{variant}: F1 medio+alto={selection['f1_medio_alto']:.4f}", flush=True)
            del cal, sel
            gc.collect()
        selected = max(variants, key=lambda v: variants[v]["selection"]["f1_medio_alto"])
        p_test = experiment.probability(model, spec["features"], "retrospective")
        if selected == "calibrated":
            p_test = calibrator.predict_proba(p_test)
        cutoffs = variants[selected]["cutoffs"]
        test = metrics(y_test, p_test, cutoffs)
        result["candidates"][name] = {
            "spec": spec,
            "selected_variant": selected,
            "variants": variants,
            "test": test,
            "test_by_month": experiment.per_month("retrospective", p_test, cutoffs),
            "evaluated_model": str(model_path),
            "elapsed_seconds": round(time.monotonic() - started, 1),
        }
        write_json(path, result)
        print(f"PRUEBA {name}: F1 medio+alto={test['f1_medio_alto']:.4f}", flush=True)
        del model, calibrator, p_cal, p_sel, p_test
        gc.collect()

    ranking = sorted(result["candidates"], key=lambda n: result["candidates"][n]["variants"]
                     [result["candidates"][n]["selected_variant"]]["selection"]["f1_medio_alto"], reverse=True)
    result["validation_ranking"] = ranking
    result["selected_by_validation"] = ranking[0]
    write_json(path, result)
    print("ELEGIDO POR VALIDACIÓN", ranking[0], flush=True)

    # Se crean pesos finales con todos los datos disponibles después de fijar
    # la evaluación. Sus métricas no se confunden con las del modelo retenido.
    if not (experiment.cache / "train_2018_2026_complete.json").exists():
        experiment.cache_training(months=final_months, tag="train_2018_2026", record_in_results=False)
    for spec in experiment.settings["candidates"]:
        name = spec["name"]
        final_path = experiment.output / f"modelo_final_{name}.joblib"
        if final_path.exists():
            result["candidates"][name]["final_model"] = str(final_path)
            continue
        print(f"AJUSTE FINAL 2018–2026 {name}", flush=True)
        model = experiment.fit(spec, tag="train_2018_2026", months=final_months)
        joblib.dump(model, final_path, compress=3)
        result["candidates"][name]["final_model"] = str(final_path)
        write_json(path, result)
        del model
        gc.collect()
    result["status"] = "completo"
    write_json(path, result)
    print("RESULTADOS", path, flush=True)


if __name__ == "__main__":
    evaluate()
