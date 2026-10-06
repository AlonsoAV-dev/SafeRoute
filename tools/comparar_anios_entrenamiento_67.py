"""Aísla el efecto de añadir 2018–2024 al entrenamiento actual."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "Backend"
sys.path.insert(0, str(BACKEND))

from app.flujo_entrenamiento_sidpol.optimizacion.ejecutar import Experiment
from app.flujo_entrenamiento_sidpol.optimizacion.evaluacion import ProbabilityCalibration, metrics, tune_joint_thresholds
from app.flujo_entrenamiento_sidpol.segmentacion import MONTHS


OUT = ROOT / "outputs" / "evaluacion-modelos-sidpol" / "comparacion_67"
CONFIG = BACKEND / "config_experimento_sidpol_completo_2018_2026.json"
BASELINE = BACKEND / "data" / "experimentos_sidpol" / "entrenamiento_completo_2018_2026" / "evaluacion_temporal.json"
RESULT = OUT / "comparacion_anios_entrenamiento.json"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if RESULT.exists():
        print(RESULT, flush=True)
        return
    experiment = Experiment(CONFIG)
    train_months = [MONTHS.index(f"2025-{month:02d}") for month in range(1, 13)]
    spec = next(item for item in experiment.settings["candidates"] if item["name"] == "xgb_multi_recent")
    experiment.cache_training(months=train_months, tag="train_only_2025", record_in_results=False)
    model_path = OUT / "modelo_xgb_solo_2025_evaluado.joblib"
    model = joblib.load(model_path) if model_path.exists() else experiment.fit(
        spec, tag="train_only_2025", months=train_months)
    if not model_path.exists():
        joblib.dump(model, model_path, compress=3)
    labels = {stage: np.load(experiment.cache / f"{stage}_y.npy", mmap_mode="r")
              for stage in ("calibration", "selection", "retrospective")}
    p_cal = experiment.probability(model, spec["features"], "calibration")
    p_sel = experiment.probability(model, spec["features"], "selection")
    calibrator = ProbabilityCalibration().fit(labels["calibration"], p_cal, experiment.settings["random_state"])
    variants = {}
    for name in ("raw", "calibrated"):
        cal = p_cal if name == "raw" else calibrator.predict_proba(p_cal)
        sel = p_sel if name == "raw" else calibrator.predict_proba(p_sel)
        cutoffs = tune_joint_thresholds(labels["calibration"], cal)
        variants[name] = {"cutoffs": cutoffs, "selection": metrics(labels["selection"], sel, cutoffs)}
        print(f"VALIDACIÓN 2025 {name}: {variants[name]['selection']['f1_medio_alto']:.4f}", flush=True)
        del cal, sel
        gc.collect()
    selected = max(variants, key=lambda v: variants[v]["selection"]["f1_medio_alto"])
    p_test = experiment.probability(model, spec["features"], "retrospective")
    if selected == "calibrated":
        p_test = calibrator.predict_proba(p_test)
    result_2025 = metrics(labels["retrospective"], p_test, variants[selected]["cutoffs"])
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    full = base["candidates"]["xgb_multi_recent"]
    outcome = {
        "status": "complete",
        "fixed": "misma base nueva, snapping 150 m, unidad tramo×mes×turno, mismas variables y parámetros XGBoost, calibración enero–febrero 2026, validación marzo–abril 2026, prueba mayo–agosto 2026",
        "training_2025_only": {"months": [MONTHS[m] for m in train_months], "selected_variant": selected,
                               "selection": variants[selected]["selection"], "test": result_2025,
                               "evaluated_model": str(model_path)},
        "training_2018_2025": {"months": base["train_months"], "selected_variant": full["selected_variant"],
                               "selection": full["variants"][full["selected_variant"]]["selection"],
                               "test": full["test"], "evaluated_model": full["evaluated_model"]},
        "note": "Solo cambian los meses de filas objetivo de entrenamiento; las variables de enero–diciembre 2025 pueden usar meses previos como historia.",
    }
    RESULT.write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PRUEBA 2025 {result_2025['f1_medio_alto']:.4f}; 2018–2025 {full['test']['f1_medio_alto']:.4f}", flush=True)
    print(RESULT, flush=True)


if __name__ == "__main__":
    main()
