"""Compara NKDE vial frente al XGBoost temporal anterior, sin usar futuro."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np

from app.flujo_entrenamiento_sidpol.optimizacion.ejecutar import Experiment, write_json
from app.flujo_entrenamiento_sidpol.optimizacion.evaluacion import ProbabilityCalibration, metrics, tune_joint_thresholds
from app.flujo_entrenamiento_sidpol.optimizacion.nkde import BANDWIDTHS, NKDEWindowFeatures
from app.flujo_entrenamiento_sidpol.segmentacion import MONTHS


BACKEND = Path(__file__).resolve().parents[3]
CONFIG = BACKEND / "config_experimento_sidpol_nkde_v1.json"
BASELINE = BACKEND / "data" / "experimentos_sidpol" / "entrenamiento_completo_2018_2026" / "evaluacion_temporal.json"
TRAIN_END = MONTHS.index("2025-12")
FINAL_END = MONTHS.index("2026-08")


def main():
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))
    history_dir = BACKEND / settings["nkde_features"]
    if any(not (history_dir / f"historia_{h}m_complete.json").exists() for h in BANDWIDTHS):
        raise ValueError("Primero calcule los kernels e historia NKDE")
    experiment = Experiment(CONFIG)
    old_builder = experiment.builder
    experiment.builder = NKDEWindowFeatures(
        old_builder.tramos, old_builder.matrices, experiment.config, experiment.audit,
        settings["windows"], settings["minimum_history_coverage"], settings["high_threshold"], history_dir)
    del old_builder
    experiment.train_months = list(range(TRAIN_END + 1))
    final_months = list(range(FINAL_END + 1))
    result_path = experiment.output / "comparacion_nkde.json"
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    original = baseline["candidates"]["xgb_multi_recent"]
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
        "source": str(experiment.config.source),
        "nkde_definition": "kernel triangular de distancia mínima vial, densidad por 100 m; solo meses anteriores al objetivo",
        "snapping_radius_m": experiment.config.match_radius_m,
        "train_months": [MONTHS[m] for m in experiment.train_months],
        "calibration_months": settings["calibration_months"],
        "selection_months": settings["selection_months"],
        "test_months": settings["retrospective_months"],
        "baseline_source": str(BASELINE),
        "baseline": {
            "name": "xgb_multi_recent_sin_nkde",
            "selected_variant": original["selected_variant"],
            "selection": original["variants"][original["selected_variant"]]["selection"],
            "test": original["test"],
        },
        "candidates": {},
    }
    for stage in ("calibration", "selection", "retrospective"):
        experiment.cache_stage(stage)
    labels = {stage: np.load(experiment.cache / f"{stage}_y.npy", mmap_mode="r")
              for stage in ("calibration", "selection", "retrospective")}
    result["support"] = {stage: np.bincount(y, minlength=3).tolist() for stage, y in labels.items()}
    if result["support"]["retrospective"] != baseline["support"]["test"]:
        raise ValueError("Las etiquetas de prueba no coinciden con la comparación anterior")
    write_json(result_path, result)
    experiment.cache_training(months=experiment.train_months, tag="train_nkde_2018_2025", record_in_results=False)
    for spec in settings["candidates"]:
        name = spec["name"]
        if "test" in result["candidates"].get(name, {}):
            continue
        started = time.monotonic()
        model_path = experiment.output / f"modelo_evaluado_{name}.joblib"
        print(f"ENTRENANDO {name}", flush=True)
        model = joblib.load(model_path) if model_path.exists() else experiment.fit(
            spec, tag="train_nkde_2018_2025", months=experiment.train_months)
        if not model_path.exists():
            joblib.dump(model, model_path, compress=3)
        p_cal = experiment.probability(model, spec["features"], "calibration")
        p_sel = experiment.probability(model, spec["features"], "selection")
        calibrator = ProbabilityCalibration().fit(labels["calibration"], p_cal, settings["random_state"])
        joblib.dump(calibrator, experiment.output / f"calibrador_evaluado_{name}.joblib")
        variants = {}
        for variant in ("raw", "calibrated"):
            cal = p_cal if variant == "raw" else calibrator.predict_proba(p_cal)
            selection_probability = p_sel if variant == "raw" else calibrator.predict_proba(p_sel)
            cutoffs = tune_joint_thresholds(labels["calibration"], cal)
            selection = metrics(labels["selection"], selection_probability, cutoffs)
            variants[variant] = {"cutoffs": cutoffs, "selection": selection}
            print(f"VALIDACIÓN {name}/{variant}: F1 medio+alto={selection['f1_medio_alto']:.4f}", flush=True)
            del cal, selection_probability
            gc.collect()
        chosen = max(variants, key=lambda v: variants[v]["selection"]["f1_medio_alto"])
        p_test = experiment.probability(model, spec["features"], "retrospective")
        if chosen == "calibrated":
            p_test = calibrator.predict_proba(p_test)
        cutoffs = variants[chosen]["cutoffs"]
        test = metrics(labels["retrospective"], p_test, cutoffs)
        result["candidates"][name] = {
            "spec": spec, "selected_variant": chosen, "variants": variants,
            "test": test,
            "test_by_month": experiment.per_month("retrospective", p_test, cutoffs),
            "evaluated_model": str(model_path),
            "elapsed_seconds": round(time.monotonic() - started, 1),
        }
        write_json(result_path, result)
        print(f"PRUEBA {name}: F1 medio+alto={test['f1_medio_alto']:.4f}", flush=True)
        del model, calibrator, p_cal, p_sel, p_test
        gc.collect()

    ranking = sorted(result["candidates"], key=lambda name: result["candidates"][name]["variants"]
                     [result["candidates"][name]["selected_variant"]]["selection"]["f1_medio_alto"], reverse=True)
    best_nkde = ranking[0]
    baseline_score = result["baseline"]["selection"]["f1_medio_alto"]
    best_candidate = result["candidates"][best_nkde]
    nkde_score = best_candidate["variants"][best_candidate["selected_variant"]]["selection"]["f1_medio_alto"]
    result["ranking_validation_nkde"] = ranking
    result["selected_by_validation"] = best_nkde if nkde_score > baseline_score else "xgb_multi_recent_sin_nkde"
    write_json(result_path, result)
    print("ELEGIDO POR VALIDACIÓN", result["selected_by_validation"], flush=True)

    final_path = experiment.output / f"modelo_final_{best_nkde}.joblib"
    if not final_path.exists():
        experiment.cache_training(months=final_months, tag="train_nkde_2018_2026", record_in_results=False)
        spec = next(spec for spec in settings["candidates"] if spec["name"] == best_nkde)
        print(f"AJUSTE FINAL 2018–2026 {best_nkde}", flush=True)
        model = experiment.fit(spec, tag="train_nkde_2018_2026", months=final_months)
        joblib.dump(model, final_path, compress=3)
        del model
        gc.collect()
    result["best_nkde_final_model"] = str(final_path)
    result["status"] = "completo"
    write_json(result_path, result)
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
