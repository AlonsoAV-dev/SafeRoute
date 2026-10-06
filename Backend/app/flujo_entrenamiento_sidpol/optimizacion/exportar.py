"""Reentrena candidatos elegidos, evalúa 2026 y exporta versiones para septiembre.

La evaluación de 2026 se calcula con modelos ajustados hasta 2024. Los modelos
finales incorporan después 2026 para predecir septiembre; sus métricas futuras
todavía no son observables.
"""

from __future__ import annotations

import gc
import json
import argparse
from pathlib import Path

import joblib
import numpy as np

from app.flujo_entrenamiento_sidpol.config import TURNOS
from app.flujo_entrenamiento_sidpol.optimizacion.ejecutar import Experiment, write_json
from app.flujo_entrenamiento_sidpol.optimizacion.evaluacion import ProbabilityCalibration, metrics, predict_levels, tune_joint_thresholds
from app.flujo_entrenamiento_sidpol.segmentacion import MONTHS


# Exporta las predicciones de las configuraciones seleccionadas y conserva sus metadatos para el aplicativo.
def run(all_families=False):
    config_path = Path(__file__).resolve().parents[3] / "config_optimizacion_sidpol.json"
    experiment = Experiment(config_path)
    results = experiment.results
    if "selected" not in results or "retrospective" not in results["candidates"][results["selected"]]:
        raise ValueError("Complete primero la comparación y evaluación retrospectiva")
    chosen = {results["selected"]}
    if all_families:
        chosen |= set(results["selected_per_family"][family] for family in ("rf", "xgb"))
    names = sorted(chosen)
    recent_months = sorted(set(experiment.train_months + experiment.stages["calibration"] + experiment.stages["selection"]))
    y = np.load(experiment.cache / "retrospective_y.npy")
    recent = results.setdefault("refit_until_2024", {})
    for name in names:
        if name in recent:
            continue
        candidate = results["candidates"][name]
        print(f"REENTRENANDO HASTA 2024: {name}", flush=True)
        model = experiment.fit(candidate["spec"], tag="train_recent", months=recent_months)
        joblib.dump(model, experiment.output / f"modelo_2024_{name}.joblib", compress=3)
        raw = experiment.probability(model, candidate["spec"]["features"], "retrospective")
        np.save(experiment.output / f"probabilidades_2026_refit_raw_{name}.npy", raw)
        variant = candidate["selected_variant"]
        probability = raw
        if variant == "calibrated":
            calibrator = joblib.load(experiment.output / f"calibrador_{name}.joblib")
            probability = calibrator.predict_proba(raw)
        cutoffs = candidate["variants"][variant]["cutoffs"]
        recent[name] = {"training_months": [MONTHS[m] for m in recent_months], "metrics": metrics(y, probability, cutoffs),
                        "by_month": experiment.per_month("retrospective", probability, cutoffs)}
        if name in ("base_rf", "base_xgb"):
            key = "random_forest" if name == "base_rf" else "xgboost"
            original = experiment.baseline["assessments"][key]["decision_thresholds"]
            recent[name]["original_cutoffs"] = metrics(y, raw, original)
        print(f"2026 con ajuste hasta 2024 {name}: F1 medio/alto={recent[name]['metrics']['f1_medio_alto']:.4f}", flush=True)
        write_json(experiment.results_path, results)
        del model, probability, raw
        gc.collect()

    final_months = sorted(set(recent_months + experiment.stages["retrospective"]))
    final_dir = experiment.output / "modelos_finales"
    final_dir.mkdir(exist_ok=True)
    exported = results.setdefault("final_exports", {})
    forecast = experiment.config.forecast_period
    month = MONTHS.index(forecast)
    labels = np.array(["bajo", "medio", "alto"])
    for name in sorted(chosen):
        if name in exported:
            continue
        candidate = results["candidates"][name]
        print(f"AJUSTE FINAL HASTA AGOSTO: {name}", flush=True)
        model = experiment.fit(candidate["spec"], tag="train_final", months=final_months)
        raw_oos = np.load(experiment.output / f"probabilidades_2026_refit_raw_{name}.npy")
        # Calibración operativa basada en predicciones de 2026 realizadas por el
        # modelo ajustado hasta 2024; no se informa como una nueva evaluación.
        calibrator = (ProbabilityCalibration().fit(y, raw_oos, experiment.settings["random_state"])
                      if candidate["selected_variant"] == "calibrated" else None)
        calibrated_oos = raw_oos if calibrator is None else calibrator.predict_proba(raw_oos)
        cutoffs = tune_joint_thresholds(y, calibrated_oos)
        del raw_oos, calibrated_oos
        column_ids = experiment.builder.groups[candidate["spec"]["features"]]
        metadata = {
            "name": name, "selected_by_validation": name == results["selected"],
            "spec": candidate["spec"], "feature_names": [experiment.builder.names[i] for i in column_ids],
            "training_months": [MONTHS[m] for m in final_months], "forecast_period": forecast,
            "target_high_threshold": experiment.settings["high_threshold"], "decision_thresholds": cutoffs,
            "probability_variant": candidate["selected_variant"],
            "calibration_source": "predicciones 2026 fuera del ajuste de árboles; árboles ajustados hasta 2024",
            "evaluation_note": "Las métricas retrospectivas corresponden a versiones anteriores al ajuste final. Septiembre todavía no tiene etiquetas observadas.",
        }
        joblib.dump({"model": model, "calibrator": calibrator, "metadata": metadata}, final_dir / f"modelo_{name}.joblib", compress=3)
        prediction = experiment.builder.tramos[["tramo_id", "latitud", "longitud"]].copy()
        prediction["periodo_objetivo"] = forecast
        turn_probabilities = []
        for turn, turn_name in enumerate(TURNOS):
            x = experiment.builder.block(month, turn)[:, column_ids]
            p = model.predict_proba(x)
            if calibrator is not None:
                p = calibrator.predict_proba(p)
            turn_probabilities.append(p)
            prediction[f"probabilidad_medio_{turn_name}"] = p[:, 1]
            prediction[f"probabilidad_alto_{turn_name}"] = p[:, 2]
            prediction[f"riesgo_score_{turn_name}"] = .5 * p[:, 1] + p[:, 2]
            prediction[f"nivel_riesgo_{turn_name}"] = labels[predict_levels(p, cutoffs)]
        mean = np.mean(turn_probabilities, axis=0)
        prediction["riesgo_score"] = .5 * mean[:, 1] + mean[:, 2]
        prediction["nivel_riesgo"] = labels[predict_levels(mean, cutoffs)]
        prediction["modelo_usado"] = name
        prediction.to_csv(final_dir / f"predicciones_tramos_{name}.csv", index=False)
        write_json(final_dir / f"metadata_{name}.json", metadata)
        exported[name] = {"model": str(final_dir / f"modelo_{name}.joblib"),
                          "predictions": str(final_dir / f"predicciones_tramos_{name}.csv"),
                          "metadata": metadata}
        write_json(experiment.results_path, results)
        del model, calibrator, prediction
        gc.collect()
    print(experiment.results_path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-families", action="store_true")
    run(parser.parse_args().all_families)
