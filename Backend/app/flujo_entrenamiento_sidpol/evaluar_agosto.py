"""Comprueba en agosto 2026 el modelo elegido sin usar agosto para ajustarlo."""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from app.flujo_entrenamiento_sidpol.config import TrainingConfig
from app.flujo_entrenamiento_sidpol.experimentos_alto_binario import combined_metrics
from app.flujo_entrenamiento_sidpol.experimentos_vecindad import SpatialFeatureBuilder, _high_scores
from app.flujo_entrenamiento_sidpol.modelado import _collect_probabilities, _evaluate, _fit, _models, _sample
from app.flujo_entrenamiento_sidpol.segmentacion import MONTHS, cargar_matrices
from app.flujo_entrenamiento_sidpol.variables import FeatureBuilder


def run():
    config = TrainingConfig.load()
    root = config.output
    output = root / "experimentos_alto_binario"
    baseline = json.loads((root / "resumen_entrenamiento.json").read_text(encoding="utf-8"))
    tuning = json.loads((output / "ajuste_comparacion.json").read_text(encoding="utf-8"))
    neighborhood = json.loads((root / "experimentos_vecindad" / "comparacion.json").read_text(encoding="utf-8"))
    train_months = [MONTHS.index(month) for month in baseline["final_training_months"]]
    holdout = [MONTHS.index("2026-08")]
    high_from = float(baseline["risk_high_threshold"])
    tramos = pd.read_csv(root / "tramos_osm.csv")
    matrices = cargar_matrices(root)
    original = FeatureBuilder(tramos, matrices, config)
    spatial = SpatialFeatureBuilder(tramos, matrices, config)

    results = {}
    for name in ("random_forest", "xgboost"):
        model = joblib.load(root / f"modelo_{name}.joblib")["pipeline"]
        truth, probability = _collect_probabilities(model, original, holdout, high_from)
        assessment = _evaluate(
            truth, probability, holdout,
            baseline["assessments"][name]["decision_thresholds"],
        )
        assessment.update(_high_scores(assessment))
        results[name] = assessment
        print(f"Agosto modelo original {name}: F1 alto {assessment['f1_alto']:.4f}", flush=True)
        del model, truth, probability

    x_spatial, y_spatial = _sample(spatial, train_months, high_from, config.random_state)
    rf = _models(config)["random_forest"]
    _fit(rf, x_spatial, y_spatial)
    del x_spatial, y_spatial
    joblib.dump(rf, output / "modelo_rf_vecindad_hasta_julio.joblib", compress=3)

    x_original, y_original = _sample(original, train_months, high_from, config.random_state)
    binary = XGBClassifier(
        n_estimators=450,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=3,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=config.random_state,
    )
    binary.fit(x_original, (y_original == 2).astype(np.int8))
    del x_original, y_original
    joblib.dump(binary, output / "modelo_alto_binario_hasta_julio.joblib", compress=3)
    result = combined_metrics(
        binary, original, rf, spatial, holdout, high_from,
        tuning["selected_validation"]["threshold"],
        neighborhood["results"]["random_forest"]["decision_thresholds"]["medio"],
    )
    results["alto_binario_medio_rf_vecindad"] = result
    print(f"Agosto modelo optimizado: F1 alto {result['f1_alto']:.4f}", flush=True)
    summary = {
        "created_at": datetime.now().isoformat(),
        "holdout": "2026-08",
        "training_months": [MONTHS[i] for i in train_months],
        "high_threshold": high_from,
        "models": results,
    }
    path = output / "holdout_agosto.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(run(), flush=True)
