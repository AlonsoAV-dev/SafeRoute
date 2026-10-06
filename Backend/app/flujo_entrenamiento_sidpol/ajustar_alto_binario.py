"""Ajuste pequeño del clasificador alto; selección solo por validación 2024."""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from app.flujo_entrenamiento_sidpol.config import TrainingConfig
from app.flujo_entrenamiento_sidpol.experimentos_alto_binario import binary_metrics, collect, combined_metrics
from app.flujo_entrenamiento_sidpol.experimentos_vecindad import SpatialFeatureBuilder
from app.flujo_entrenamiento_sidpol.modelado import _sample, _threshold
from app.flujo_entrenamiento_sidpol.segmentacion import cargar_matrices
from app.flujo_entrenamiento_sidpol.variables import FeatureBuilder, eligible_months


def run():
    config = TrainingConfig.load()
    root = config.output
    output = root / "experimentos_alto_binario"
    previous = json.loads((output / "comparacion.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "auditoria_fuente.json").read_text(encoding="utf-8"))
    splits = eligible_months(config, audit)
    tramos = pd.read_csv(root / "tramos_osm.csv")
    matrices = cargar_matrices(root)
    builder = FeatureBuilder(tramos, matrices, config)
    high_from = _threshold(builder, splits["train"])
    x_train, y_train = _sample(builder, splits["train"], high_from, config.random_state)
    binary_y = (y_train == 2).astype(np.int8)
    candidates = {}
    incumbent_name = previous["selected"]
    incumbent = previous["candidates"][incumbent_name]["validation"]
    best = (incumbent["f1_alto"], incumbent["pr_auc_alto"])
    selected = incumbent_name
    designs = (
        ("original_depth2_500", 2, 500, 0.05),
        ("original_depth3_450", 3, 450, 0.05),
        ("original_depth5_250", 5, 250, 0.05),
        ("original_depth4_600", 4, 600, 0.03),
    )
    for name, depth, trees, rate in designs:
        model = XGBClassifier(
            n_estimators=trees,
            max_depth=depth,
            learning_rate=rate,
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
        print(f"Entrenando {name}", flush=True)
        model.fit(x_train, binary_y)
        truth, scores = collect(model, builder, splits["validation"], high_from)
        validation = binary_metrics(truth, scores)
        candidates[name] = validation
        print(f"{name}: F1 alto {validation['f1_alto']:.4f}, PR-AUC {validation['pr_auc_alto']:.4f}", flush=True)
        ranking = (validation["f1_alto"], validation["pr_auc_alto"])
        if ranking > best:
            best = ranking
            selected = name
            joblib.dump(model, output / "modelo_ajustado_seleccionado_validacion.joblib", compress=3)
        del model, truth, scores
    del x_train, y_train, binary_y
    if selected == incumbent_name:
        model = joblib.load(output / "modelo_seleccionado_validacion.joblib")
        cutoff = incumbent["threshold"]
        test = previous["test_binary"]
        combined = previous["test_combined_with_spatial_rf_for_medium"]
    else:
        model = joblib.load(output / "modelo_ajustado_seleccionado_validacion.joblib")
        cutoff = candidates[selected]["threshold"]
        truth, scores = collect(model, builder, splits["test"], high_from)
        test = binary_metrics(truth, scores, cutoff)
        del truth, scores
        spatial_builder = SpatialFeatureBuilder(tramos, matrices, config)
        rf = joblib.load(root / "experimentos_vecindad" / "modelo_entrenamiento_random_forest.joblib")
        neighborhood = json.loads((root / "experimentos_vecindad" / "comparacion.json").read_text(encoding="utf-8"))
        medium_cutoff = neighborhood["results"]["random_forest"]["decision_thresholds"]["medio"]
        combined = combined_metrics(model, builder, rf, spatial_builder, splits["test"], high_from, cutoff, medium_cutoff)
    summary = {
        "created_at": datetime.now().isoformat(),
        "selection_basis": "mayor F1 de riesgo alto en validación 2024; prueba 2026 reservada",
        "incumbent": incumbent_name,
        "incumbent_validation": incumbent,
        "candidates": candidates,
        "selected": selected,
        "selected_validation": incumbent if selected == incumbent_name else candidates[selected],
        "test_binary": test,
        "test_combined_with_spatial_rf_for_medium": combined,
    }
    path = output / "ajuste_comparacion.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(run(), flush=True)
