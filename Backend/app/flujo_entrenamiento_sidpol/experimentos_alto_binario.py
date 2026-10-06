"""Experimento reproducible: clasificador especializado de riesgo alto.

Selecciona arquitectura y umbral usando solo la validación de 2024. La prueba
de 2026 se evalúa una vez, después de fijar la selección.
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier

from app.flujo_entrenamiento_sidpol.config import TrainingConfig, TURNOS
from app.flujo_entrenamiento_sidpol.experimentos_vecindad import SpatialFeatureBuilder
from app.flujo_entrenamiento_sidpol.modelado import _evaluate, _labels, _sample, _threshold
from app.flujo_entrenamiento_sidpol.segmentacion import cargar_matrices
from app.flujo_entrenamiento_sidpol.variables import FeatureBuilder, eligible_months


def collect(model, builder, months, high_from):
    truths, scores = [], []
    for month in months:
        for turn in range(len(TURNOS)):
            truths.append((_labels(builder.target_risk(month, turn), high_from) == 2).astype(np.int8))
            scores.append(model.predict_proba(builder.features(month, turn))[:, 1].astype(np.float32))
    return np.concatenate(truths), np.concatenate(scores)


def binary_metrics(truth, scores, cutoff=None):
    precision_curve, recall_curve, thresholds = precision_recall_curve(truth, scores)
    if cutoff is None:
        f1_curve = np.divide(
            2 * precision_curve[:-1] * recall_curve[:-1],
            precision_curve[:-1] + recall_curve[:-1],
            out=np.zeros(len(thresholds)),
            where=(precision_curve[:-1] + recall_curve[:-1]) > 0,
        )
        cutoff = float(thresholds[int(np.argmax(f1_curve))])
    matrix = confusion_matrix(truth, scores >= cutoff, labels=[0, 1])
    tp = int(matrix[1, 1])
    positive = int(matrix[1].sum())
    predicted = int(matrix[:, 1].sum())
    precision = tp / predicted if predicted else 0.0
    recall = tp / positive if positive else 0.0
    return {
        "threshold": cutoff,
        "precision_alto": precision,
        "recall_alto": recall,
        "f1_alto": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "pr_auc_alto": float(average_precision_score(truth, scores)),
        "matriz_binaria": matrix.tolist(),
    }


def combined_metrics(model, binary_builder, rf, spatial_builder, months, high_from, high_cutoff, medium_cutoff):
    truths, binary_scores, medium_scores = [], [], []
    for month in months:
        for turn in range(len(TURNOS)):
            truths.append(_labels(binary_builder.target_risk(month, turn), high_from))
            binary_scores.append(model.predict_proba(binary_builder.features(month, turn))[:, 1].astype(np.float32))
            medium_scores.append(rf.predict_proba(spatial_builder.features(month, turn))[:, 1].astype(np.float32))
    truth = np.concatenate(truths)
    probability = np.zeros((len(truth), 3), dtype=np.float32)
    probability[:, 1] = np.concatenate(medium_scores)
    probability[:, 2] = np.concatenate(binary_scores)
    result = _evaluate(truth, probability, months, {"alto": high_cutoff, "medio": medium_cutoff})
    matrix = np.asarray(result["matriz_confusion"])
    tp, actual, predicted = int(matrix[2, 2]), int(matrix[2].sum()), int(matrix[:, 2].sum())
    precision = tp / predicted if predicted else 0.0
    recall = tp / actual if actual else 0.0
    result.update({"precision_alto": precision, "f1_alto": 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
    return result


def run():
    config = TrainingConfig.load()
    root = config.output
    output = root / "experimentos_alto_binario"
    output.mkdir(exist_ok=True)
    audit = json.loads((root / "auditoria_fuente.json").read_text(encoding="utf-8"))
    splits = eligible_months(config, audit)
    tramos = pd.read_csv(root / "tramos_osm.csv")
    matrices = cargar_matrices(root)
    builders = {
        "original": FeatureBuilder(tramos, matrices, config),
        "vecindad": SpatialFeatureBuilder(tramos, matrices, config),
    }
    high_from = _threshold(builders["original"], splits["train"])
    designs = (("original_depth4", "original", 4), ("vecindad_depth4", "vecindad", 4), ("vecindad_depth6", "vecindad", 6))
    candidates = {}
    selected = None
    best_f1 = -1.0
    for features in ("original", "vecindad"):
        builder = builders[features]
        x_train, y_train = _sample(builder, splits["train"], high_from, config.random_state)
        binary_y = (y_train == 2).astype(np.int8)
        for name, design_features, depth in designs:
            if design_features != features:
                continue
            print(f"Entrenando {name}", flush=True)
            model = XGBClassifier(
                n_estimators=350,
                max_depth=depth,
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
            model.fit(x_train, binary_y)
            truth, score = collect(model, builder, splits["validation"], high_from)
            result = binary_metrics(truth, score)
            candidates[name] = {"features": features, "depth": depth, "validation": result}
            print(f"{name}: F1 alto {result['f1_alto']:.4f}, PR-AUC {result['pr_auc_alto']:.4f}", flush=True)
            if (result["f1_alto"], result["pr_auc_alto"]) > (best_f1, candidates[selected]["validation"]["pr_auc_alto"] if selected else -1):
                best_f1 = result["f1_alto"]
                selected = name
                joblib.dump(model, output / "modelo_seleccionado_validacion.joblib", compress=3)
            del truth, score, model
        del x_train, y_train, binary_y
    model = joblib.load(output / "modelo_seleccionado_validacion.joblib")
    chosen = candidates[selected]
    builder = builders[chosen["features"]]
    truth, score = collect(model, builder, splits["test"], high_from)
    test = binary_metrics(truth, score, chosen["validation"]["threshold"])
    del truth, score
    previous = json.loads((root / "experimentos_vecindad" / "comparacion.json").read_text(encoding="utf-8"))
    rf = joblib.load(root / "experimentos_vecindad" / "modelo_entrenamiento_random_forest.joblib")
    medium_cutoff = previous["results"]["random_forest"]["decision_thresholds"]["medio"]
    combined = combined_metrics(model, builder, rf, builders["vecindad"], splits["test"], high_from, chosen["validation"]["threshold"], medium_cutoff)
    summary = {
        "created_at": datetime.now().isoformat(),
        "selection_basis": "mayor F1 de riesgo alto en validación 2024, desempate PR-AUC",
        "selected": selected,
        "candidates": candidates,
        "test_binary": test,
        "test_combined_with_spatial_rf_for_medium": combined,
    }
    path = output / "comparacion.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(run(), flush=True)
