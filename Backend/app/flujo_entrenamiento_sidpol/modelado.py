from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier

from app.flujo_entrenamiento_sidpol.config import TURNOS, TrainingConfig
from app.flujo_entrenamiento_sidpol.segmentacion import MONTHS
from app.flujo_entrenamiento_sidpol.variables import FEATURE_NAMES, FeatureBuilder


LABELS = ("bajo", "medio", "alto")


# Estima el umbral de riesgo Alto usando los meses de entrenamiento admitidos.
def _threshold(builder: FeatureBuilder, months: list[int]) -> float:
    positives = []
    for month in months:
        for turn in range(len(TURNOS)):
            risk = builder.target_risk(month, turn)
            positive = risk[risk > 0]
            if len(positive):
                positives.append(positive)
    if not positives:
        raise ValueError("No existen delitos asociados en el entrenamiento")
    return float(np.quantile(np.concatenate(positives), 0.75))


def _labels(risk: np.ndarray, high_from: float) -> np.ndarray:
    result = np.zeros(len(risk), dtype=np.int8)
    result[risk > 0] = 1
    result[risk >= high_from] = 2
    return result


# Selecciona las filas de entrenamiento por clase manteniendo la semilla configurada.
def _sample(builder: FeatureBuilder, months: list[int], high_from: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(seed)
    capacity = builder.config.samples_per_class_per_month_turn
    x_parts, y_parts = [], []
    for month in months:
        for turn in range(len(TURNOS)):
            y = _labels(builder.target_risk(month, turn), high_from)
            selected = []
            for label in range(3):
                indices = np.flatnonzero(y == label)
                amount = min(capacity if label else capacity * 2, len(indices))
                if amount:
                    selected.append(random.choice(indices, amount, replace=False))
            if selected:
                indices = np.concatenate(selected)
                x = builder.features(month, turn)
                x_parts.append(x[indices])
                y_parts.append(y[indices])
        print(f"Muestras {MONTHS[month]}: {sum(len(y) for y in y_parts):,}", flush=True)
    result_x = np.vstack(x_parts)
    result_y = np.concatenate(y_parts)
    print(f"Clases muestreadas: {np.bincount(result_y, minlength=3).tolist()}", flush=True)
    if len(np.unique(result_y)) < 3:
        raise ValueError("Falta una clase de riesgo en el entrenamiento")
    return result_x, result_y


# Construye Random Forest y XGBoost con los hiperparámetros de la configuración.
def _models(config: TrainingConfig) -> dict[str, object]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=config.rf_trees,
            max_depth=14,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.random_state,
        ),
        "xgboost": XGBClassifier(
            n_estimators=config.xgb_trees,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            num_class=3,
            tree_method="hist",
            n_jobs=-1,
            random_state=config.random_state,
        ),
    }


def _fit(model, x: np.ndarray, y: np.ndarray) -> None:
    if isinstance(model, XGBClassifier):
        class_counts = np.bincount(y, minlength=3)
        weights = len(y) / (3 * class_counts[y])
        model.fit(x, y, sample_weight=weights)
    else:
        model.fit(x, y)


# Recupera las probabilidades de los meses evaluados sin mezclar sus etiquetas con el ajuste.
def _collect_probabilities(model, builder: FeatureBuilder, months: list[int], high_from: float) -> tuple[np.ndarray, np.ndarray]:
    truths, probabilities = [], []
    for month in months:
        for turn in range(len(TURNOS)):
            x = builder.features(month, turn)
            truth = _labels(builder.target_risk(month, turn), high_from)
            truths.append(truth)
            probabilities.append(model.predict_proba(x).astype(np.float32))
        print(f"Evaluado {MONTHS[month]}", flush=True)
    return np.concatenate(truths), np.vstack(probabilities)


def _best_binary_threshold(truth: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(truth, score)
    if len(thresholds) == 0:
        return 1.0
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros(len(thresholds)),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def _decision_thresholds(truth: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    high = _best_binary_threshold((truth == 2).astype(np.int8), probability[:, 2])
    remaining = probability[:, 2] < high
    medium = _best_binary_threshold(
        (truth[remaining] == 1).astype(np.int8), probability[remaining, 1]
    )
    return {"alto": high, "medio": medium}


def _predicted_labels(probability: np.ndarray, cutoffs: dict[str, float]) -> np.ndarray:
    predicted = np.zeros(len(probability), dtype=np.int8)
    predicted[probability[:, 1] >= cutoffs["medio"]] = 1
    predicted[probability[:, 2] >= cutoffs["alto"]] = 2
    return predicted


# Calcula métricas por clase y la matriz de confusión a partir de las probabilidades y etiquetas.
def _evaluate(
    truth: np.ndarray,
    probability: np.ndarray,
    months: list[int],
    cutoffs: dict[str, float],
) -> dict:
    predicted = _predicted_labels(probability, cutoffs)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1, 2])
    support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    recall = np.divide(true_positive, support, out=np.zeros(3), where=support > 0)
    precision = np.divide(true_positive, predicted_support, out=np.zeros(3), where=predicted_support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=(precision + recall) > 0)
    return {
        "periodo_inicio": MONTHS[months[0]],
        "periodo_fin": MONTHS[months[-1]],
        "registros_prueba": int(matrix.sum()),
        "accuracy": float(true_positive.sum() / matrix.sum()),
        "balanced_accuracy": float(recall.mean()),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "recall_riesgo_alto": float(recall[2]),
        "pr_auc_riesgo_alto": float(average_precision_score(truth == 2, probability[:, 2])),
        "matriz_confusion": matrix.tolist(),
        "soporte_clases": support.tolist(),
    }


def _write_metrics(output: Path, name: str, evaluation: dict) -> None:
    fields = [key for key in evaluation if key not in {"matriz_confusion", "soporte_clases"}]
    with (output / f"metricas_{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["modelo", *fields])
        writer.writeheader()
        writer.writerow({"modelo": name, **{key: evaluation[key] for key in fields}})
    pd.DataFrame(evaluation["matriz_confusion"], index=LABELS, columns=LABELS).to_csv(
        output / f"matriz_confusion_{name}.csv"
    )


# Exporta el índice y el nivel de riesgo por segmento para que los consulte la API.
def _write_predictions(
    output: Path, name: str, model, builder: FeatureBuilder, forecast: str,
    cutoffs: dict[str, float],
) -> None:
    month = MONTHS.index(forecast)
    scores = np.empty((len(TURNOS), builder.n_segments), dtype=np.float32)
    levels = np.empty((len(TURNOS), builder.n_segments), dtype=np.int8)
    for turn in range(len(TURNOS)):
        probabilities = model.predict_proba(builder.features(month, turn))
        scores[turn] = 0.5 * probabilities[:, 1] + probabilities[:, 2]
        levels[turn] = _predicted_labels(probabilities, cutoffs)
    result = builder.tramos[["tramo_id", "latitud", "longitud"]].copy()
    result["periodo_objetivo"] = forecast
    result["riesgo_score"] = scores.mean(axis=0)
    result["nivel_riesgo"] = np.select(
        [result["riesgo_score"] >= 0.66, result["riesgo_score"] >= 0.34],
        ["alto", "medio"], default="bajo",
    )
    for index, turn in enumerate(TURNOS):
        result[f"riesgo_score_{turn}"] = scores[index]
        result[f"nivel_riesgo_{turn}"] = np.asarray(LABELS)[levels[index]]
    result["modelo_usado"] = "Random Forest" if name == "random_forest" else "XGBoost"
    result.to_csv(output / f"predicciones_tramos_{name}.csv", index=False)
    if name == "random_forest":
        result.to_csv(output / "predicciones_tramos.csv", index=False)


# Entrena, valida y exporta los modelos junto con las métricas y su procedencia.
def entrenar_y_exportar(
    config: TrainingConfig,
    builder: FeatureBuilder,
    splits: dict[str, list[int]],
    source_audit: dict,
    spatial_audit: dict,
) -> dict:
    output = config.output
    threshold = _threshold(builder, splits["train"])
    print(f"Umbral de riesgo alto: {threshold:.6f}", flush=True)
    x_train, y_train = _sample(builder, splits["train"], threshold, config.random_state)
    preliminary = _models(config)
    assessments = {}
    for name, model in preliminary.items():
        print(f"Entrenando {name} para validación", flush=True)
        _fit(model, x_train, y_train)
        validation_truth, validation_probability = _collect_probabilities(
            model, builder, splits["validation"], threshold
        )
        cutoffs = _decision_thresholds(validation_truth, validation_probability)
        validation = _evaluate(validation_truth, validation_probability, splits["validation"], cutoffs)
        assessments[name] = {"validation": validation, "decision_thresholds": cutoffs}
        del validation_truth, validation_probability
    selected = max(
        assessments,
        key=lambda name: (
            assessments[name]["validation"]["f1_macro"],
            assessments[name]["validation"]["recall_riesgo_alto"],
            assessments[name]["validation"]["pr_auc_riesgo_alto"],
        ),
    )
    print(f"Modelo elegido por validación: {selected}", flush=True)
    for name, model in preliminary.items():
        test_truth, test_probability = _collect_probabilities(
            model, builder, splits["test"], threshold
        )
        test = _evaluate(
            test_truth, test_probability, splits["test"],
            assessments[name]["decision_thresholds"],
        )
        assessments[name]["test"] = test
        _write_metrics(output, name, test)
        del test_truth, test_probability
    # La prueba queda intacta hasta haber comparado ambos modelos. Después se
    # incorpora al ajuste final para producir predicciones operativas.
    final_months = sorted(set(splits["train"] + splits["validation"] + splits["test"]))
    x_final, y_final = _sample(builder, final_months, threshold, config.random_state)
    final_models = _models(config)
    for name, model in final_models.items():
        print(f"Entrenando versión final {name}", flush=True)
        _fit(model, x_final, y_final)
        cutoffs = assessments[name]["decision_thresholds"]
        _write_predictions(output, name, model, builder, config.forecast_period, cutoffs)
        metadata = {
            "modelo": "Random Forest" if name == "random_forest" else "XGBoost",
            "version_variables": "sidpol_v2_turno_mes",
            "variables": FEATURE_NAMES,
            "periodo_prediccion": config.forecast_period,
            "periodo_prueba": f"{MONTHS[splits['test'][0]]}..{MONTHS[splits['test'][-1]]}",
            "umbrales": {"bajo_max": 0.0, "alto_desde": threshold},
            "umbrales_decision": cutoffs,
            "tramo_turno": True,
            "entrenado_en": datetime.now().isoformat(),
        }
        joblib.dump({"pipeline": model, "metadata": metadata}, output / f"modelo_{name}.joblib")
        (output / ("metadata_modelo.json" if name == "random_forest" else "metadata_modelo_xgboost.json")).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    summary = {
        "source": str(config.source),
        "source_totals": source_audit["totals"],
        "spatial": spatial_audit,
        "splits": {name: [MONTHS[i] for i in months] for name, months in splits.items()},
        "coverage_threshold": config.min_monthly_geo_coverage,
        "history_months": config.history_months,
        "forecast_period": config.forecast_period,
        "feature_names": FEATURE_NAMES,
        "risk_high_threshold": threshold,
        "assessments": assessments,
        "selected_for_routing": selected,
        "final_training_months": [MONTHS[i] for i in final_months],
        "warning": "Métricas de prueba anteriores al reentrenamiento final; meses con baja cobertura espacial excluidos.",
    }
    (output / "resumen_entrenamiento.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "entrenamiento_completo.json").write_text(
        json.dumps({"selected_for_routing": selected, "forecast_period": config.forecast_period}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
