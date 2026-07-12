from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.flujo_entrenamiento.modelos import (
    CODIGO_NIVEL,
    NIVELES_RIESGO,
    _probabilidades_completas,
    _submuestrear_clase_baja,
    calcular_umbrales,
    clasificar_riesgo,
)
from app.flujo_entrenamiento.riesgo import VARIABLES_MODELO


BASE = ROOT / "Backend" / "data" / "procesados" / "evaluacion_2026"
PANEL_PATH = BASE / "panel_tramo_temporal_2025_2026.csv"
OUT = BASE / "tuning"
TUNE_TRAIN_PERIODS = [f"2025-{month:02d}" for month in range(4, 12)]
VALIDATION_PERIODS = ["2025-12"]
FINAL_TRAIN_PERIODS = [f"2025-{month:02d}" for month in range(4, 13)]
EXTERNAL_PERIODS = [f"2026-{month:02d}" for month in range(1, 6)]
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelConfig:
    name: str
    slug: str
    params: dict
    sample_weight_strength: float = 0.0


def make_pipeline(config: ModelConfig) -> Pipeline:
    if config.name == "Random Forest":
        estimator = RandomForestClassifier(
            **config.params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif config.name == "XGBoost":
        estimator = XGBClassifier(
            **config.params,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Modelo no soportado: {config.name}")
    return Pipeline(
        [
            ("imputacion", SimpleImputer(strategy="median")),
            ("modelo", estimator),
        ]
    )


def load_panel() -> pd.DataFrame:
    usecols = ["tramo_id", "periodo_objetivo", *VARIABLES_MODELO, "riesgo_bruto_futuro"]
    panel = pd.read_csv(PANEL_PATH, usecols=usecols)
    panel["periodo_objetivo"] = panel["periodo_objetivo"].astype(str)
    return panel


def prepare_xy(frame: pd.DataFrame, umbrales: dict) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    y_text = clasificar_riesgo(frame["riesgo_bruto_futuro"], umbrales)
    y = y_text.map(CODIGO_NIVEL).to_numpy()
    return frame[VARIABLES_MODELO], y, y_text


def fit_model(config: ModelConfig, train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    indices = _submuestrear_clase_baja(y_train, RANDOM_STATE)
    x_model = train.iloc[indices][VARIABLES_MODELO]
    y_model = y_train[indices]
    pipeline = make_pipeline(config)
    fit_params = {}
    if config.sample_weight_strength > 0:
        weights = compute_sample_weight(class_weight="balanced", y=y_model)
        if config.sample_weight_strength != 1.0:
            weights = np.power(weights, config.sample_weight_strength)
        fit_params["modelo__sample_weight"] = weights
    pipeline.fit(x_model, y_model, **fit_params)
    return pipeline


def predict_with_rule(proba: np.ndarray, rule: dict) -> np.ndarray:
    if rule["type"] == "argmax":
        return proba.argmax(axis=1)
    pred = np.zeros(len(proba), dtype=int)
    pred[proba[:, 1] >= rule["medium_threshold"]] = 1
    pred[proba[:, 2] >= rule["high_threshold"]] = 2
    return pred


def tune_decision_rule(y_true: np.ndarray, proba: np.ndarray) -> tuple[dict, dict]:
    candidates = [{"type": "argmax"}]
    for high_threshold in np.arange(0.35, 0.71, 0.05):
        for medium_threshold in np.arange(0.30, 0.61, 0.05):
            candidates.append(
                {
                    "type": "threshold",
                    "high_threshold": round(float(high_threshold), 2),
                    "medium_threshold": round(float(medium_threshold), 2),
                }
            )

    best_rule = candidates[0]
    best_metrics = evaluate_predictions("validation", "validation", y_true, proba.argmax(axis=1), proba)
    best_score = _selection_score(best_metrics)
    for rule in candidates[1:]:
        pred = predict_with_rule(proba, rule)
        metrics = evaluate_predictions("validation", "validation", y_true, pred, proba)
        score = _selection_score(metrics)
        if score > best_score:
            best_score = score
            best_rule = rule
            best_metrics = metrics
    return best_rule, best_metrics


def _selection_score(metrics: dict) -> float:
    # Prioriza equilibrio general y mantiene peso alto para detectar riesgo alto.
    return (
        0.45 * metrics["f1_macro"]
        + 0.25 * metrics["balanced_accuracy"]
        + 0.20 * metrics["recall_riesgo_alto"]
        + 0.10 * metrics["pr_auc_riesgo_alto"]
    )


def evaluate_predictions(model: str, period: str, y_true: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> dict:
    binario_alto = (y_true == CODIGO_NIVEL["alto"]).astype(int)
    pr_auc_alto = (
        average_precision_score(binario_alto, proba[:, CODIGO_NIVEL["alto"]])
        if binario_alto.sum() > 0
        else 0.0
    )
    return {
        "modelo": model,
        "periodo": period,
        "registros": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_macro": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "recall_riesgo_alto": float(
            recall_score(
                y_true,
                pred,
                labels=[CODIGO_NIVEL["alto"]],
                average="macro",
                zero_division=0,
            )
        ),
        "pr_auc_riesgo_alto": float(pr_auc_alto),
    }


def configs() -> list[ModelConfig]:
    return [
        ModelConfig(
            "Random Forest",
            "rf_base",
            {
                "n_estimators": 160,
                "max_depth": 14,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
            },
        ),
        ModelConfig(
            "Random Forest",
            "rf_deep",
            {
                "n_estimators": 220,
                "max_depth": 18,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
            },
        ),
        ModelConfig(
            "Random Forest",
            "rf_regularized",
            {
                "n_estimators": 220,
                "max_depth": 12,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
            },
        ),
        ModelConfig(
            "XGBoost",
            "xgb_base",
            {
                "n_estimators": 260,
                "max_depth": 6,
                "learning_rate": 0.06,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
            },
            sample_weight_strength=1.0,
        ),
        ModelConfig(
            "XGBoost",
            "xgb_regularized",
            {
                "n_estimators": 420,
                "max_depth": 5,
                "learning_rate": 0.04,
                "subsample": 0.90,
                "colsample_bytree": 0.80,
                "min_child_weight": 3,
                "reg_lambda": 2.0,
                "gamma": 0.05,
            },
            sample_weight_strength=0.85,
        ),
        ModelConfig(
            "XGBoost",
            "xgb_shallow",
            {
                "n_estimators": 520,
                "max_depth": 4,
                "learning_rate": 0.035,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
                "min_child_weight": 2,
                "reg_lambda": 1.5,
            },
            sample_weight_strength=0.75,
        ),
    ]


def run_tuning(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel.loc[panel["periodo_objetivo"].isin(TUNE_TRAIN_PERIODS)].copy()
    val = panel.loc[panel["periodo_objetivo"].isin(VALIDATION_PERIODS)].copy()
    umbrales = calcular_umbrales(train["riesgo_bruto_futuro"])
    _, y_train, _ = prepare_xy(train, umbrales)
    x_val, y_val, _ = prepare_xy(val, umbrales)
    rows = []
    for config in configs():
        print(f"Entrenando validacion: {config.slug}")
        model = fit_model(config, train, y_train)
        proba = _probabilidades_completas(model, x_val)
        argmax_pred = proba.argmax(axis=1)
        argmax_metrics = evaluate_predictions(config.name, "2025-12", y_val, argmax_pred, proba)
        rule, tuned_metrics = tune_decision_rule(y_val, proba)
        for metrics, rule_name, selected_rule in [
            (argmax_metrics, "argmax", {"type": "argmax"}),
            (tuned_metrics, "tuned_thresholds", rule),
        ]:
            rows.append(
                {
                    "config_slug": config.slug,
                    "modelo": config.name,
                    "decision_rule_name": rule_name,
                    "decision_rule": json.dumps(selected_rule),
                    "selection_score": _selection_score(metrics),
                    "params": json.dumps(config.params),
                    "sample_weight_strength": config.sample_weight_strength,
                    **{k: v for k, v in metrics.items() if k not in {"modelo", "periodo"}},
                }
            )
    results = pd.DataFrame(rows).sort_values(
        ["selection_score", "f1_macro", "recall_riesgo_alto"],
        ascending=False,
    )
    results.to_csv(OUT / "resultados_validacion_2025_tuning.csv", index=False)
    return results


def final_evaluation(panel: pd.DataFrame, tuning_results: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    for model_name in ["Random Forest", "XGBoost"]:
        selected_rows.append(tuning_results.loc[tuning_results["modelo"].eq(model_name)].iloc[0])

    train = panel.loc[panel["periodo_objetivo"].isin(FINAL_TRAIN_PERIODS)].copy()
    test = panel.loc[panel["periodo_objetivo"].isin(EXTERNAL_PERIODS)].copy()
    umbrales = calcular_umbrales(train["riesgo_bruto_futuro"])
    _, y_train, _ = prepare_xy(train, umbrales)
    x_test, y_test, y_test_text = prepare_xy(test, umbrales)
    final_rows = []
    for selected in selected_rows:
        config = next(c for c in configs() if c.slug == selected["config_slug"])
        rule = json.loads(selected["decision_rule"])
        print(f"Reentrenando final: {config.slug} con regla {rule}")
        model = fit_model(config, train, y_train)
        proba = _probabilidades_completas(model, x_test)
        pred = predict_with_rule(proba, rule)
        metrics = evaluate_predictions(config.name, "2026-01_a_2026-05", y_test, pred, proba)
        metrics.update(
            {
                "config_slug": config.slug,
                "decision_rule_name": selected["decision_rule_name"],
                "decision_rule": selected["decision_rule"],
                "params": selected["params"],
                "sample_weight_strength": float(selected["sample_weight_strength"]),
                "umbral_alto": float(umbrales["alto_desde"]),
            }
        )
        final_rows.append(metrics)
        slug = config.slug
        report = pd.DataFrame(
            classification_report(
                y_test,
                pred,
                labels=[0, 1, 2],
                target_names=NIVELES_RIESGO,
                output_dict=True,
                zero_division=0,
            )
        ).transpose()
        matrix = confusion_matrix(y_test, pred, labels=[0, 1, 2])
        report.to_csv(OUT / f"classification_report_externo_ajustado_{slug}.csv")
        pd.DataFrame(matrix, index=NIVELES_RIESGO, columns=NIVELES_RIESGO).to_csv(
            OUT / f"matriz_confusion_externa_ajustada_{slug}.csv"
        )
        predictions = test[["tramo_id", "periodo_objetivo", "latitud", "longitud", "riesgo_bruto_futuro"]].copy()
        predictions["nivel_real"] = y_test_text.to_numpy()
        predictions["nivel_predicho"] = [NIVELES_RIESGO[int(value)] for value in pred]
        predictions["prob_bajo"] = proba[:, 0]
        predictions["prob_medio"] = proba[:, 1]
        predictions["prob_alto"] = proba[:, 2]
        predictions["riesgo_score"] = (0.5 * predictions["prob_medio"] + predictions["prob_alto"]).clip(0, 1)
        predictions.to_csv(OUT / f"predicciones_externas_2026_ajustadas_{slug}.csv", index=False)
        joblib.dump(
            {
                "pipeline": model,
                "umbrales": umbrales,
                "variables": VARIABLES_MODELO,
                "decision_rule": rule,
                "config": config.params,
            },
            OUT / f"modelo_ajustado_2025_{slug}.joblib",
        )
    final = pd.DataFrame(final_rows)
    final.to_csv(OUT / "comparacion_modelos_ajustados_evaluacion_externa_2026.csv", index=False)
    return final


def plot_final(final: pd.DataFrame) -> None:
    graph_dir = OUT / "graficos"
    graph_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["accuracy", "balanced_accuracy", "f1_macro", "recall_riesgo_alto", "pr_auc_riesgo_alto"]
    labels = ["Accuracy", "Bal. acc.", "F1 macro", "Recall alto", "PR-AUC alto"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(metrics))
    width = 0.36
    colors = {"Random Forest": "#2563eb", "XGBoost": "#dc2626"}
    for offset, (_, row) in zip((-width / 2, width / 2), final.iterrows()):
        values = [row[m] for m in metrics]
        bars = ax.bar(x + offset, values, width, label=f"{row['modelo']} ({row['config_slug']})", color=colors[row["modelo"]])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Puntuación")
    ax.set_title("Modelos ajustados en evaluación futura 2026", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(graph_dir / "01_comparacion_modelos_ajustados_2026.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    tuning_results = run_tuning(panel)
    final = final_evaluation(panel, tuning_results)
    plot_final(final)
    summary = {
        "validacion_2025_top": tuning_results.head(8).to_dict(orient="records"),
        "evaluacion_externa_2026": final.to_dict(orient="records"),
    }
    (OUT / "resumen_tuning_2026.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
