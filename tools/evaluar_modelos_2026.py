from __future__ import annotations

import json
import sys
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

from app.flujo_entrenamiento.limpieza import cargar_fuente, limpiar_delitos
from app.flujo_entrenamiento.modelos import (
    CODIGO_NIVEL,
    NIVELES_RIESGO,
    _probabilidades_completas,
    _submuestrear_clase_baja,
    calcular_umbrales,
    clasificar_riesgo,
)
from app.flujo_entrenamiento.red_vial import cargar_o_descargar_grafo, extraer_tramos
from app.flujo_entrenamiento.riesgo import (
    VARIABLES_MODELO,
    agregar_riesgo_base,
    crear_panel_temporal,
)


DATA = BACKEND / "data"
OUT = DATA / "procesados" / "evaluacion_2026"
GRAPH_PATH = DATA / "red_vial_lima.graphml"
FUENTE_2025 = DATA / "DELITOS TOTAL.csv"
FUENTE_2026 = DATA / "DELITOS TOTAL-2026.csv"
TRAIN_PERIODS = [f"2025-{month:02d}" for month in range(4, 13)]
EXTERNAL_PERIODS = [f"2026-{month:02d}" for month in range(1, 6)]
RADIO_BUFFER_M = 200.0
VENTANA_MESES = 3


def cargar_delitos_integrados() -> pd.DataFrame:
    frames = []
    for fuente in (FUENTE_2025, FUENTE_2026):
        raw = cargar_fuente(fuente)
        clean = agregar_riesgo_base(limpiar_delitos(raw))
        clean["archivo_origen"] = fuente.name
        frames.append(clean)
    delitos = pd.concat(frames, ignore_index=True)
    delitos = delitos.sort_values("fecha").drop_duplicates("id_hecho", keep="last")
    delitos = delitos.loc[delitos["periodo"].between("2025-01", "2026-05")].copy()
    return delitos.reset_index(drop=True)


def cargar_o_crear_panel(delitos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    panel_path = OUT / "panel_tramo_temporal_2025_2026.csv"
    futuro_path = OUT / "datos_prediccion_junio_2026.csv"
    metadata_path = OUT / "metadata_panel_2025_2026.json"
    if panel_path.exists() and futuro_path.exists() and metadata_path.exists():
        panel = pd.read_csv(panel_path)
        futuro = pd.read_csv(futuro_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return panel, futuro, metadata

    graph = cargar_o_descargar_grafo(GRAPH_PATH)
    tramos, tramos_geo = extraer_tramos(graph)
    tramos.to_csv(OUT / "tramos_osm.csv", index=False)
    panel, futuro, metadata = crear_panel_temporal(
        delitos,
        tramos,
        tramos_geo,
        ventana_meses=VENTANA_MESES,
        radio_m=RADIO_BUFFER_M,
    )
    panel.to_csv(panel_path, index=False)
    futuro.to_csv(futuro_path, index=False)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return panel, futuro, metadata


def crear_pipeline(nombre: str, random_state: int) -> tuple[Pipeline, bool]:
    if nombre == "Random Forest":
        return (
            Pipeline(
                [
                    ("imputacion", SimpleImputer(strategy="median")),
                    (
                        "modelo",
                        RandomForestClassifier(
                            n_estimators=160,
                            max_depth=14,
                            min_samples_leaf=2,
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            False,
        )
    if nombre == "XGBoost":
        return (
            Pipeline(
                [
                    ("imputacion", SimpleImputer(strategy="median")),
                    (
                        "modelo",
                        XGBClassifier(
                            n_estimators=260,
                            max_depth=6,
                            learning_rate=0.06,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            objective="multi:softprob",
                            num_class=3,
                            eval_metric="mlogloss",
                            tree_method="hist",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            True,
        )
    raise ValueError(f"Modelo no soportado: {nombre}")


def metricas(nombre: str, periodo: str, y_true, y_pred, proba) -> dict:
    binario_alto = (y_true == CODIGO_NIVEL["alto"]).astype(int)
    pr_auc_alto = (
        average_precision_score(binario_alto, proba[:, CODIGO_NIVEL["alto"]])
        if binario_alto.sum() > 0
        else 0.0
    )
    return {
        "modelo": nombre,
        "periodo": periodo,
        "registros": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_riesgo_alto": float(
            recall_score(
                y_true,
                y_pred,
                labels=[CODIGO_NIVEL["alto"]],
                average="macro",
                zero_division=0,
            )
        ),
        "pr_auc_riesgo_alto": float(pr_auc_alto),
    }


def entrenar_y_evaluar(panel: pd.DataFrame, nombre: str, random_state: int = 42) -> dict:
    train = panel.loc[panel["periodo_objetivo"].isin(TRAIN_PERIODS)].copy()
    test = panel.loc[panel["periodo_objetivo"].isin(EXTERNAL_PERIODS)].copy()
    if train.empty or test.empty:
        raise ValueError("No hay suficientes filas para entrenar o evaluar.")

    umbrales = calcular_umbrales(train["riesgo_bruto_futuro"])
    y_train_text = clasificar_riesgo(train["riesgo_bruto_futuro"], umbrales)
    y_test_text = clasificar_riesgo(test["riesgo_bruto_futuro"], umbrales)
    y_train = y_train_text.map(CODIGO_NIVEL).to_numpy()
    y_test = y_test_text.map(CODIGO_NIVEL).to_numpy()

    indices_modelo = _submuestrear_clase_baja(y_train, random_state)
    train_model = train.iloc[indices_modelo]
    y_train_model = y_train[indices_modelo]

    modelo, usar_pesos = crear_pipeline(nombre, random_state)
    fit_params = {}
    if usar_pesos:
        fit_params["modelo__sample_weight"] = compute_sample_weight(
            class_weight="balanced",
            y=y_train_model,
        )
    modelo.fit(train_model[VARIABLES_MODELO], y_train_model, **fit_params)

    y_pred = modelo.predict(test[VARIABLES_MODELO])
    proba = _probabilidades_completas(modelo, test[VARIABLES_MODELO])
    global_metrics = metricas(nombre, "2026-01_a_2026-05", y_test, y_pred, proba)
    global_metrics["registros_entrenamiento_total"] = int(len(train))
    global_metrics["registros_entrenamiento_modelo"] = int(len(train_model))
    global_metrics["umbral_alto"] = float(umbrales["alto_desde"])

    monthly = []
    for periodo in EXTERNAL_PERIODS:
        mask = test["periodo_objetivo"].eq(periodo).to_numpy()
        monthly.append(metricas(nombre, periodo, y_test[mask], y_pred[mask], proba[mask]))

    report = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=NIVELES_RIESGO,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    predictions = test[["tramo_id", "periodo_objetivo", "latitud", "longitud", "riesgo_bruto_futuro"]].copy()
    predictions["nivel_real"] = y_test_text.to_numpy()
    predictions["nivel_predicho"] = [NIVELES_RIESGO[int(value)] for value in y_pred]
    predictions["prob_bajo"] = proba[:, 0]
    predictions["prob_medio"] = proba[:, 1]
    predictions["prob_alto"] = proba[:, 2]
    predictions["riesgo_score"] = (0.5 * predictions["prob_medio"] + predictions["prob_alto"]).clip(0, 1)

    slug = nombre.lower().replace(" ", "_")
    pd.DataFrame([global_metrics]).to_csv(OUT / f"metricas_externas_{slug}.csv", index=False)
    pd.DataFrame(monthly).to_csv(OUT / f"metricas_externas_por_mes_{slug}.csv", index=False)
    report.to_csv(OUT / f"classification_report_externo_{slug}.csv")
    pd.DataFrame(matrix, index=NIVELES_RIESGO, columns=NIVELES_RIESGO).to_csv(
        OUT / f"matriz_confusion_externa_{slug}.csv"
    )
    predictions.to_csv(OUT / f"predicciones_externas_2026_{slug}.csv", index=False)
    joblib.dump(
        {"pipeline": modelo, "umbrales": umbrales, "variables": VARIABLES_MODELO},
        OUT / f"modelo_entrenado_2025_{slug}.joblib",
    )
    return {
        "modelo": nombre,
        "metricas": global_metrics,
        "mensual": monthly,
        "reporte": report,
        "matriz": matrix,
        "predicciones": predictions,
        "pipeline": modelo,
    }


def graficar(resultados: list[dict]) -> None:
    graph_dir = OUT / "graficos"
    graph_dir.mkdir(parents=True, exist_ok=True)
    comparacion = pd.DataFrame([r["metricas"] for r in resultados])
    comparacion.to_csv(OUT / "comparacion_modelos_evaluacion_externa_2026.csv", index=False)

    metric_cols = ["accuracy", "balanced_accuracy", "f1_macro", "recall_riesgo_alto", "pr_auc_riesgo_alto"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(metric_cols))
    width = 0.36
    for offset, (_, row) in zip((-width / 2, width / 2), comparacion.iterrows()):
        ax.bar(x + offset, [row[col] for col in metric_cols], width, label=row["modelo"])
    ax.set_xticks(x, ["Accuracy", "Bal. acc.", "F1 macro", "Recall alto", "PR-AUC alto"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Puntuación")
    ax.set_title("Evaluación externa 2026: comparación de modelos", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(graph_dir / "01_comparacion_modelos_2026.png", dpi=180)
    plt.close(fig)

    monthly = pd.concat([pd.DataFrame(r["mensual"]) for r in resultados], ignore_index=True)
    monthly.to_csv(OUT / "metricas_externas_2026_por_mes.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for model, group in monthly.groupby("modelo"):
        group = group.sort_values("periodo")
        ax.plot(group["periodo"], group["f1_macro"], marker="o", label=f"F1 {model}")
        ax.plot(group["periodo"], group["recall_riesgo_alto"], marker="s", linestyle="--", label=f"Recall alto {model}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Puntuación")
    ax.set_title("Desempeño mensual en evaluación futura 2026", weight="bold")
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(graph_dir / "02_metricas_mensuales_2026.png", dpi=180)
    plt.close(fig)

    for result in resultados:
        slug = result["modelo"].lower().replace(" ", "_")
        matrix = result["matriz"]
        fig, ax = plt.subplots(figsize=(6.5, 5.8))
        image = ax.imshow(matrix, cmap="YlOrRd")
        threshold = matrix.max() / 2 if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{int(matrix[i, j]):,}",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > threshold else "#111827",
                    weight="bold",
                    fontsize=9,
                )
        ax.set_xticks(range(3), [n.capitalize() for n in NIVELES_RIESGO])
        ax.set_yticks(range(3), [n.capitalize() for n in NIVELES_RIESGO])
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
        ax.set_title(f"Matriz de confusión externa: {result['modelo']}", weight="bold")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(graph_dir / f"03_matriz_confusion_{slug}.png", dpi=180)
        plt.close(fig)

        importances = result["pipeline"].named_steps["modelo"].feature_importances_
        series = pd.Series(importances, index=VARIABLES_MODELO).nlargest(15).sort_values()
        fig, ax = plt.subplots(figsize=(9, 6.5))
        ax.barh(series.index, series.values, color="#0f766e")
        ax.set_xlabel("Importancia")
        ax.set_title(f"Importancia de variables: {result['modelo']}", weight="bold")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(graph_dir / f"04_importancia_variables_{slug}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    delitos = cargar_delitos_integrados()
    delitos.to_csv(OUT / "delitos_2025_2026_limpios.csv", index=False)
    resumen_periodos = delitos.groupby("periodo").size().reset_index(name="delitos")
    resumen_periodos.to_csv(OUT / "resumen_delitos_por_periodo.csv", index=False)
    panel, futuro, metadata = cargar_o_crear_panel(delitos)
    resultados = [
        entrenar_y_evaluar(panel, "Random Forest"),
        entrenar_y_evaluar(panel, "XGBoost"),
    ]
    graficar(resultados)
    resumen = {
        "delitos_limpios_integrados": int(len(delitos)),
        "filas_panel": int(len(panel)),
        "filas_prediccion_junio_2026": int(len(futuro)),
        "periodos_entrenamiento": TRAIN_PERIODS,
        "periodos_evaluacion_externa": EXTERNAL_PERIODS,
        "metadata_panel": metadata,
        "metricas": [r["metricas"] for r in resultados],
    }
    (OUT / "resumen_evaluacion_2026.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
