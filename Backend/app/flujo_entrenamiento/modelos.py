from __future__ import annotations

from dataclasses import dataclass

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

from app.flujo_entrenamiento.riesgo import VARIABLES_MODELO


NIVELES_RIESGO = ["bajo", "medio", "alto"]
CODIGO_NIVEL = {nivel: indice for indice, nivel in enumerate(NIVELES_RIESGO)}


@dataclass
class ResultadoRandomForest:
    modelo: Pipeline
    metricas: dict
    reporte: pd.DataFrame
    matriz_confusion: np.ndarray
    predicciones_futuras: pd.DataFrame
    umbrales: dict
    periodo_prueba: str


def entrenar_random_forest(
    panel: pd.DataFrame,
    futuro: pd.DataFrame,
    random_state: int = 42,
) -> ResultadoRandomForest:
    """Entrena con periodos anteriores y reserva el último mes para prueba."""
    periodos = sorted(panel["periodo_objetivo"].unique())
    if len(periodos) < 2:
        raise ValueError("Se requieren al menos dos periodos objetivo para evaluar el modelo.")
    periodo_prueba = periodos[-1]
    train = panel.loc[panel["periodo_objetivo"].ne(periodo_prueba)].copy()
    test = panel.loc[panel["periodo_objetivo"].eq(periodo_prueba)].copy()
    umbrales = calcular_umbrales(train["riesgo_bruto_futuro"])
    y_train_texto = clasificar_riesgo(train["riesgo_bruto_futuro"], umbrales)
    y_test_texto = clasificar_riesgo(test["riesgo_bruto_futuro"], umbrales)
    y_train = y_train_texto.map(CODIGO_NIVEL).to_numpy()
    y_test = y_test_texto.map(CODIGO_NIVEL).to_numpy()
    indices_modelo = _submuestrear_clase_baja(y_train, random_state)
    train_modelo = train.iloc[indices_modelo]
    y_train_modelo = y_train[indices_modelo]

    modelo = Pipeline(
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
    )
    modelo.fit(train_modelo[VARIABLES_MODELO], y_train_modelo)
    prediccion = modelo.predict(test[VARIABLES_MODELO])
    probabilidades = _probabilidades_completas(modelo, test[VARIABLES_MODELO])
    matriz = confusion_matrix(y_test, prediccion, labels=[0, 1, 2])
    reporte = pd.DataFrame(
        classification_report(
            y_test,
            prediccion,
            labels=[0, 1, 2],
            target_names=NIVELES_RIESGO,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    binario_alto = (y_test == CODIGO_NIVEL["alto"]).astype(int)
    pr_auc_alto = (
        average_precision_score(binario_alto, probabilidades[:, 2])
        if binario_alto.sum() > 0
        else 0.0
    )
    metricas = {
        "modelo": "Random Forest",
        "periodo_prueba": periodo_prueba,
        "registros_entrenamiento_total": int(len(train)),
        "registros_entrenamiento_modelo": int(len(train_modelo)),
        "registros_prueba": int(len(test)),
        "accuracy": float(accuracy_score(y_test, prediccion)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediccion)),
        "precision_macro": float(
            precision_score(y_test, prediccion, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, prediccion, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_test, prediccion, average="macro", zero_division=0)),
        "recall_riesgo_alto": float(
            recall_score(
                y_test,
                prediccion,
                labels=[CODIGO_NIVEL["alto"]],
                average="macro",
                zero_division=0,
            )
        ),
        "pr_auc_riesgo_alto": float(pr_auc_alto),
    }

    probabilidades_futuras = _probabilidades_completas(modelo, futuro[VARIABLES_MODELO])
    codigo_futuro = probabilidades_futuras.argmax(axis=1)
    predicciones = futuro[["tramo_id", "periodo_objetivo", "latitud", "longitud"]].copy()
    predicciones["prob_bajo"] = probabilidades_futuras[:, 0]
    predicciones["prob_medio"] = probabilidades_futuras[:, 1]
    predicciones["prob_alto"] = probabilidades_futuras[:, 2]
    predicciones["riesgo_score"] = (
        0.5 * predicciones["prob_medio"] + predicciones["prob_alto"]
    ).clip(0, 1)
    predicciones["nivel_riesgo"] = [NIVELES_RIESGO[codigo] for codigo in codigo_futuro]
    predicciones["modelo_usado"] = "Random Forest"
    return ResultadoRandomForest(
        modelo=modelo,
        metricas=metricas,
        reporte=reporte,
        matriz_confusion=matriz,
        predicciones_futuras=predicciones,
        umbrales=umbrales,
        periodo_prueba=periodo_prueba,
    )


def calcular_umbrales(valores: pd.Series) -> dict:
    """Separa gravedad positiva en riesgo medio y alto sin fragmentar los ceros."""
    positivos = valores.loc[valores > 0]
    if positivos.empty:
        raise ValueError("El periodo de entrenamiento no contiene delitos futuros asociados.")
    umbral_alto = float(positivos.quantile(0.75))
    return {"bajo_max": 0.0, "alto_desde": max(umbral_alto, 1e-9)}


def clasificar_riesgo(valores: pd.Series, umbrales: dict) -> pd.Series:
    condiciones = [
        valores.le(umbrales["bajo_max"]),
        valores.lt(umbrales["alto_desde"]),
    ]
    return pd.Series(
        np.select(condiciones, ["bajo", "medio"], default="alto"),
        index=valores.index,
    )


def _probabilidades_completas(modelo: Pipeline, x: pd.DataFrame) -> np.ndarray:
    parciales = modelo.predict_proba(x)
    clases = modelo.named_steps["modelo"].classes_
    completas = np.zeros((len(x), 3), dtype=float)
    for posicion, clase in enumerate(clases):
        completas[:, int(clase)] = parciales[:, posicion]
    return completas


def _submuestrear_clase_baja(y: np.ndarray, random_state: int) -> np.ndarray:
    """Conserva todas las clases minoritarias y una muestra reproducible de bajo."""
    indices_bajo = np.flatnonzero(y == CODIGO_NIVEL["bajo"])
    indices_relevantes = np.flatnonzero(y != CODIGO_NIVEL["bajo"])
    maximo_bajo = min(max(50_000, len(indices_relevantes) * 2), 250_000)
    if len(indices_bajo) <= maximo_bajo:
        return np.arange(len(y))
    generador = np.random.default_rng(random_state)
    muestra_bajo = generador.choice(indices_bajo, size=maximo_bajo, replace=False)
    return np.sort(np.concatenate([indices_relevantes, muestra_bajo]))
