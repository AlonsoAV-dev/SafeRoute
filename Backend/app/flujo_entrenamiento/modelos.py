from __future__ import annotations

import json
import os
from dataclasses import dataclass

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


NIVELES_RIESGO = ["bajo", "medio", "alto"]
VARIABLES_NUMERICAS = [
    "latitud",
    "longitud",
    "frecuencia_delitos",
    "suma_riesgo_base",
    "promedio_riesgo",
    "peso_delito_promedio",
    "peso_delito_maximo",
    "delitos_graves_cercanos",
    "delitos_manana",
    "delitos_tarde",
    "delitos_noche",
    "delitos_madrugada",
    "cluster_kmeans",
]
VARIABLES_CATEGORICAS = ["distrito"]


@dataclass
class ResultadoModelos:
    tramos: pd.DataFrame
    metricas: pd.DataFrame
    modelos: dict[str, Pipeline]
    ganador: dict
    codificador_objetivo: LabelEncoder


def clasificar_clusters(tramos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    resultado = tramos.copy()
    variables = [
        "frecuencia_delitos",
        "suma_riesgo_base",
        "peso_delito_promedio",
        "peso_delito_maximo",
        "delitos_graves_cercanos",
        "riesgo_score",
        "delitos_noche",
        "delitos_madrugada",
    ]
    escalados = StandardScaler().fit_transform(resultado[variables])
    modelo = KMeans(n_clusters=3, random_state=42, n_init=20)
    resultado["cluster_kmeans"] = modelo.fit_predict(escalados)

    orden = (
        resultado.groupby("cluster_kmeans")["riesgo_score"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    interpretacion = {
        int(cluster): nivel for cluster, nivel in zip(orden, NIVELES_RIESGO)
    }
    resultado["nivel_riesgo_cluster"] = resultado["cluster_kmeans"].map(interpretacion)
    resultado["nivel_riesgo"] = pd.qcut(
        resultado["riesgo_score"].rank(method="first"),
        q=3,
        labels=NIVELES_RIESGO,
    ).astype(str)
    resumen = (
        resultado.groupby(["cluster_kmeans", "nivel_riesgo_cluster"], as_index=False)
        .agg(
            tramos=("id_segmento", "size"),
            riesgo_promedio=("riesgo_score", "mean"),
            frecuencia_promedio=("frecuencia_delitos", "mean"),
        )
        .sort_values("riesgo_promedio")
    )
    return resultado, resumen


def entrenar_y_comparar(tramos: pd.DataFrame) -> ResultadoModelos:
    datos = tramos.copy()
    objetivo = LabelEncoder()
    objetivo.fit(NIVELES_RIESGO)
    y = objetivo.transform(datos["nivel_riesgo"])
    x = datos[VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS]
    indices_train, indices_test = train_test_split(
        np.arange(len(datos)),
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    x_train, x_test = x.iloc[indices_train], x.iloc[indices_test]
    y_train, y_test = y[indices_train], y[indices_test]

    modelos = {
        "Random Forest": RandomForestClassifier(
            n_estimators=260,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=4,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=260,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=4,
        ),
    }
    pipelines = {}
    filas_metricas = []
    predicciones_completas = {}
    clase_alto = int(objetivo.transform(["alto"])[0])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for nombre, estimador in modelos.items():
        pipeline = Pipeline(
            [
                (
                    "preprocesamiento",
                    ColumnTransformer(
                        [
                            ("numericas", StandardScaler(), VARIABLES_NUMERICAS),
                            (
                                "categoricas",
                                OneHotEncoder(handle_unknown="ignore"),
                                VARIABLES_CATEGORICAS,
                            ),
                        ]
                    ),
                ),
                ("modelo", estimador),
            ]
        )
        pipeline.fit(x_train, y_train)
        prediccion_test = pipeline.predict(x_test)
        prediccion_train = pipeline.predict(x_train)
        probabilidades = pipeline.predict_proba(x_test)
        matriz = confusion_matrix(y_test, prediccion_test, labels=range(3))
        recall_clases = recall_score(
            y_test, prediccion_test, labels=range(3), average=None, zero_division=0
        )
        cv_scores = cross_val_score(
            pipeline, x, y, scoring="f1_macro", cv=cv, n_jobs=1
        )
        accuracy_test = accuracy_score(y_test, prediccion_test)
        accuracy_train = accuracy_score(y_train, prediccion_train)

        filas_metricas.append(
            {
                "modelo": nombre,
                "accuracy": round(float(accuracy_test), 6),
                "precision_macro": round(
                    float(precision_score(y_test, prediccion_test, average="macro", zero_division=0)),
                    6,
                ),
                "recall_macro": round(
                    float(recall_score(y_test, prediccion_test, average="macro", zero_division=0)),
                    6,
                ),
                "f1_macro": round(
                    float(f1_score(y_test, prediccion_test, average="macro", zero_division=0)),
                    6,
                ),
                "recall_riesgo_alto": round(float(recall_clases[clase_alto]), 6),
                "f1_cv_promedio": round(float(cv_scores.mean()), 6),
                "f1_cv_desviacion": round(float(cv_scores.std()), 6),
                "accuracy_train": round(float(accuracy_train), 6),
                "brecha_train_test": round(float(accuracy_train - accuracy_test), 6),
                "matriz_confusion": json.dumps(matriz.tolist()),
                "registros_train": int(len(x_train)),
                "registros_test": int(len(x_test)),
                "probabilidad_alto_promedio_test": round(
                    float(probabilidades[:, clase_alto].mean()), 6
                ),
            }
        )
        pipelines[nombre] = pipeline
        all_predictions = pipeline.predict(x)
        all_probabilities = pipeline.predict_proba(x)
        class_names = objetivo.inverse_transform(np.arange(len(objetivo.classes_)))
        class_scores = np.asarray(
            [{"bajo": 0.2, "medio": 0.55, "alto": 0.9}[name] for name in class_names]
        )
        slug = "random_forest" if nombre == "Random Forest" else "xgboost"
        predicciones_completas[nombre] = {
            "level": objetivo.inverse_transform(all_predictions),
            "high_probability": all_probabilities[:, clase_alto],
            "score": all_probabilities @ class_scores,
            "slug": slug,
        }

    metricas = pd.DataFrame(filas_metricas)
    ranking = metricas.sort_values(
        ["recall_riesgo_alto", "f1_macro", "f1_cv_promedio", "brecha_train_test"],
        ascending=[False, False, False, True],
    )
    fila_ganadora = ranking.iloc[0]
    nombre_ganador = str(fila_ganadora["modelo"])
    ganador = {
        "modelo": nombre_ganador,
        "criterio_seleccion": (
            "Mayor recall de riesgo alto; desempate por F1 macro, estabilidad "
            "en validación cruzada y menor brecha train-test."
        ),
        "recall_riesgo_alto": float(fila_ganadora["recall_riesgo_alto"]),
        "f1_macro": float(fila_ganadora["f1_macro"]),
        "f1_cv_promedio": float(fila_ganadora["f1_cv_promedio"]),
        "brecha_train_test": float(fila_ganadora["brecha_train_test"]),
    }

    for nombre, values in predicciones_completas.items():
        slug = values["slug"]
        datos[f"riesgo_predicho_{slug}"] = values["level"]
        datos[f"probabilidad_alto_{slug}"] = values["high_probability"].round(6)
        datos[f"score_modelo_{slug}"] = values["score"].round(6)

    winner_values = predicciones_completas[nombre_ganador]
    datos["riesgo_predicho"] = winner_values["level"]
    datos["probabilidad_riesgo_alto"] = winner_values["high_probability"].round(6)
    datos["score_modelo"] = winner_values["score"].round(6)
    datos["modelo_usado"] = nombre_ganador
    return ResultadoModelos(datos, metricas, pipelines, ganador, objetivo)
