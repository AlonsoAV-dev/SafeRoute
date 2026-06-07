from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from app.flujo_entrenamiento.limpieza import cargar_fuente, limpiar_delitos
from app.flujo_entrenamiento.modelos import clasificar_clusters, entrenar_y_comparar
from app.flujo_entrenamiento.riesgo import agregar_riesgo_base, crear_riesgo_por_tramo


BACKEND_DIR = Path(__file__).resolve().parents[2]


def ejecutar_flujo(
    fuente: Path,
    salida: Path,
    muestra: int = 0,
    tamano_segmento_m: int = 200,
) -> dict:
    salida.mkdir(parents=True, exist_ok=True)
    originales = cargar_fuente(fuente)
    originales.to_csv(salida / "delitos_original.csv", index=False)

    limpios = limpiar_delitos(originales)
    if muestra and len(limpios) > muestra:
        limpios = limpios.sample(muestra, random_state=42).reset_index(drop=True)
    if len(limpios) < 100:
        raise ValueError("Se necesitan al menos 100 delitos válidos para entrenar.")
    limpios.to_csv(salida / "delitos_limpios.csv", index=False)

    con_riesgo = agregar_riesgo_base(limpios)
    con_riesgo.to_csv(salida / "delitos_con_riesgo.csv", index=False)
    delitos_segmentados, tramos = crear_riesgo_por_tramo(
        con_riesgo, tamano_segmento_m=tamano_segmento_m
    )
    delitos_segmentados.to_csv(salida / "delitos_asignados_a_tramo.csv", index=False)

    tramos, clusters = clasificar_clusters(tramos)
    tramos.to_csv(salida / "riesgo_por_tramo.csv", index=False)
    clusters.to_csv(salida / "clusters_riesgo.csv", index=False)

    resultados = entrenar_y_comparar(tramos)
    resultados.metricas.to_csv(salida / "metricas_modelos.csv", index=False)
    predicciones = resultados.tramos[
        [
            "id_segmento",
            "nodo_origen",
            "nodo_destino",
            "latitud",
            "longitud",
            "distancia_m",
            "geometria",
            "riesgo_score",
            "nivel_riesgo",
            "riesgo_predicho",
            "probabilidad_riesgo_alto",
            "score_modelo",
            "riesgo_predicho_random_forest",
            "probabilidad_alto_random_forest",
            "score_modelo_random_forest",
            "riesgo_predicho_xgboost",
            "probabilidad_alto_xgboost",
            "score_modelo_xgboost",
            "modelo_usado",
        ]
    ]
    predicciones.to_csv(salida / "predicciones_riesgo.csv", index=False)
    pd.DataFrame(
        columns=[
            "id_ruta",
            "fecha_calculo",
            "origen",
            "destino",
            "distancia_total_m",
            "tiempo_estimado_min",
            "riesgo_promedio",
            "nivel_riesgo",
            "tramos_criticos",
            "modelo_usado",
        ]
    ).to_csv(salida / "rutas_calculadas.csv", index=False)

    with (salida / "modelo_ganador.json").open("w", encoding="utf-8") as archivo:
        json.dump(resultados.ganador, archivo, ensure_ascii=False, indent=2)
    trained_at = datetime.now().isoformat()
    for model_name, pipeline in resultados.modelos.items():
        filename = (
            "modelo_random_forest.joblib"
            if model_name == "Random Forest"
            else "modelo_xgboost.joblib"
        )
        joblib.dump(
            {
                "pipeline": pipeline,
                "codificador_objetivo": resultados.codificador_objetivo,
                "entrenado_en": trained_at,
            },
            salida / filename,
        )
    joblib.dump(
        {
            "pipeline": resultados.modelos[resultados.ganador["modelo"]],
            "codificador_objetivo": resultados.codificador_objetivo,
            "entrenado_en": trained_at,
        },
        salida / "modelo_ganador.joblib",
    )

    resumen = {
        "fuente": str(fuente),
        "registros_originales": int(len(originales)),
        "registros_limpios": int(len(limpios)),
        "tramos_entrenamiento": int(len(tramos)),
        "tamano_segmento_m": tamano_segmento_m,
        "modelo_ganador": resultados.ganador,
        "nota_metodologica": (
            "El entrenamiento usa celdas espaciales reproducibles como unidad de riesgo. "
            "La API aplica el riesgo sobre los tramos reales de OpenStreetMap al ejecutar A*. "
            "El día de la semana se conserva solo para análisis y no interviene en el score."
        ),
    }
    with (salida / "resumen_flujo.json").open("w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, ensure_ascii=False, indent=2)
    return resumen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Limpia delitos, calcula riesgo, compara RF/XGBoost y guarda artefactos."
    )
    parser.add_argument(
        "--fuente",
        type=Path,
        default=BACKEND_DIR / "data" / "AAV-DATASET.csv",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=BACKEND_DIR / "data" / "procesados",
    )
    parser.add_argument("--muestra", type=int, default=0)
    parser.add_argument("--tamano-segmento-m", type=int, default=200)
    argumentos = parser.parse_args()
    resumen = ejecutar_flujo(
        argumentos.fuente,
        argumentos.salida,
        muestra=argumentos.muestra,
        tamano_segmento_m=argumentos.tamano_segmento_m,
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
