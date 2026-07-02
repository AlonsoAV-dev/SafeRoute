from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from app.flujo_entrenamiento.graficos import generar_graficos
from app.flujo_entrenamiento.limpieza import cargar_fuente, limpiar_delitos
from app.flujo_entrenamiento.modelos import NIVELES_RIESGO, entrenar_random_forest
from app.flujo_entrenamiento.red_vial import cargar_o_descargar_grafo, extraer_tramos
from app.flujo_entrenamiento.riesgo import (
    VARIABLES_MODELO,
    agregar_riesgo_base,
    crear_panel_temporal,
)


BACKEND_DIR = Path(__file__).resolve().parents[2]


def ejecutar_flujo(
    fuente: Path,
    salida: Path,
    ruta_graphml: Path,
    ventana_meses: int = 3,
    radios_m: tuple[float, ...] = (100.0, 150.0, 200.0),
    lugar_osm: str = "Lima Metropolitana, Lima, Peru",
    grafo=None,
) -> dict:
    """Compara buffers espaciales y conserva el Random Forest con mejor F1 macro."""
    salida.mkdir(parents=True, exist_ok=True)
    originales = cargar_fuente(fuente)
    limpios = agregar_riesgo_base(limpiar_delitos(originales))
    if len(limpios) < 100:
        raise ValueError("Se necesitan al menos 100 delitos válidos para entrenar.")
    limpios.to_csv(salida / "delitos_limpios.csv", index=False)

    if grafo is None:
        grafo = cargar_o_descargar_grafo(ruta_graphml, lugar_osm=lugar_osm)
    tramos, tramos_geo = extraer_tramos(grafo)
    tramos.to_csv(salida / "tramos_osm.csv", index=False)

    comparaciones = []
    mejor = None
    mejor_metadata = None
    mejor_clave = None
    filas_panel = 0
    for radio_m in sorted(set(float(radio) for radio in radios_m)):
        panel, futuro, metadata = crear_panel_temporal(
            limpios,
            tramos,
            tramos_geo,
            ventana_meses=ventana_meses,
            radio_m=radio_m,
        )
        resultados = entrenar_random_forest(panel, futuro)
        fila_comparacion = {"radio_buffer_m": radio_m, **resultados.metricas}
        comparaciones.append(fila_comparacion)
        clave = (
            resultados.metricas["f1_macro"],
            resultados.metricas["recall_riesgo_alto"],
            resultados.metricas["pr_auc_riesgo_alto"],
            resultados.metricas["balanced_accuracy"],
        )
        if mejor_clave is None or clave > mejor_clave:
            mejor = resultados
            mejor_metadata = metadata
            mejor_clave = clave
            filas_panel = len(panel)
            panel.to_csv(salida / "datos_random_forest.csv", index=False)
            futuro.to_csv(salida / "datos_prediccion_futura.csv", index=False)
        del panel, futuro
        gc.collect()

    if mejor is None or mejor_metadata is None:
        raise RuntimeError("No se pudo entrenar ningún radio espacial.")
    comparacion_radios = pd.DataFrame(comparaciones).sort_values(
        ["f1_macro", "recall_riesgo_alto", "pr_auc_riesgo_alto"],
        ascending=False,
    )
    comparacion_radios["seleccionado"] = comparacion_radios["radio_buffer_m"].eq(
        mejor_metadata["radio_buffer_m"]
    )
    comparacion_radios.to_csv(salida / "comparacion_radios.csv", index=False)

    predicciones = mejor.predicciones_futuras.merge(
        tramos[["tramo_id", "u", "v", "key", "longitud_m", "geometria"]],
        on="tramo_id",
        how="left",
    )
    predicciones.to_csv(salida / "predicciones_tramos.csv", index=False)
    metricas_seleccionadas = {
        "radio_buffer_m": mejor_metadata["radio_buffer_m"],
        **mejor.metricas,
    }
    pd.DataFrame([metricas_seleccionadas]).to_csv(
        salida / "metricas_random_forest.csv", index=False
    )
    mejor.reporte.to_csv(salida / "classification_report_random_forest.csv")
    pd.DataFrame(
        mejor.matriz_confusion,
        index=NIVELES_RIESGO,
        columns=NIVELES_RIESGO,
    ).to_csv(salida / "matriz_confusion_random_forest.csv")

    metadata_modelo = {
        **mejor_metadata,
        "modelo": "Random Forest",
        "version_variables": "reducido_16",
        "variables": VARIABLES_MODELO,
        "umbrales": mejor.umbrales,
        "periodo_prueba": mejor.periodo_prueba,
        "radios_evaluados_m": [float(radio) for radio in radios_m],
        "criterio_seleccion_radio": (
            "Mayor F1 macro; desempate por recall alto, PR-AUC alto y accuracy balanceada."
        ),
        "entrenado_en": datetime.now().isoformat(),
    }
    joblib.dump(
        {"pipeline": mejor.modelo, "metadata": metadata_modelo},
        salida / "modelo_random_forest.joblib",
    )
    archivos_graficos = generar_graficos(
        metricas_seleccionadas,
        mejor.reporte,
        mejor.matriz_confusion,
        predicciones,
        mejor.modelo,
        salida / "graficos",
        comparacion_radios,
    )
    with (salida / "metadata_modelo.json").open("w", encoding="utf-8") as archivo:
        json.dump(metadata_modelo, archivo, ensure_ascii=False, indent=2)

    resumen = {
        "fuente": str(fuente),
        "registros_originales": int(len(originales)),
        "registros_limpios": int(len(limpios)),
        "tramos_osm": int(len(tramos)),
        "filas_panel": int(filas_panel),
        **mejor_metadata,
        "comparacion_radios": comparacion_radios.to_dict(orient="records"),
        "metricas": metricas_seleccionadas,
        "graficos": archivos_graficos,
        "advertencia": (
            "El radio y las métricas deben volver a evaluarse al incorporar nuevos periodos."
        ),
    }
    with (salida / "resumen_flujo.json").open("w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, ensure_ascii=False, indent=2)
    return resumen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara buffers, crea el panel temporal y entrena Random Forest."
    )
    parser.add_argument(
        "--fuente", type=Path, default=BACKEND_DIR / "data" / "DELITOS TOTAL.csv"
    )
    parser.add_argument(
        "--salida", type=Path, default=BACKEND_DIR / "data" / "procesados"
    )
    parser.add_argument(
        "--grafo", type=Path, default=BACKEND_DIR / "data" / "red_vial_lima.graphml"
    )
    parser.add_argument("--ventana-meses", type=int, default=3)
    parser.add_argument("--radios", type=float, nargs="+", default=[100, 150, 200])
    parser.add_argument("--lugar-osm", default="Lima Metropolitana, Lima, Peru")
    argumentos = parser.parse_args()
    resumen = ejecutar_flujo(
        fuente=argumentos.fuente,
        salida=argumentos.salida,
        ruta_graphml=argumentos.grafo,
        ventana_meses=argumentos.ventana_meses,
        radios_m=tuple(argumentos.radios),
        lugar_osm=argumentos.lugar_osm,
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
