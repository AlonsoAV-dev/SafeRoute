from __future__ import annotations

from math import cos, radians

import numpy as np
import pandas as pd

from app.services.pesos_delito import obtener_peso_delito


def agregar_riesgo_base(delitos: pd.DataFrame) -> pd.DataFrame:
    resultado = delitos.copy()
    resultado["peso_delito"] = resultado["modalidad"].map(obtener_peso_delito).astype(int)
    resultado["crime_weight"] = resultado["peso_delito"]
    resultado["riesgo_base"] = resultado["peso_delito"]
    return resultado


def crear_riesgo_por_tramo(
    delitos: pd.DataFrame,
    tamano_segmento_m: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crea unidades espaciales reproducibles para entrenar el riesgo.

    En producción, el modelo resultante se evalúa sobre los nodos y tramos OSM
    cargados por routing.py. La celda evita depender de red al reentrenar.
    """
    resultado = delitos.copy()
    latitud_referencia = float(resultado["latitud"].mean())
    metros_latitud = 111_320.0
    metros_longitud = metros_latitud * cos(radians(latitud_referencia))
    x = resultado["longitud"] * metros_longitud
    y = resultado["latitud"] * metros_latitud
    resultado["celda_x"] = np.floor(x / tamano_segmento_m).astype(int)
    resultado["celda_y"] = np.floor(y / tamano_segmento_m).astype(int)
    resultado["id_segmento"] = (
        "SEG-" + resultado["celda_x"].astype(str) + "-" + resultado["celda_y"].astype(str)
    )
    resultado["es_delito_grave"] = (resultado["peso_delito"] >= 4).astype(int)

    turno_conteos = pd.crosstab(resultado["id_segmento"], resultado["turno"]).add_prefix(
        "delitos_"
    )
    for columna in [
        "delitos_manana",
        "delitos_tarde",
        "delitos_noche",
        "delitos_madrugada",
    ]:
        if columna not in turno_conteos:
            turno_conteos[columna] = 0

    tramos = (
        resultado.groupby("id_segmento", as_index=False)
        .agg(
            latitud=("latitud", "mean"),
            longitud=("longitud", "mean"),
            distrito=("distrito", _moda),
            frecuencia_delitos=("id_hecho", "size"),
            suma_riesgo_base=("peso_delito", "sum"),
            promedio_riesgo=("peso_delito", "mean"),
            peso_delito_promedio=("peso_delito", "mean"),
            peso_delito_maximo=("peso_delito", "max"),
            delitos_graves_cercanos=("es_delito_grave", "sum"),
        )
        .merge(turno_conteos.reset_index(), on="id_segmento", how="left")
    )
    frecuencia_normalizada = _normalizar(tramos["frecuencia_delitos"])
    gravedad_normalizada = (tramos["peso_delito_promedio"] / 5).clip(0, 1)
    tramos["riesgo_frecuencia"] = frecuencia_normalizada.round(6)
    tramos["riesgo_gravedad"] = gravedad_normalizada.round(6)
    tramos["riesgo_score"] = (
        0.30 * frecuencia_normalizada + 0.70 * gravedad_normalizada
    ).clip(0, 1).round(6)
    tramos["distancia_m"] = float(tamano_segmento_m)
    tramos["nodo_origen"] = tramos["id_segmento"] + "-A"
    tramos["nodo_destino"] = tramos["id_segmento"] + "-B"
    tramos["geometria"] = tramos.apply(
        lambda fila: f"POINT ({fila['longitud']:.6f} {fila['latitud']:.6f})",
        axis=1,
    )
    return resultado.drop(columns=["celda_x", "celda_y"]), tramos


def _normalizar(serie: pd.Series) -> pd.Series:
    minimo = float(serie.min())
    maximo = float(serie.max())
    if np.isclose(minimo, maximo):
        return pd.Series(np.zeros(len(serie)), index=serie.index)
    return (serie - minimo) / (maximo - minimo)


def _moda(serie: pd.Series) -> str:
    moda = serie.mode()
    return str(moda.iloc[0]) if not moda.empty else "NO ESPECIFICADO"
