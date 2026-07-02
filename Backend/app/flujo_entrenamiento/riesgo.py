from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely import STRtree

from app.services.pesos_delito import obtener_peso_desde_campos


CONTEOS = [
    "frecuencia_delitos",
    "suma_pesos",
    "delitos_graves",
    "hurtos",
    "robos",
    "extorsiones",
    "homicidios",
    "delitos_manana",
    "delitos_tarde",
    "delitos_noche",
    "delitos_madrugada",
]

VARIABLES_MODELO = [
    "latitud",
    "longitud",
    "longitud_m",
    "frecuencia_delitos_hist",
    "suma_pesos_hist",
    "delitos_graves_hist",
    "hurtos_hist",
    "robos_hist",
    "extorsiones_hist",
    "homicidios_hist",
    "delitos_manana_hist",
    "delitos_tarde_hist",
    "delitos_noche_hist",
    "delitos_madrugada_hist",
    "densidad_delictiva_100m",
    "densidad_gravedad_100m",
]


def agregar_riesgo_base(delitos: pd.DataFrame) -> pd.DataFrame:
    """Asigna los pesos sustentados y categorías requeridas para la agregación."""
    resultado = delitos.copy()
    resultado["peso_delito"] = resultado.apply(
        lambda row: obtener_peso_desde_campos(
            row.get("modalidad"), row.get("subtipo_delito")
        ),
        axis=1,
    ).astype(int)
    texto = (resultado["modalidad"] + " " + resultado["subtipo_delito"]).str.upper()
    resultado["categoria_delito"] = np.select(
        [
            texto.str.contains("HOMICIDIO|ASESINATO", regex=True),
            texto.str.contains("EXTORSION", regex=False),
            texto.str.contains("ROBO", regex=False),
            texto.str.contains("HURTO", regex=False),
        ],
        ["homicidios", "extorsiones", "robos", "hurtos"],
        default="otros",
    )
    resultado["es_delito_grave"] = (resultado["peso_delito"] >= 4).astype(int)
    return resultado


def detectar_periodos_completos(delitos: pd.DataFrame) -> tuple[list[pd.Period], list[str]]:
    """Considera completo un mes que contiene registros hasta su último día calendario."""
    fechas = pd.to_datetime(delitos["fecha"], errors="coerce")
    resumen = pd.DataFrame({"fecha": fechas}).dropna()
    resumen["periodo"] = resumen["fecha"].dt.to_period("M")
    maximos = resumen.groupby("periodo")["fecha"].max()
    completos = [
        periodo
        for periodo, fecha_maxima in maximos.items()
        if int(fecha_maxima.day) == int(periodo.days_in_month)
    ]
    excluidos = [str(periodo) for periodo in maximos.index if periodo not in completos]
    completos = sorted(completos)
    if completos:
        esperado = list(pd.period_range(completos[0], completos[-1], freq="M"))
        if completos != esperado:
            raise ValueError("Los meses completos no forman una serie temporal continua.")
    return completos, excluidos


def crear_panel_temporal(
    delitos: pd.DataFrame,
    tramos: pd.DataFrame,
    tramos_geo: gpd.GeoDataFrame,
    ventana_meses: int = 3,
    radio_m: float = 150.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Construye el panel usando delitos dentro del buffer de cada tramo."""
    if ventana_meses < 1:
        raise ValueError("La ventana temporal debe contener al menos un mes.")
    if radio_m <= 0:
        raise ValueError("El radio espacial debe ser positivo.")
    periodos, excluidos = detectar_periodos_completos(delitos)
    if len(periodos) <= ventana_meses:
        raise ValueError(
            f"Se requieren más de {ventana_meses} meses completos para entrenar."
        )
    mensual, cobertura = _agregar_mensualmente_buffer(
        delitos,
        tramos,
        tramos_geo,
        periodos,
        radio_m,
    )

    periodo_futuro = periodos[-1] + 1
    periodos_modelo = [*periodos, periodo_futuro]
    indice = pd.MultiIndex.from_product(
        [tramos["tramo_id"].astype(str), periodos_modelo],
        names=["tramo_id", "periodo_objetivo"],
    )
    mensual = mensual.set_index(["tramo_id", "periodo_objetivo"]).reindex(
        indice, fill_value=0
    ).reset_index()
    mensual = mensual.merge(
        tramos[["tramo_id", "latitud", "longitud", "longitud_m"]],
        on="tramo_id",
        how="left",
    ).sort_values(["tramo_id", "periodo_objetivo"]).reset_index(drop=True)

    cantidad_periodos = len(periodos_modelo)
    cantidad_tramos = len(tramos)
    if len(mensual) != cantidad_tramos * cantidad_periodos:
        raise ValueError("No se pudo construir la matriz completa tramo-periodo.")
    for columna in CONTEOS:
        mensual[f"{columna}_hist"] = _ventana_anterior(
            mensual[columna],
            cantidad_tramos,
            cantidad_periodos,
            ventana_meses,
            operacion="sum",
        )
    factor_longitud = (mensual["longitud_m"] / 100.0).clip(lower=1.0)
    mensual["densidad_delictiva_100m"] = (
        mensual["frecuencia_delitos_hist"] / factor_longitud
    )
    mensual["densidad_gravedad_100m"] = mensual["suma_pesos_hist"] / factor_longitud
    mensual["riesgo_bruto_futuro"] = mensual["suma_pesos"] / factor_longitud

    columnas_base = [
        "tramo_id",
        "periodo_objetivo",
        *VARIABLES_MODELO,
        "riesgo_bruto_futuro",
    ]
    disponibles = mensual.dropna(subset=["frecuencia_delitos_hist"])[columnas_base].copy()
    panel = disponibles.loc[disponibles["periodo_objetivo"].isin(periodos)].copy()
    futuro = disponibles.loc[
        disponibles["periodo_objetivo"].eq(periodo_futuro)
    ].drop(columns=["riesgo_bruto_futuro"])
    panel["periodo_objetivo"] = panel["periodo_objetivo"].astype(str)
    futuro["periodo_objetivo"] = futuro["periodo_objetivo"].astype(str)
    metadata = {
        "ventana_meses": ventana_meses,
        "periodos_completos": [str(periodo) for periodo in periodos],
        "periodos_excluidos": excluidos,
        "periodo_prediccion": str(periodo_futuro),
        "radio_buffer_m": radio_m,
        "metodo_espacial": "buffer de la geometria vial con decaimiento gaussiano",
        "cobertura_por_periodo": cobertura,
    }
    return panel.reset_index(drop=True), futuro.reset_index(drop=True), metadata


def _agregar_mensualmente_buffer(
    delitos: pd.DataFrame,
    tramos: pd.DataFrame,
    tramos_geo: gpd.GeoDataFrame,
    periodos: list[pd.Period],
    radio_m: float,
) -> tuple[pd.DataFrame, dict]:
    """Agrega cada delito a todos los tramos cuya geometría está dentro del radio."""
    if len(tramos) != len(tramos_geo):
        raise ValueError("Los catálogos tabular y geográfico de tramos no coinciden.")
    delitos = delitos.copy()
    delitos["periodo"] = pd.to_datetime(delitos["fecha"]).dt.to_period("M")
    delitos = delitos.loc[delitos["periodo"].isin(periodos)].reset_index(drop=True)
    puntos = gpd.GeoDataFrame(
        delitos,
        geometry=gpd.points_from_xy(delitos["longitud"], delitos["latitud"]),
        crs="EPSG:4326",
    )
    crs_metrico = tramos_geo.estimate_utm_crs()
    lineas = tramos_geo.to_crs(crs_metrico).geometry.array
    puntos = puntos.to_crs(crs_metrico)
    cantidad_tramos = len(tramos)
    ids_tramos = tramos["tramo_id"].astype(str).to_numpy()
    filas = []
    cobertura = {}

    for periodo in periodos:
        mascara_periodo = puntos["periodo"].eq(periodo).to_numpy()
        delitos_mes = puntos.loc[mascara_periodo].reset_index(drop=True)
        geometrias_delitos = delitos_mes.geometry.array
        pares = STRtree(geometrias_delitos).query(
            lineas,
            predicate="dwithin",
            distance=radio_m,
        )
        indices_tramo = pares[0]
        indices_delito = pares[1]
        distancias = shapely.distance(
            lineas.take(indices_tramo),
            geometrias_delitos.take(indices_delito),
        )
        ancho_banda = max(radio_m / 2.0, 1.0)
        decaimiento = np.exp(-0.5 * (distancias / ancho_banda) ** 2)
        pesos = delitos_mes["peso_delito"].to_numpy(dtype=float)[indices_delito]

        datos_mes: dict[str, np.ndarray] = {
            "frecuencia_delitos": np.bincount(
                indices_tramo, minlength=cantidad_tramos
            ).astype(float),
            "suma_pesos": np.bincount(
                indices_tramo,
                weights=pesos * decaimiento,
                minlength=cantidad_tramos,
            ),
            "delitos_graves": _conteo_condicional(
                indices_tramo,
                delitos_mes["es_delito_grave"].to_numpy()[indices_delito].astype(bool),
                cantidad_tramos,
            ),
        }
        categorias = delitos_mes["categoria_delito"].to_numpy()[indices_delito]
        for categoria in ("hurtos", "robos", "extorsiones", "homicidios"):
            datos_mes[categoria] = _conteo_condicional(
                indices_tramo, categorias == categoria, cantidad_tramos
            )
        turnos = delitos_mes["turno"].to_numpy()[indices_delito]
        for turno in ("manana", "tarde", "noche", "madrugada"):
            datos_mes[f"delitos_{turno}"] = _conteo_condicional(
                indices_tramo, turnos == turno, cantidad_tramos
            )

        frame = pd.DataFrame(
            {
                "tramo_id": ids_tramos,
                "periodo_objetivo": periodo,
                **datos_mes,
            }
        )
        filas.append(frame)
        tramos_positivos = int((datos_mes["frecuencia_delitos"] > 0).sum())
        cobertura[str(periodo)] = {
            "delitos": int(len(delitos_mes)),
            "tramos_con_senal": tramos_positivos,
            "porcentaje_tramos_con_senal": round(
                tramos_positivos / cantidad_tramos * 100, 4
            ),
        }
    return pd.concat(filas, ignore_index=True), cobertura


def _conteo_condicional(
    indices_tramo: np.ndarray,
    condicion: np.ndarray,
    cantidad_tramos: int,
) -> np.ndarray:
    return np.bincount(
        indices_tramo[condicion], minlength=cantidad_tramos
    ).astype(float)


def _ventana_anterior(
    serie: pd.Series,
    cantidad_tramos: int,
    cantidad_periodos: int,
    ventana: int,
    operacion: str,
) -> np.ndarray:
    matriz = serie.to_numpy(dtype=float).reshape(cantidad_tramos, cantidad_periodos)
    resultado = np.full_like(matriz, np.nan, dtype=float)
    for indice in range(ventana, cantidad_periodos):
        valores = matriz[:, indice - ventana : indice]
        resultado[:, indice] = (
            valores.sum(axis=1) if operacion == "sum" else valores.max(axis=1)
        )
    return resultado.ravel()
