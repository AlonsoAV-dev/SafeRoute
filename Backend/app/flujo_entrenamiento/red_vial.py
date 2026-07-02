from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd

from app.services.segmentos import construir_id_segmento


def cargar_o_descargar_grafo(
    ruta_graphml: Path,
    lugar_osm: str = "Lima Metropolitana, Lima, Peru",
):
    """Carga una red OSM persistida o la descarga una sola vez."""
    if ruta_graphml.exists():
        return ox.load_graphml(ruta_graphml)
    ruta_graphml.parent.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.requests_timeout = 180
    grafo = ox.graph_from_place(lugar_osm, network_type="drive", simplify=True)
    grafo = ox.distance.add_edge_lengths(grafo)
    ox.save_graphml(grafo, ruta_graphml)
    return grafo


def extraer_tramos(grafo) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Obtiene una fila por tramo físico, compartida por ambos sentidos de circulación."""
    aristas = ox.graph_to_gdfs(
        grafo,
        nodes=False,
        edges=True,
        fill_edge_geometry=True,
    ).reset_index()
    aristas["tramo_id"] = aristas.apply(
        lambda row: construir_id_segmento(
            row["u"], row["v"], row["key"], row.to_dict()
        ),
        axis=1,
    )
    aristas["longitud_m"] = pd.to_numeric(
        aristas["length"], errors="coerce"
    ).fillna(0.0)
    tramos_geo = aristas.sort_values("longitud_m", ascending=False).drop_duplicates(
        "tramo_id", keep="first"
    )
    tramos_geo = gpd.GeoDataFrame(tramos_geo, geometry="geometry", crs=aristas.crs)
    crs_metrico = tramos_geo.estimate_utm_crs()
    tramos_metricos = tramos_geo.to_crs(crs_metrico)
    puntos_medios = gpd.GeoSeries(
        tramos_metricos.geometry.interpolate(0.5, normalized=True),
        crs=crs_metrico,
    ).to_crs(tramos_geo.crs)
    tramos_geo["latitud"] = puntos_medios.y.to_numpy()
    tramos_geo["longitud"] = puntos_medios.x.to_numpy()
    tramos = tramos_geo[
        ["tramo_id", "u", "v", "key", "longitud_m", "latitud", "longitud"]
    ].copy()
    tramos["geometria"] = tramos_geo.geometry.to_wkt()
    return tramos.reset_index(drop=True), tramos_geo[
        ["tramo_id", "geometry"]
    ].reset_index(drop=True)
