from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import cos, radians
from threading import RLock

import numpy as np
import shapely
from shapely import LineString, STRtree, box

from app.services.risk_model import RiskModel
from app.services.routing import _load_local_road_network
from app.services.segmentos import construir_id_segmento


LEVELS = ("bajo", "medio", "alto")
MIN_STREET_ZOOM = 13
ROAD_INDEX_LOCK = RLock()


@dataclass(frozen=True)
class RoadGeometryIndex:
    ids: tuple[str, ...]
    names: tuple[str, ...]
    geometries: np.ndarray
    tree: STRtree


@lru_cache(maxsize=1)
# Construye un índice de las geometrías físicas para consultar solo las calles visibles.
def _road_geometry_index() -> RoadGeometryIndex:
    """Geometría original de cada tramo físico, compartida por sus sentidos."""
    graph, _, _, _ = _load_local_road_network()
    roads = {}
    for u, v, key, data in graph.edges(keys=True, data=True):
        segment_id = construir_id_segmento(u, v, key, data)
        if segment_id in roads:
            continue
        geometry = data.get("geometry")
        if geometry is None:
            geometry = LineString([(graph.nodes[u]["x"], graph.nodes[u]["y"]),
                                   (graph.nodes[v]["x"], graph.nodes[v]["y"])])
        if geometry.is_empty:
            continue
        name = data.get("name") or "Tramo vial evaluado"
        if isinstance(name, (list, tuple)):
            name = " / ".join(map(str, name))
        roads[segment_id] = (str(name), geometry)
    geometries = np.asarray([road[1] for road in roads.values()], dtype=object)
    return RoadGeometryIndex(tuple(roads), tuple(road[0] for road in roads.values()),
                             geometries, STRtree(geometries))


def _line_parts(geometry):
    if geometry.geom_type == "LineString":
        if len(geometry.coords) > 1:
            yield geometry
    elif geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        for part in geometry.geoms:
            yield from _line_parts(part)


class RiskSegments:
    """Predicciones sobre geometrías viales: sin interpolación entre calles.

    La consulta incluye todos los tramos con predicción de la vista. Solo se
    simplifica la geometría a una tolerancia inferior a un píxel del mapa;
    los scores y las clases proceden de los mismos registros usados por A*.
    """

    def __init__(self, risk_model: RiskModel):
        self.risk_model = risk_model

    # Combina geometrías y predicciones del modelo seleccionado para representar el riesgo por calle.
    def get_segments(self, bounds: tuple[float, float, float, float], zoom: int,
                     model: str | None = None) -> dict:
        with ROAD_INDEX_LOCK:
            roads = _road_geometry_index()
        info = self.risk_model.model_info(model)
        predictions = self.risk_model.segment_predictions(info["key"])
        south, west, north, east = bounds
        visible = box(west, south, east, north)
        visible_indices = roads.tree.query(visible, predicate="intersects")
        counts = dict.fromkeys(LEVELS, 0)
        without_prediction = 0
        very_low = 0
        for index in visible_indices:
            prediction = predictions.get(roads.ids[index])
            if prediction is None or prediction.level not in counts:
                without_prediction += 1
                continue
            counts[prediction.level] += 1
            very_low += prediction.level == "bajo" and prediction.score <= 0.12

        segments = []
        if zoom >= MIN_STREET_ZOOM:
            meters_per_pixel = 156543.03392 * cos(radians((south + north) / 2)) / (2 ** zoom)
            degrees_per_pixel = meters_per_pixel / 111_320
            padding_lat = degrees_per_pixel * 20
            padding_lng = padding_lat / max(cos(radians((south + north) / 2)), 0.1)
            drawing_area = box(west - padding_lng, south - padding_lat,
                               east + padding_lng, north + padding_lat)
            indices = roads.tree.query(drawing_area, predicate="intersects")
            indices = np.asarray([index for index in indices if roads.ids[index] in predictions], dtype=int)
            clipped = shapely.intersection(roads.geometries[indices], drawing_area)
            geometries = shapely.simplify(clipped, degrees_per_pixel * 0.45, preserve_topology=True)
            for index, geometry in zip(indices, geometries):
                prediction = predictions[roads.ids[index]]
                if prediction.level not in LEVELS:
                    continue
                lines = [[[round(float(lat), 7), round(float(lng), 7)] for lng, lat, *_ in line.coords]
                         for line in _line_parts(geometry)]
                if lines:
                    segments.append({"id_segmento": roads.ids[index], "nombre": roads.names[index],
                                     "risk_score": round(prediction.score, 6), "risk_level": prediction.level,
                                     "lines": lines})
            segments.sort(key=lambda segment: (LEVELS.index(segment["risk_level"]), segment["risk_score"]))
        return {
            "segments": segments, "counts_by_level": counts, "total_segments": sum(counts.values()),
            "very_low_segments": int(very_low), "without_prediction": without_prediction,
            "model": info["name"], "model_key": info["key"],
            "prediction_period": info["prediction_period"], "risk_scope": "mensual",
            "representation": "geometria vial", "min_zoom": MIN_STREET_ZOOM,
            "zoom": zoom, "detail_available": zoom >= MIN_STREET_ZOOM,
        }
