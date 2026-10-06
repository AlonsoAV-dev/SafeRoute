from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from functools import lru_cache
from math import cos, radians, log, sqrt
from threading import RLock

import numpy as np
import shapely
from pyproj import Transformer
from shapely import LineString, Point, STRtree
from scipy.spatial import cKDTree

from app.services.risk_model import RiskModel
from app.services.segmentos import construir_id_segmento


LEVELS = ("bajo", "medio", "alto")
# Configura el muestreo vial, los focos y la escala física común del mapa y el ruteo.
ROAD_SAMPLE_SPACING_M = 20.0
HOTSPOT_CELL_SIZE_M = 100
HOTSPOT_RADIUS_M = 250
HOTSPOT_SEPARATION_M = 450
HOTSPOT_MIN_SCORE = 0.66
FIELD_VERSION = "shared-risk-field-v1"
FIELD_FLOOR = 0.6
FIELD_OPACITY_EXPONENT = 2
FIELD_MIN_INFLUENCE = 0.025
FIELD_KERNEL_EXPONENT = 4.5
RED_SCORE = 0.80
ORANGE_SCORE = 0.66
ROAD_CLEARANCE_M = 20.0
ROUTING_ROAD_CACHE_SIZE = 48_000
COLOR_STOPS = [[0, [187, 247, 208]], [0.20, [74, 222, 128]], [0.34, [163, 230, 53]],
               [0.50, [250, 204, 21]], [ORANGE_SCORE, [249, 115, 22]],
               [RED_SCORE, [239, 68, 68]], [1, [185, 28, 28]]]
TO_METERS = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)
TO_COORDINATES = Transformer.from_crs("EPSG:32718", "EPSG:4326", always_xy=True)
INDEX_LOCK = RLock()


@dataclass(frozen=True)
class RoadSamples:
    ids: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray
    lat: np.ndarray
    lng: np.ndarray
    road_index: np.ndarray
    length_m: np.ndarray


@dataclass(frozen=True)
class SpatialFieldIndex:
    hotspots: np.ndarray
    strengths: np.ndarray
    tree: cKDTree
    zone_centers: np.ndarray
    zone_radii: np.ndarray
    zone_colors: tuple[str, ...]
    zone_polygons: np.ndarray
    protected_polygons: np.ndarray
    zone_tree: STRtree


@lru_cache(maxsize=1)
# Representa cada segmento físico una sola vez mediante muestras distribuidas a lo largo de su geometría.
def _sample_road_geometry() -> RoadSamples:
    """Representa cada tramo físico una vez, con muestras de longitud uniforme."""
    from app.services.routing import _load_local_road_network
    graph, _, _, _ = _load_local_road_network()
    lines = {}
    for u, v, key, data in graph.edges(keys=True, data=True):
        segment_id = construir_id_segmento(u, v, key, data)
        if segment_id in lines:
            continue
        geometry = data.get("geometry")
        if geometry is None:
            geometry = LineString([(graph.nodes[u]["x"], graph.nodes[u]["y"]),
                                   (graph.nodes[v]["x"], graph.nodes[v]["y"])])
        if not geometry.is_empty:
            lines[segment_id] = geometry
    geometries = np.asarray(list(lines.values()), dtype=object)
    projected = shapely.transform(geometries, lambda xy: np.column_stack(TO_METERS.transform(xy[:, 0], xy[:, 1])))
    lengths = shapely.length(projected)
    counts = np.maximum(1, np.ceil(lengths / ROAD_SAMPLE_SPACING_M)).astype(np.int32)
    road_index = np.repeat(np.arange(len(lengths), dtype=np.int32), counts)
    starts = np.cumsum(counts) - counts
    fractions = (np.arange(len(road_index)) - starts[road_index] + 0.5) / counts[road_index]
    points = shapely.line_interpolate_point(projected[road_index], fractions, normalized=True)
    x, y = shapely.get_x(points), shapely.get_y(points)
    lng, lat = TO_COORDINATES.transform(x, y)
    return RoadSamples(tuple(lines), x, y, lat, lng, road_index,
                       (lengths[road_index] / counts[road_index]).astype(np.float32))


class RiskSurface:
    """Focos visuales del score predicho, sobre una cuadrícula física fija.

    Cada celda conserva el máximo score vial. El cliente mezcla los focos con
    un máximo, no con una suma: la densidad de calles no satura el mapa.
    Los scores originales del modelo se conservan. El campo derivado tiene
    una versión y ecuación compartidas por el heatmap y el costo de A*.
    """

    def __init__(self, risk_model: RiskModel):
        self.risk_model = risk_model
        self._model_arrays = {}
        self._hotspot_arrays = {}
        self._field_indexes = {}
        self._routing_road_cache = {}
        self._routing_cache_lock = RLock()

    # Integra el campo sobre cada calle y reutiliza sus medias, picos e intersecciones geométricas.
    def routing_road_info(self, roads: dict, model: str | None, spacing_m: float) -> dict:
        """Reutiliza integración y cruces por calle, nunca excepciones de extremos.

        La caché vive con la misma superficie que el heatmap y está separada
        por modelo, periodo, versión del campo y paso de integración.
        """
        metadata = self.risk_model.model_info(model)
        cache_key = (metadata["key"], metadata.get("prediction_period"), FIELD_VERSION, spacing_m)
        with self._routing_cache_lock:
            cache = self._routing_road_cache.setdefault(cache_key, OrderedDict())
            result = {}
            for segment_id in roads:
                if segment_id in cache:
                    result[segment_id] = cache[segment_id]
                    cache.move_to_end(segment_id)
            missing = [segment_id for segment_id in roads if segment_id not in result]
            if missing:
                projected = shapely.transform(np.asarray([roads[segment_id] for segment_id in missing], dtype=object),
                    lambda xy: np.column_stack(TO_METERS.transform(xy[:, 0], xy[:, 1])))
                lengths = shapely.length(projected)
                counts = np.maximum(1, np.ceil(lengths / spacing_m)).astype(int)
                owners = np.repeat(np.arange(len(missing)), counts)
                starts = np.cumsum(counts) - counts
                fractions = (np.arange(len(owners)) - starts[owners] + 0.5) / counts[owners]
                points = shapely.line_interpolate_point(projected[owners], fractions, normalized=True)
                scores = self.scores_at_xy(np.column_stack((shapely.get_x(points), shapely.get_y(points))), model)
                means = np.bincount(owners, weights=scores, minlength=len(missing)) / counts
                peaks = np.zeros(len(missing))
                np.maximum.at(peaks, owners, scores)
                zones = self.road_zones_many(projected, model)
                for index, segment_id in enumerate(missing):
                    row = {"geometry": projected[index], "projected_length": float(lengths[index]),
                           "mean": float(means[index]), "peak": float(peaks[index]),
                           "zones": zones[index]}
                    result[segment_id] = row
                    cache[segment_id] = row
                while len(cache) > ROUTING_ROAD_CACHE_SIZE:
                    cache.popitem(last=False)
            return result

    def _arrays(self, samples: RoadSamples, model: str | None):
        key = self.risk_model.model_info(model)["key"]
        with INDEX_LOCK:
            if key not in self._model_arrays:
                predictions = self.risk_model.segment_predictions(key)
                scores = np.full(len(samples.ids), np.nan, dtype=np.float32)
                levels = np.full(len(samples.ids), -1, dtype=np.int8)
                for index, segment_id in enumerate(samples.ids):
                    prediction = predictions.get(segment_id)
                    if prediction and prediction.level in LEVELS:
                        scores[index] = prediction.score
                        levels[index] = LEVELS.index(prediction.level)
                self._model_arrays[key] = (scores, levels)
            return self._model_arrays[key]

    # Agrupa máximos locales en una cuadrícula fija y evita focos demasiado próximos.
    def _hotspots(self, samples: RoadSamples, model: str | None) -> np.ndarray:
        """Agrupa focos vecinos conservando el máximo; selección global estable.

        No elimina predicciones ni delitos y no depende del viewport. Estos
        mismos focos generan el campo espacial compartido por el mapa y A*.
        """
        key = self.risk_model.model_info(model)["key"]
        with INDEX_LOCK:
            if key in self._hotspot_arrays:
                return self._hotspot_arrays[key]
            scores, _ = self._arrays(samples, key)
            sample_scores = scores[samples.road_index]
            mask = np.isfinite(sample_scores) & (sample_scores >= HOTSPOT_MIN_SCORE)
            positions = np.column_stack((np.floor(samples.x[mask] / HOTSPOT_CELL_SIZE_M),
                                         np.floor(samples.y[mask] / HOTSPOT_CELL_SIZE_M))).astype(np.int32)
            keys, inverse = np.unique(positions, axis=0, return_inverse=True)
            peaks = np.zeros(len(keys), dtype=np.float32)
            np.maximum.at(peaks, inverse, sample_scores[mask])
            order = np.lexsort((keys[:, 1], keys[:, 0], -peaks))
            buckets = {}
            selected = []
            for index in order:
                x, y = (keys[index] + 0.5) * HOTSPOT_CELL_SIZE_M
                bx, by = int(x // HOTSPOT_SEPARATION_M), int(y // HOTSPOT_SEPARATION_M)
                neighbors = [point for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                             for point in buckets.get((bx + dx, by + dy), ())]
                if any((x - px) ** 2 + (y - py) ** 2 < HOTSPOT_SEPARATION_M ** 2
                       for px, py in neighbors):
                    continue
                buckets.setdefault((bx, by), []).append((x, y))
                selected.append((float(x), float(y), float(peaks[index])))
            self._hotspot_arrays[key] = np.asarray(selected, dtype=float).reshape(-1, 3)
            return self._hotspot_arrays[key]

    def field_config(self) -> dict:
        return {"version": FIELD_VERSION, "radius_m": HOTSPOT_RADIUS_M,
                "focus_floor": FIELD_FLOOR, "opacity_exponent": FIELD_OPACITY_EXPONENT,
                "min_influence": FIELD_MIN_INFLUENCE, "kernel_exponent": FIELD_KERNEL_EXPONENT,
                "red_score": RED_SCORE, "orange_score": ORANGE_SCORE,
                "color_stops": COLOR_STOPS, "road_clearance_m": ROAD_CLEARANCE_M,
                "score_source": "score del mismo campo espacial para mapa y ruteo"}

    # Construye las zonas roja y naranja y los índices espaciales que utilizan mapa y rutas.
    def field_index(self, model: str | None = None) -> SpatialFieldIndex:
        key = self.risk_model.model_info(model)["key"]
        with INDEX_LOCK:
            if key in self._field_indexes:
                return self._field_indexes[key]
            hotspots = self._hotspots(_sample_road_geometry(), key)
            strengths = np.clip((hotspots[:, 2] - FIELD_FLOOR) / (1 - FIELD_FLOOR), 0, 1) ** FIELD_OPACITY_EXPONENT
            centers, radii, colors, polygons, protected = [], [], [], [], []
            for (x, y, score), strength in zip(hotspots, strengths):
                if strength <= FIELD_MIN_INFLUENCE:
                    continue
                visible_radius = HOTSPOT_RADIUS_M * sqrt(log(strength / FIELD_MIN_INFLUENCE) / FIELD_KERNEL_EXPONENT)
                for color, threshold in (("red", RED_SCORE), ("orange", ORANGE_SCORE)):
                    if score <= threshold:
                        continue
                    radius = min(HOTSPOT_RADIUS_M, visible_radius,
                                 HOTSPOT_RADIUS_M * sqrt(2 * log(score / threshold) / FIELD_KERNEL_EXPONENT))
                    centers.append((float(x), float(y)))
                    radii.append(radius)
                    colors.append(color)
                    polygons.append(Point(x, y).buffer(radius, quad_segs=32))
                    protected.append(Point(x, y).buffer(radius + ROAD_CLEARANCE_M, quad_segs=32))
            index = SpatialFieldIndex(hotspots, strengths, cKDTree(hotspots[:, :2]),
                                      np.asarray(centers, dtype=float).reshape(-1, 2),
                                      np.asarray(radii), tuple(colors),
                                      np.asarray(polygons, dtype=object), np.asarray(protected, dtype=object),
                                      STRtree(np.asarray(protected, dtype=object)))
            self._field_indexes[key] = index
            return index

    # Evalúa el score continuo usando el mismo kernel y la misma combinación de máximos del frontend.
    def scores_at_xy(self, coordinates: np.ndarray, model: str | None = None) -> np.ndarray:
        """RISK_SCORE(x): mismo máximo de influencia y mismo kernel que el cliente."""
        field = self.field_index(model)
        coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 2)
        result = np.zeros(len(coordinates), dtype=float)
        neighbors = field.tree.query_ball_point(coordinates, HOTSPOT_RADIUS_M)
        counts = np.fromiter((len(items) for items in neighbors), dtype=int, count=len(coordinates))
        if not counts.sum():
            return result
        owners = np.repeat(np.arange(len(coordinates)), counts)
        ids = np.concatenate([items for items in neighbors if len(items)]).astype(int)
        squared_distance = ((coordinates[owners] - field.hotspots[ids, :2]) ** 2).sum(axis=1)
        kernel = np.exp(-FIELD_KERNEL_EXPONENT * squared_distance / HOTSPOT_RADIUS_M ** 2)
        influences = field.strengths[ids] * kernel
        strongest = np.zeros(len(coordinates))
        np.maximum.at(strongest, owners, influences)
        winners = (influences >= strongest[owners]) & (influences > FIELD_MIN_INFLUENCE)
        np.maximum.at(result, owners[winners], field.hotspots[ids[winners], 2] * np.sqrt(kernel[winners]))
        return result

    # Identifica las zonas que contienen el origen o destino para aplicar las excepciones del ruteo.
    def endpoint_zones(self, coordinates, model: str | None = None) -> set[int]:
        field = self.field_index(model)
        allowed = set()
        for lat, lng in coordinates:
            xy = np.asarray(TO_METERS.transform(lng, lat))
            distances = np.linalg.norm(field.zone_centers - xy, axis=1)
            allowed.update(np.flatnonzero(distances <= field.zone_radii).tolist())
        return allowed

    def road_zones(self, geometry, model: str | None = None) -> dict:
        """Intersecciones geométricas exactas; no depende solo del punto medio."""
        field = self.field_index(model)
        indices = field.zone_tree.query(geometry, predicate="intersects")
        zones = {color: tuple(int(index) for index in indices if field.zone_colors[index] == color)
                 for color in ("red", "orange")}
        pieces = {color: shapely.intersection(geometry, shapely.union_all(field.zone_polygons[list(zones[color])]))
                  if zones[color] else LineString() for color in zones}
        return {"red_ids": zones["red"], "orange_ids": zones["orange"],
                "red_m": float(shapely.length(pieces["red"])),
                "orange_m": float(shapely.length(shapely.difference(pieces["orange"], pieces["red"]))) }

    # Mide las intersecciones de las calles con zonas de riesgo mediante operaciones geométricas.
    def road_zones_many(self, geometries, model: str | None = None) -> list[dict]:
        """Interseca en lote las calles y los focos, con los mismos límites.

        La separación actual mantiene disjuntos los núcleos de distintos
        focos. Cada núcleo rojo está contenido en su núcleo naranja; por eso
        los metros naranjas se obtienen restando el rojo del total naranja.
        Si cambia esa condición, se conserva la unión geométrica por calle.
        """
        field = self.field_index(model)
        if 2 * field.zone_radii.max(initial=0.) > HOTSPOT_SEPARATION_M:
            return [self.road_zones(geometry, model) for geometry in geometries]
        owners, zones = field.zone_tree.query(geometries, predicate="intersects")
        red_ids, orange_ids = {}, {}
        for owner, zone in zip(owners, zones):
            target = red_ids if field.zone_colors[zone] == "red" else orange_ids
            target.setdefault(int(owner), []).append(int(zone))
        lengths = shapely.length(shapely.intersection(geometries[owners], field.zone_polygons[zones]))
        is_red = np.fromiter((field.zone_colors[zone] == "red" for zone in zones), dtype=bool, count=len(zones))
        red_m = np.bincount(owners[is_red], weights=lengths[is_red], minlength=len(geometries))
        orange_m = np.bincount(owners[~is_red], weights=lengths[~is_red], minlength=len(geometries))
        return [{"red_ids": tuple(red_ids.get(index, ())), "orange_ids": tuple(orange_ids.get(index, ())),
                 "red_m": float(red_m[index]), "orange_m": float(max(0., orange_m[index] - red_m[index]))}
                for index in range(len(geometries))]

    # Entrega los focos y metadatos visibles según el área y el zoom solicitados.
    def get_surface(self, bounds: tuple[float, float, float, float], zoom: int,
                    model: str | None = None) -> dict:
        with INDEX_LOCK:
            samples = _sample_road_geometry()
        info = self.risk_model.model_info(model)
        scores, levels = self._arrays(samples, model)
        south, west, north, east = bounds
        meters_per_pixel = 156543.03392 * cos(radians((south + north) / 2)) / (2 ** zoom)
        cell_size = HOTSPOT_CELL_SIZE_M
        xs, ys = TO_METERS.transform([west, west, east, east], [south, north, south, north])
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        radius_px = HOTSPOT_RADIUS_M / meters_per_pixel
        padding = meters_per_pixel * radius_px + cell_size
        known = np.isfinite(scores[samples.road_index])
        inside = (known & (samples.lat >= south) & (samples.lat <= north)
                  & (samples.lng >= west) & (samples.lng <= east))
        visible_ids = np.unique(samples.road_index[inside])
        counts = {level: int(np.count_nonzero(levels[visible_ids] == index)) for index, level in enumerate(LEVELS)}
        hotspots = self._hotspots(samples, model)
        mask = ((hotspots[:, 0] >= min_x - padding) & (hotspots[:, 0] <= max_x + padding)
                & (hotspots[:, 1] >= min_y - padding) & (hotspots[:, 1] <= max_y + padding))
        visible_hotspots = hotspots[mask]
        cells = []
        if len(visible_hotspots):
            lngs, lats = TO_COORDINATES.transform(visible_hotspots[:, 0], visible_hotspots[:, 1])
            cells = [[round(float(lat), 7), round(float(lng), 7), round(float(score), 6)]
                     for lat, lng, score in zip(lats, lngs, visible_hotspots[:, 2])]
        return {
            "cells": cells, "cell_size_m": cell_size, "counts_by_level": counts,
            "total_segments": int(len(visible_ids)), "model": info["name"], "model_key": info["key"],
            "prediction_period": info["prediction_period"], "risk_scope": "mensual",
            "aggregation": "máximos viales de 100 m agrupados en focos con separación visual de 450 m",
            "color_source": "RISK_SCORE del campo espacial compartido", "no_data": "transparent",
            "risk_field": self.field_config(),
            "visualization": {"focus_floor": FIELD_FLOOR, "opacity_exponent": FIELD_OPACITY_EXPONENT,
                              "radius_m": HOTSPOT_RADIUS_M, "peak_separation_m": HOTSPOT_SEPARATION_M,
                              "min_peak_score": HOTSPOT_MIN_SCORE, "changes_prediction": False},
            "visible_hotspots": len(cells),
            "smoothing_radius_px": radius_px,
            "coverage": {"road_segments": len(samples.ids), "predicted_segments": int(np.count_nonzero(np.isfinite(scores))),
                         "without_prediction": int(np.count_nonzero(~np.isfinite(scores)))},
            "road_length_by_level_m": {level: round(float(samples.length_m[inside][levels[samples.road_index[inside]] == index].sum()), 2)
                                       for index, level in enumerate(LEVELS)},
        }
