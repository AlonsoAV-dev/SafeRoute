from __future__ import annotations

import logging
import socket
from itertools import islice
from math import ceil
from typing import Callable

import networkx as nx
import numpy as np
import osmnx as ox
from requests import RequestException

from app.services.risk_model import RiskModel, haversine_m, level_from_score
from app.services.segmentos import construir_id_segmento


LOGGER = logging.getLogger("saferoute.routing")
LOGGER.setLevel(logging.INFO)
BETA_VALUES = (1, 3, 5, 10, 20, 50, 100)
MAX_SAFE_DISTANCE_FACTOR = 1.50
MIN_RISK_REDUCTION_PERCENT = 0.50
DEFAULT_SPEED_KMH = 25
HISTORICAL_RISK_WEIGHT = 0.70
PREDICTED_RISK_WEIGHT = 0.30


def generate_route_comparison(
    origin: tuple[float, float],
    destination: tuple[float, float],
    risk_model: RiskModel,
    modelo_riesgo: str = "auto",
    beta: float = 10,
    buffer_m: int = 200,
    risk_mode: str = "predicted",
) -> dict:
    model_used = risk_model.resolve_model(modelo_riesgo)
    try:
        graph, start_node, end_node, node_to_latlng = _build_osm_graph_base(
            origin, destination
        )
        graph_source = "OpenStreetMap"
    except (nx.NetworkXException, RequestException, ValueError, RuntimeError, OSError):
        graph, start_node, end_node, node_to_latlng = _build_grid_graph_base(
            origin, destination
        )
        graph_source = "grilla local"

    risk_summary = _assign_segment_risks(
        graph,
        node_to_latlng,
        risk_model,
        buffer_m,
        risk_mode,
    )
    fast_path = _compute_path(
        graph, start_node, end_node, node_to_latlng, "cost_fast"
    )
    fast_route = _route_metrics(
        graph,
        fast_path,
        origin,
        destination,
        node_to_latlng,
        "cost_fast",
    )

    beta_diagnostics = []
    safe_candidates = []
    tested_betas = sorted({*BETA_VALUES, float(beta)})
    for tested_beta in tested_betas:
        _set_safe_cost(graph, tested_beta)
        path = _compute_path(
            graph, start_node, end_node, node_to_latlng, "cost_safe"
        )
        route = _route_metrics(
            graph,
            path,
            origin,
            destination,
            node_to_latlng,
            "cost_safe",
        )
        reduction = _risk_reduction(
            fast_route["risk_total"], route["risk_total"]
        )
        geometry_changed = route["node_path"] != fast_route["node_path"]
        diagnostic = {
            "beta": tested_beta,
            "distance_km": route["distance_km"],
            "risk_total": route["risk_total"],
            "risk_average": route["risk_score"],
            "risk_reduction": reduction,
            "geometry_changed": geometry_changed,
        }
        beta_diagnostics.append(diagnostic)
        safe_candidates.append((tested_beta, route, reduction))

    alternatives = [
        candidate
        for candidate in safe_candidates
        if candidate[2] >= MIN_RISK_REDUCTION_PERCENT
        and candidate[1]["node_path"] != fast_route["node_path"]
        and candidate[1]["distance_km"]
        <= fast_route["distance_km"] * MAX_SAFE_DISTANCE_FACTOR
    ]
    if alternatives:
        safe_beta, safe_route, reduction = max(
            alternatives,
            key=lambda candidate: (
                candidate[2],
                -candidate[1]["distance_km"],
            ),
        )
    else:
        safe_beta = 0
        safe_route = fast_route
        reduction = 0.0

    same_route = safe_route["node_path"] == fast_route["node_path"]
    if same_route:
        message = (
            "La ruta segura coincide con la ruta más rápida porque no se encontró "
            "una alternativa con menor riesgo significativo."
        )
        reduction = 0.0 if same_route else reduction
    else:
        distance_delta = safe_route["distance_km"] - fast_route["distance_km"]
        message = (
            f"La ruta segura reduce la exposición estimada en {reduction:.2f}% "
            f"con una variación de distancia de {distance_delta:+.2f} km."
        )

    alternative_count = _count_alternatives(graph, start_node, end_node)
    LOGGER.info(
        "route_comparison source=%s alternatives=%s model=%s buffer=%sm "
        "beta=%s fast_risk=%.4f safe_risk=%.4f fast_high=%s safe_high=%s "
        "formula=distance_m*(1+beta*risk_norm) risk_min=%.4f risk_p95=%.4f "
        "risk_max=%.4f",
        graph_source,
        alternative_count,
        model_used,
        buffer_m,
        safe_beta,
        fast_route["risk_total"],
        safe_route["risk_total"],
        fast_route["high_risk_segments"],
        safe_route["high_risk_segments"],
        risk_summary["min"],
        risk_summary["p95"],
        risk_summary["max"],
    )

    for route in (safe_route, fast_route):
        route.pop("node_path", None)

    return {
        "safe_route": safe_route,
        "traditional_route": fast_route,
        "ruta_segura": _spanish_route_alias(safe_route),
        "ruta_rapida": _spanish_route_alias(fast_route),
        "risk_reduction": round(reduction, 2),
        "reduccion_riesgo": round(reduction, 2),
        "misma_ruta": same_route,
        "modelo_riesgo_solicitado": modelo_riesgo,
        "modelo_usado": model_used,
        "modo_riesgo": risk_mode,
        "periodo_prediccion": risk_model.prediction_period,
        "metricas_modelo": risk_model.metrics_for_model(modelo_riesgo),
        "parametros_a_star": {
            "alpha": 1,
            "beta_ruta_rapida": 0,
            "beta_ruta_segura": safe_beta,
            "buffer_m": buffer_m,
            "peso_riesgo_historico": (
                1.0 if risk_mode == "historical" else HISTORICAL_RISK_WEIGHT
                if risk_mode == "hybrid"
                else 0.0
            ),
            "peso_riesgo_predicho": (
                1.0 if risk_mode == "predicted" else PREDICTED_RISK_WEIGHT
                if risk_mode == "hybrid"
                else 0.0
            ),
            "desvio_maximo_porcentaje": round(
                (MAX_SAFE_DISTANCE_FACTOR - 1) * 100, 1
            ),
            "formula": "distancia_m * (1 + beta * riesgo_segmento_normalizado)",
        },
        "diagnostico_beta": beta_diagnostics,
        "diagnostico_grafo": {
            "fuente": graph_source,
            "alternativas_encontradas": alternative_count,
            "segmentos": graph.number_of_edges(),
            "riesgo_segmento_min": round(risk_summary["min"], 6),
            "riesgo_segmento_p95": round(risk_summary["p95"], 6),
            "riesgo_segmento_max": round(risk_summary["max"], 6),
            "p95_riesgo_historico_bruto": round(
                risk_summary["historical_p95_raw"], 6
            ),
        },
        "mensaje": message,
    }


def generate_safe_route(
    origin: tuple[float, float],
    destination: tuple[float, float],
    turno: str,
    risk_model: RiskModel,
    safety_weight: float,
    alpha: float | None = None,
) -> dict:
    del turno, safety_weight
    comparison = generate_route_comparison(
        origin=origin,
        destination=destination,
        risk_model=risk_model,
        beta=10 if alpha is None else max(0, min(20, alpha * 20)),
    )
    return comparison["safe_route"]


def _build_osm_graph_base(
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> tuple[nx.MultiDiGraph, int, int, Callable[[int], tuple[float, float]]]:
    if not _can_reach_overpass():
        raise ValueError("OpenStreetMap no está disponible.")
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.requests_timeout = 12
    mid_lat = (origin[0] + destination[0]) / 2
    mid_lng = (origin[1] + destination[1]) / 2
    straight_distance = haversine_m(
        origin[0], origin[1], destination[0], destination[1]
    )
    radius = min(7000, max(2200, straight_distance / 2 + 1400))
    graph = ox.graph_from_point(
        center_point=(mid_lat, mid_lng),
        dist=radius,
        network_type="drive",
        simplify=True,
    )
    graph = ox.distance.add_edge_lengths(graph)
    start_node = ox.distance.nearest_nodes(graph, origin[1], origin[0])
    end_node = ox.distance.nearest_nodes(graph, destination[1], destination[0])

    def node_to_latlng(node: int) -> tuple[float, float]:
        data = graph.nodes[node]
        return float(data["y"]), float(data["x"])

    return graph, start_node, end_node, node_to_latlng


def _can_reach_overpass() -> bool:
    try:
        with socket.create_connection(("overpass-api.de", 443), timeout=1):
            return True
    except OSError:
        return False


def _build_grid_graph_base(
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> tuple[
    nx.Graph,
    tuple[float, float],
    tuple[float, float],
    Callable[[tuple[float, float]], tuple[float, float]],
]:
    min_lat, max_lat = sorted([origin[0], destination[0]])
    min_lng, max_lng = sorted([origin[1], destination[1]])
    padding = 0.018
    min_lat -= padding
    max_lat += padding
    min_lng -= padding
    max_lng += padding
    steps = max(
        12,
        min(28, ceil(max(max_lat - min_lat, max_lng - min_lng) / 0.0025)),
    )
    lat_step = (max_lat - min_lat) / steps
    lng_step = (max_lng - min_lng) / steps
    graph = nx.Graph()
    node_grid = [
        [
            (
                round(min_lat + row * lat_step, 6),
                round(min_lng + col * lng_step, 6),
            )
            for col in range(steps + 1)
        ]
        for row in range(steps + 1)
    ]
    graph.add_nodes_from(node for row in node_grid for node in row)
    for row in range(steps + 1):
        for col in range(steps + 1):
            node = node_grid[row][col]
            for row_delta, col_delta in ((1, 0), (0, 1), (1, 1), (1, -1)):
                next_row = row + row_delta
                next_col = col + col_delta
                if 0 <= next_row <= steps and 0 <= next_col <= steps:
                    neighbor = node_grid[next_row][next_col]
                    graph.add_edge(
                        node,
                        neighbor,
                        length=haversine_m(*node, *neighbor),
                    )
    start_node = min(
        graph.nodes,
        key=lambda node: haversine_m(origin[0], origin[1], node[0], node[1]),
    )
    end_node = min(
        graph.nodes,
        key=lambda node: haversine_m(
            destination[0], destination[1], node[0], node[1]
        ),
    )
    return graph, start_node, end_node, lambda node: node


def _assign_segment_risks(
    graph,
    node_to_latlng,
    risk_model: RiskModel,
    buffer_m: int,
    risk_mode: str,
) -> dict:
    if risk_mode not in {"predicted", "historical", "hybrid"}:
        raise ValueError(f"Modo de riesgo no soportado: {risk_mode}")
    edge_rows = []
    for u, v, key, data in _iter_edges(graph):
        start = node_to_latlng(u)
        end = node_to_latlng(v)
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        length = float(data.get("length") or haversine_m(*start, *end))
        crime_stats = risk_model.nearby_crime_stats(
            _edge_sample_points(data, start, end),
            buffer_m,
        )
        raw_risk = crime_stats["weight_sum"] / max(length / 100, 1)
        tramo_id = construir_id_segmento(u, v, key, data)
        prediction = risk_model.predict_segment(tramo_id, midpoint)
        edge_rows.append(
            {
                "u": u,
                "v": v,
                "key": key,
                "data": data,
                "start": start,
                "end": end,
                "midpoint": midpoint,
                "length": length,
                "crime_stats": crime_stats,
                "raw_risk": raw_risk,
                "tramo_id": tramo_id,
                "prediction": prediction,
            }
        )

    historical_values = np.asarray([row["raw_risk"] for row in edge_rows], dtype=float)
    historical_positive = historical_values[historical_values > 0]
    historical_p95 = (
        float(np.percentile(historical_positive, 95))
        if len(historical_positive)
        else 1.0
    )
    combined_values = []
    for row in edge_rows:
        historical = min(1.0, max(0.0, row["raw_risk"] / historical_p95))
        predicted = min(1.0, max(0.0, float(row["prediction"].score)))
        if risk_mode == "historical":
            normalized = historical
        elif risk_mode == "hybrid":
            normalized = (
                HISTORICAL_RISK_WEIGHT * historical
                + PREDICTED_RISK_WEIGHT * predicted
            )
        else:
            normalized = predicted
        combined_values.append(normalized)
        data = row["data"]
        data.update(
            {
                "id_segmento": row["tramo_id"],
                "distance_m": row["length"],
                "time_min": row["length"] / (DEFAULT_SPEED_KMH * 1000 / 60),
                "nearby_crime_count": row["crime_stats"]["count"],
                "crime_weight_sum": row["crime_stats"]["weight_sum"],
                "crime_weight_avg": row["crime_stats"]["weight_avg"],
                "risk_segment_raw": row["raw_risk"],
                "risk_historical_normalized": historical,
                "risk_predicted": predicted,
                "risk_segment_normalized": normalized,
                "risk_level": level_from_score(normalized),
                "cost_fast": row["length"],
            }
        )
    _set_safe_cost(graph, 10)
    p95 = float(np.percentile(combined_values, 95)) if combined_values else 0.0
    return {
        "min": float(min((data["risk_segment_normalized"] for _, _, _, data in _iter_edges(graph)), default=0)),
        "p95": p95,
        "historical_p95_raw": historical_p95,
        "max": float(max((data["risk_segment_normalized"] for _, _, _, data in _iter_edges(graph)), default=0)),
    }


def _edge_sample_points(data, start, end) -> list[tuple[float, float]]:
    geometry = data.get("geometry")
    if geometry is not None and hasattr(geometry, "interpolate"):
        points = [start]
        for fraction in (0.25, 0.5, 0.75):
            point = geometry.interpolate(fraction, normalized=True)
            points.append((float(point.y), float(point.x)))
        points.append(end)
        return points
    return [
        start,
        ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2),
        end,
    ]


def _set_safe_cost(graph, beta: float) -> None:
    for _, _, _, data in _iter_edges(graph):
        data["cost_safe"] = _edge_cost(
            float(data["distance_m"]),
            float(data["risk_segment_normalized"]),
            beta,
        )


def _edge_cost(distance_m: float, risk_score: float, beta: float) -> float:
    return distance_m * (1 + beta * risk_score)


def _compute_path(graph, start_node, end_node, node_to_latlng, weight):
    return nx.astar_path(
        graph,
        start_node,
        end_node,
        heuristic=lambda a, b: haversine_m(
            node_to_latlng(a)[0],
            node_to_latlng(a)[1],
            node_to_latlng(b)[0],
            node_to_latlng(b)[1],
        ),
        weight=weight,
    )


def _route_metrics(
    graph,
    path,
    origin,
    destination,
    node_to_latlng,
    weight,
) -> dict:
    segments = []
    for index, (u, v) in enumerate(zip(path, path[1:]), start=1):
        data = _best_edge_data(graph, u, v, weight)
        segments.append(
            {
                "id_segmento": str(data["id_segmento"]),
                "nodo_origen": str(u),
                "nodo_destino": str(v),
                "distancia_metros": round(float(data["distance_m"]), 2),
                "tiempo_min": round(float(data["time_min"]), 3),
                "cantidad_delitos_cercanos": int(data["nearby_crime_count"]),
                "peso_delito_acumulado": round(float(data["crime_weight_sum"]), 3),
                "riesgo_segmento": round(float(data["risk_segment_raw"]), 6),
                "riesgo_historico": round(
                    float(data["risk_historical_normalized"]), 6
                ),
                "riesgo_predicho": round(float(data["risk_predicted"]), 6),
                "riesgo_segmento_normalizado": round(
                    float(data["risk_segment_normalized"]), 6
                ),
                "nivel_riesgo": str(data["risk_level"]),
                "orden": index,
            }
        )
    path_coords = [node_to_latlng(node) for node in path]
    route_coords = [origin, *path_coords, destination]
    connector_distance = haversine_m(*origin, *path_coords[0]) + haversine_m(
        *path_coords[-1], *destination
    )
    segment_distance = sum(segment["distancia_metros"] for segment in segments)
    distance = connector_distance + segment_distance
    risk_values = [segment["riesgo_segmento_normalizado"] for segment in segments]
    weighted_risk_m = sum(
        segment["riesgo_segmento_normalizado"] * segment["distancia_metros"]
        for segment in segments
    )
    risk_total = weighted_risk_m / 1000.0
    risk_average = weighted_risk_m / segment_distance if segment_distance else 0.0
    return {
        "route": [
            {"lat": round(lat, 6), "lng": round(lng, 6)}
            for lat, lng in route_coords
        ],
        "distance_km": round(distance / 1000, 3),
        "time_min": round(distance / (DEFAULT_SPEED_KMH * 1000 / 60), 1),
        "risk_total": round(risk_total, 6),
        "risk_score": round(risk_average, 6),
        "risk_average": round(risk_average, 6),
        "risk_level": level_from_score(risk_average),
        "high_risk_segments": sum(value >= 0.66 for value in risk_values),
        "segments": segments,
        "node_path": [str(node) for node in path],
    }


def _best_edge_data(graph, u, v, weight):
    data = graph.get_edge_data(u, v)
    if graph.is_multigraph():
        return min(data.values(), key=lambda edge: float(edge.get(weight, 0)))
    return data


def _iter_edges(graph):
    if graph.is_multigraph():
        yield from graph.edges(keys=True, data=True)
    else:
        for u, v, data in graph.edges(data=True):
            yield u, v, 0, data


def _risk_reduction(fast_risk: float, safe_risk: float) -> float:
    if fast_risk <= 1e-12:
        return 0.0
    return round(max(0.0, (fast_risk - safe_risk) / fast_risk * 100), 2)


def _count_alternatives(graph, start_node, end_node) -> int:
    try:
        simple_graph = nx.DiGraph(graph) if graph.is_directed() else nx.Graph(graph)
        return len(
            list(
                islice(
                    nx.shortest_simple_paths(
                        simple_graph,
                        start_node,
                        end_node,
                        weight="cost_fast",
                    ),
                    3,
                )
            )
        )
    except (nx.NetworkXException, nx.NetworkXNoPath):
        return 1


def _spanish_route_alias(route: dict) -> dict:
    return {
        "distancia_km": route["distance_km"],
        "tiempo_min": route["time_min"],
        "riesgo_total": route["risk_total"],
        "riesgo_promedio": route["risk_average"],
        "nivel_riesgo": route["risk_level"],
        "coordenadas": route["route"],
        "segmentos": route["segments"],
    }
