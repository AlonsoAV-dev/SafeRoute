from __future__ import annotations

import logging
from functools import lru_cache
from math import ceil, cos, radians
from pathlib import Path
from time import perf_counter
from typing import Callable
from weakref import WeakKeyDictionary

import networkx as nx
import numpy as np
import osmnx as ox
from sklearn.neighbors import BallTree

from app.services.risk_model import (
    RiskModel,
    RiskPrediction,
    haversine_m,
    level_from_score,
)
from app.services.segmentos import construir_id_segmento
from app.services.alternative_routes import route_signature, search_alternatives
from app.services.hotspot_routing import (
    assign_heatmap_scores, choose_avoidance_graph, crossing_reasons, field_edge_cost,
    RED_PENALTY, ORANGE_PENALTY,
)


LOGGER = logging.getLogger("uvicorn.error")
LOGGER.setLevel(logging.INFO)
# Establece las penalizaciones exploradas, el límite de desvío y la velocidad estimada.
BETA_VALUES = (1, 3, 5, 10, 20, 50, 100)
MAX_SAFE_DISTANCE_FACTOR = 1.50
MIN_RISK_REDUCTION_PERCENT = 0.50
DEFAULT_SPEED_KMH = 25
HOTSPOT_PENALTY_WEIGHT = 2.0
HISTORICAL_RISK_WEIGHT = 0.70
PREDICTED_RISK_WEIGHT = 0.30
EARTH_RADIUS_M = 6_371_000
ROAD_GRAPH_PATH = Path(__file__).resolve().parents[2] / "data" / "red_vial_lima.graphml"


# Compara la ruta de menor distancia con alternativas evaluadas sobre la misma red vial.
def generate_route_comparison(
    origin: tuple[float, float],
    destination: tuple[float, float],
    risk_model: RiskModel,
    modelo_riesgo: str = "auto",
    beta: float = 10,
    buffer_m: int = 200,
    risk_mode: str = "predicted",
    turno: str | None = None,
) -> dict:
    started_at = perf_counter()
    model_used = risk_model.resolve_model(modelo_riesgo)
    info_provider = getattr(risk_model, "model_info", None)
    model_details = info_provider(modelo_riesgo) if callable(info_provider) else {
        "prediction_period": risk_model.prediction_period,
        "supports_turns": False, "unit": "segmento",
    }
    try:
        graph, start_node, end_node, node_to_latlng = _build_osm_graph_base(
            origin, destination
        )
        graph_source = "OpenStreetMap local"
    except (nx.NetworkXException, ValueError, RuntimeError, OSError) as error:
        raise ValueError(
            "No se pudo encontrar una ruta conectada en la red vial local. "
            "Selecciona puntos cercanos a calles dentro de Lima Metropolitana."
        ) from error
    graph_ready_at = perf_counter()

    risk_summary = _assign_segment_risks(
        graph,
        node_to_latlng,
        risk_model,
        buffer_m,
        risk_mode,
        modelo_riesgo,
        turno,
    )
    # En el modo mensual asigna a las calles el mismo campo de riesgo que representa el mapa.
    field_info = None
    if risk_mode == "predicted" and turno is None:
        field_info = assign_heatmap_scores(graph, node_to_latlng,
            risk_model.spatial_risk_surface(), modelo_riesgo,
            [origin, destination, node_to_latlng(start_node), node_to_latlng(end_node)])
        field_scores = [data["risk_segment_normalized"] for _, _, _, data in _iter_edges(graph)]
        risk_summary.update(min=min(field_scores, default=0), max=max(field_scores, default=0),
                            p95=float(np.percentile(field_scores, 95)) if field_scores else 0)
    risks_ready_at = perf_counter()
    path_cache = WeakKeyDictionary()
    summary_cache = {}

    def solve(target, weight):
        field_cost = bool(target.graph.get("risk_field_version"))
        if weight == "cost_safe":
            key = ("field_cost" if field_cost else "cost_safe", target.graph["safe_beta"])
        elif weight == "cost_hotspot" and field_cost:
            key = ("field_cost", 10)
        else:
            key = weight
        cached = path_cache.setdefault(target, {})
        if key not in cached:
            cached[key] = _compute_path(target, start_node, end_node, node_to_latlng, weight)
        return cached[key]

    def summarize(target, path, weight):
        key = tuple((u, v, _best_edge_data(target, u, v, weight).get("edge_key", "0"))
                    for u, v in zip(path, path[1:]))
        if key not in summary_cache:
            summary_cache[key] = _route_metrics(target, path, origin, destination, node_to_latlng, weight)
        return summary_cache[key]

    # Calcula la referencia por distancia y fija el presupuesto máximo para las alternativas.
    fast_path = solve(graph, "cost_fast")
    fast_route = summarize(graph, fast_path, "cost_fast")

    distance_limit_km = fast_route["distance_km"] * MAX_SAFE_DISTANCE_FACTOR
    avoidance_policy = None
    search_graph, reference_path, reference_route = graph, fast_path, fast_route
    if field_info is not None:
        search_graph, reference_path, reference_route, avoidance_policy = choose_avoidance_graph(
            graph, solve=solve, summarize=summarize, distance_limit_km=distance_limit_km,
            endpoint_info=field_info)

    safe_candidates, beta_diagnostics = search_alternatives(
        search_graph, reference_path, reference_route,
        solve=solve,
        summarize=summarize,
        configure_beta=lambda value: _set_safe_cost(search_graph, value),
        beta_values=BETA_VALUES, requested_beta=beta,
        distance_limit_km=distance_limit_km,
    )
    if field_info is not None:
        safe_candidates.append({"beta": 0, "strategy": "menor_distancia_del_grafo_admisible",
                                "route": reference_route, "weight": "cost_fast"})
    fast_signature = route_signature(fast_route)
    alternatives = []
    rejected = {"mismo_recorrido": 0, "desvio_excesivo": 0, "mejora_insuficiente": 0}
    for candidate in safe_candidates:
        candidate["reduction"] = _risk_reduction(fast_route["risk_total"], candidate["route"]["risk_total"])
        if candidate["route"]["distance_km"] > distance_limit_km + 1e-9:
            rejected["desvio_excesivo"] += 1
        elif field_info is not None:
            # La prioridad es no atravesar rojo/naranja. Un desvío puede ser
            # válido aunque su exposición lineal o su distancia sean mayores.
            alternatives.append(candidate)
        elif route_signature(candidate["route"]) == fast_signature:
            rejected["mismo_recorrido"] += 1
        elif candidate["route"]["distance_km"] > fast_route["distance_km"] * MAX_SAFE_DISTANCE_FACTOR:
            rejected["desvio_excesivo"] += 1
        elif candidate["reduction"] < MIN_RISK_REDUCTION_PERCENT:
            rejected["mejora_insuficiente"] += 1
        else:
            alternatives.append(candidate)
    # Selecciona una alternativa priorizando los cruces de riesgo y la exposición dentro del desvío permitido.
    if alternatives:
        selected = max(
            alternatives,
            key=(lambda candidate: (
                -candidate["route"].get("red_avoidable_m", 0),
                -candidate["route"].get("orange_avoidable_m", 0),
                -candidate["route"].get("red_distance_m", 0),
                -candidate["route"].get("orange_distance_m", 0),
                -candidate["route"]["risk_total"],
                -candidate["route"]["distance_km"],
            )) if field_info is not None else lambda candidate: (
                -candidate["route"]["risk_distance_by_level_m"]["alto"],
                candidate["reduction"],
                -candidate["route"]["distance_km"],
            ),
        )
        safe_beta, safe_route, reduction = selected["beta"], selected["route"], selected["reduction"]
        selected_strategy = selected["strategy"]
        selected_weight = selected["weight"]
    else:
        safe_beta = 0
        safe_route = fast_route
        reduction = 0.0
        selected_strategy = "recorrido_base"
        selected_weight = "cost_fast"

    selected_formula = (
        "distancia_m * (riesgo + 2 * max(0, (riesgo - 0.66) / 0.34)^2 + 0.25 * indicador_alto)"
        if selected_weight == "cost_hotspot" else
        "distancia_m * riesgo_segmento_normalizado" if safe_beta is None else
        "distancia_m * (1 + beta * riesgo_segmento_normalizado)"
    )
    if field_info is not None:
        selected_formula = (
            "distancia_m * (1 + beta * RISK_SCORE_medio_del_heatmap) + 200 * metros_rojos + 20 * metros_naranjas"
            if selected_weight not in {"cost_fast", "cost_exposure"} else
            "distancia_m" if selected_weight == "cost_fast" else
            "integral_vial_del_RISK_SCORE_del_heatmap"
        )

    same_route = route_signature(safe_route) == fast_signature
    if same_route:
        minimum_coincides = any(item["strategy"] == "minima_exposicion" and not item["geometry_changed"] for item in beta_diagnostics)
        if minimum_coincides:
            message = "El recorrido de menor distancia también minimiza la exposición estimada en la red evaluada."
        elif rejected["desvio_excesivo"] and not rejected["mejora_insuficiente"]:
            message = "Las alternativas encontradas superan el desvío permitido de 50 %. Los recorridos recomendados coinciden."
        else:
            message = "Las alternativas evaluadas dentro del desvío de 50 % no reducen la exposición al menos 0,5 %. Los recorridos coinciden."
        reduction = 0.0 if same_route else reduction
    else:
        distance_delta = safe_route["distance_km"] - fast_route["distance_km"]
        message = (
            f"La ruta segura reduce la exposición estimada en {reduction:.2f}% "
            f"con una variación de distancia de {distance_delta:+.2f} km."
        )
    if avoidance_policy is not None:
        exceptions = crossing_reasons(safe_route, avoidance_policy)
        avoidance_policy["crossing_reasons"] = exceptions
        avoidance_policy["red_distance_m"] = safe_route.get("red_distance_m", 0)
        avoidance_policy["orange_distance_m"] = safe_route.get("orange_distance_m", 0)
        message = ("La ruta recomendada evita los centros rojos del heatmap." if safe_route.get("red_distance_m", 0) < 0.01
                   else " ".join(exceptions))
        if same_route:
            message += " La ruta más corta coincide con el recorrido recomendado bajo este criterio."

    alternative_count = len({route_signature(candidate["route"]) for candidate in safe_candidates} | {fast_signature})
    shared_ids = {segment["id_segmento"] for segment in safe_route["segments"]} & {segment["id_segmento"] for segment in fast_route["segments"]}
    for route in (safe_route, fast_route):
        for segment in route["segments"]:
            segment["compartido"] = segment["id_segmento"] in shared_ids
    shared_distance = sum(segment["distancia_metros"] for segment in safe_route["segments"] if segment["compartido"])
    paths_ready_at = perf_counter()
    LOGGER.info(
        "route_comparison source=%s alternatives=%s model=%s buffer=%sm "
        "beta=%s fast_risk=%.4f safe_risk=%.4f fast_high=%s safe_high=%s "
        "formula=%s risk_min=%.4f risk_p95=%.4f "
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
        selected_formula,
        risk_summary["min"],
        risk_summary["p95"],
        risk_summary["max"],
    )

    for route in (safe_route, fast_route):
        route.pop("node_path", None)
        route.pop("edge_path", None)

    return {
        "safe_route": safe_route,
        "traditional_route": fast_route,
        "ruta_segura": _spanish_route_alias(safe_route),
        "ruta_rapida": _spanish_route_alias(fast_route),
        "risk_reduction": round(reduction, 2),
        "reduccion_riesgo": round(reduction, 2),
        "misma_ruta": same_route,
        "distancia_compartida_m": round(shared_distance, 2),
        "estrategia_seleccionada": selected_strategy,
        "modelo_riesgo_solicitado": modelo_riesgo,
        "modelo_usado": model_used,
        "modo_riesgo": risk_mode,
        "turno_riesgo": turno if model_details["supports_turns"] else None,
        "periodo_prediccion": model_details["prediction_period"],
        "unidad_prediccion": "segmento × mes × turno" if turno and model_details["supports_turns"] else "segmento × mes",
        "metricas_modelo": risk_model.metrics_for_model(modelo_riesgo),
        "parametros_a_star": {
            "alpha": 0 if safe_beta is None else 1,
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
            "formula": selected_formula,
            "criterio_seleccion": "evitar rojo, después naranja, luego exposición y distancia; desvío máximo de 50 %" if field_info else
                                  "menor distancia en riesgo alto entre rutas con menor exposición y desvío máximo de 50 %",
            "penalizacion_rojo": RED_PENALTY if field_info else None,
            "penalizacion_naranja": ORANGE_PENALTY if field_info else None,
        },
        "risk_field": field_info["field"] if field_info else None,
        "hotspot_policy": avoidance_policy,
        "diagnostico_beta": beta_diagnostics,
        "diagnostico_busqueda": {
            "intentos_con_ruta": len(beta_diagnostics), "recorridos_distintos": alternative_count,
            "alternativas_aceptables": len(alternatives), "descartes": rejected,
            "busqueda_exhaustiva": False,
        },
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
            "tiempo_grafo_ms": round((graph_ready_at - started_at) * 1000, 2),
            "tiempo_riesgo_ms": round(
                (risks_ready_at - graph_ready_at) * 1000, 2
            ),
            "tiempo_rutas_ms": round(
                (paths_ready_at - risks_ready_at) * 1000, 2
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
    del safety_weight
    comparison = generate_route_comparison(
        origin=origin,
        destination=destination,
        risk_model=risk_model,
        beta=10 if alpha is None else max(0, min(20, alpha * 20)),
        turno=turno,
    )
    return comparison["safe_route"]


# Extrae la red necesaria para el recorrido y asocia sus extremos con los nodos viales cercanos.
def _build_osm_graph_base(
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> tuple[nx.MultiDiGraph, int, int, Callable[[int], tuple[float, float]]]:
    base_graph, node_ids, coordinates, node_tree = _load_local_road_network()
    distances, indices = node_tree.query(
        np.radians(np.asarray([origin, destination])),
        k=1,
    )
    if float(distances.max()) * EARTH_RADIUS_M > 3_000:
        raise ValueError("Los puntos están fuera de la red vial local.")
    start_node = node_ids[int(indices[0][0])]
    end_node = node_ids[int(indices[1][0])]
    def base_node_to_latlng(node: int) -> tuple[float, float]:
        return float(base_graph.nodes[node]["y"]), float(base_graph.nodes[node]["x"])

    shortest_path = _compute_path(base_graph, start_node, end_node, base_node_to_latlng, "length")
    shortest_length = sum(float(_best_edge_data(base_graph, u, v, "length")["length"])
                          for u, v in zip(shortest_path, shortest_path[1:]))
    access_length = haversine_m(*origin, *base_node_to_latlng(start_node)) + haversine_m(*base_node_to_latlng(end_node), *destination)
    road_budget = MAX_SAFE_DISTANCE_FACTOR * (shortest_length + access_length) - access_length
    # Todo nodo de una alternativa admisible cumple esta cota por distancia.
    # Así no se recorta el grafo solo porque ya existe una conexión directa.
    destination_distances = _distances_to_point_m(coordinates, base_node_to_latlng(end_node))
    lower_bound = _distances_to_point_m(coordinates, base_node_to_latlng(start_node)) + destination_distances
    selected = np.flatnonzero(lower_bound <= road_budget + 1.0)
    selected_nodes = {node_ids[index] for index in selected} | set(shortest_path)
    graph = base_graph.subgraph(selected_nodes).copy()

    def node_to_latlng(node: int) -> tuple[float, float]:
        data = graph.nodes[node]
        return float(data["y"]), float(data["x"])

    node_to_latlng.route_destination = end_node
    node_to_latlng.destination_distances = {node_ids[index]: float(destination_distances[index]) for index in selected}
    for node in shortest_path:
        if node not in node_to_latlng.destination_distances:
            node_to_latlng.destination_distances[node] = haversine_m(*node_to_latlng(node), *node_to_latlng(end_node))

    return graph, start_node, end_node, node_to_latlng


def _distances_to_point_m(coordinates: np.ndarray, point: tuple[float, float]) -> np.ndarray:
    latitudes, longitudes = np.radians(coordinates).T
    latitude, longitude = np.radians(point)
    value = np.sin((latitudes - latitude) / 2) ** 2 + np.cos(latitude) * np.cos(latitudes) * np.sin((longitudes - longitude) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


@lru_cache(maxsize=1)
# Carga y conserva en memoria la red local y el índice utilizado para localizar nodos.
def _load_local_road_network() -> tuple[
    nx.MultiDiGraph,
    tuple[int, ...],
    np.ndarray,
    BallTree,
]:
    if not ROAD_GRAPH_PATH.exists():
        raise FileNotFoundError(f"No existe la red vial local: {ROAD_GRAPH_PATH}")
    graph = ox.load_graphml(ROAD_GRAPH_PATH)
    node_ids = tuple(graph.nodes)
    coordinates = np.asarray(
        [
            [
                float(graph.nodes[node]["y"]),
                float(graph.nodes[node]["x"]),
            ]
            for node in node_ids
        ],
        dtype=float,
    )
    node_tree = BallTree(np.radians(coordinates), metric="haversine")
    return graph, node_ids, coordinates, node_tree


def preload_road_network() -> dict:
    graph, node_ids, _, _ = _load_local_road_network()
    return {
        "nodes": len(node_ids),
        "edges": graph.number_of_edges(),
        "source": ROAD_GRAPH_PATH.name,
    }


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


# Anota en cada arista el riesgo predicho, histórico o combinado y los costos correspondientes.
def _assign_segment_risks(
    graph,
    node_to_latlng,
    risk_model: RiskModel,
    buffer_m: int,
    risk_mode: str,
    modelo_riesgo: str,
    turno: str | None = None,
) -> dict:
    if risk_mode not in {"predicted", "historical", "hybrid"}:
        raise ValueError(f"Modo de riesgo no soportado: {risk_mode}")
    needs_historical = risk_mode in {"historical", "hybrid"}
    needs_prediction = risk_mode in {"predicted", "hybrid"}
    prediction_lookup = (risk_model.segment_predictions(modelo_riesgo, turno)
                         if needs_prediction and hasattr(risk_model, "segment_predictions") else {})
    edge_rows = []
    for u, v, key, data in _iter_edges(graph):
        start = node_to_latlng(u)
        end = node_to_latlng(v)
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        length = float(data.get("length") or haversine_m(*start, *end))
        crime_stats = (
            _nearby_stats_with_turno(
                risk_model,
                _edge_sample_points(data, start, end),
                buffer_m,
                turno,
            )
            if needs_historical
            else {"count": 0, "weight_sum": 0.0, "weight_avg": 0.0}
        )
        raw_risk = crime_stats["weight_sum"] / max(length / 100, 1)
        tramo_id = construir_id_segmento(u, v, key, data)
        prediction = (
            (prediction_lookup.get(tramo_id) or _predict_segment_with_model(risk_model, tramo_id, midpoint, modelo_riesgo, turno))
            if needs_prediction
            else RiskPrediction(score=0.0, level="bajo")
        )
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
                "edge_key": str(row["key"]),
                "distance_m": row["length"],
                "time_min": row["length"] / (DEFAULT_SPEED_KMH * 1000 / 60),
                "nearby_crime_count": row["crime_stats"]["count"],
                "crime_weight_sum": row["crime_stats"]["weight_sum"],
                "crime_weight_avg": row["crime_stats"]["weight_avg"],
                "risk_segment_raw": row["raw_risk"],
                "risk_historical_normalized": historical,
                "risk_predicted": predicted,
                "risk_segment_normalized": normalized,
                "risk_level": (
                    row["prediction"].level
                    if risk_mode == "predicted"
                    else level_from_score(normalized)
                ),
                "cost_fast": row["length"],
                "cost_exposure": row["length"] * normalized,
                "cost_hotspot": row["length"] * (
                    normalized + HOTSPOT_PENALTY_WEIGHT * (max(0.0, (normalized - 0.66) / 0.34) ** 2)
                    + (0.25 if (row["prediction"].level if risk_mode == "predicted" else level_from_score(normalized)) == "alto" else 0.0)
                ),
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


def _predict_segment_with_model(
    risk_model: RiskModel,
    tramo_id: str,
    midpoint: tuple[float, float],
    modelo_riesgo: str,
    turno: str | None = None,
) -> RiskPrediction:
    try:
        return risk_model.predict_segment(tramo_id, midpoint, modelo_riesgo, turno=turno)
    except TypeError:
        try:
            return risk_model.predict_segment(tramo_id, midpoint, modelo_riesgo)
        except TypeError:
            return risk_model.predict_segment(tramo_id, midpoint)


def _nearby_stats_with_turno(
    risk_model: RiskModel,
    points: list[tuple[float, float]],
    radius_m: float,
    turno: str | None,
) -> dict:
    if turno:
        try:
            return risk_model.nearby_crime_stats(points, radius_m, turno=turno)
        except TypeError:
            pass
    return risk_model.nearby_crime_stats(points, radius_m)


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


# Actualiza el costo de búsqueda para la penalización de riesgo solicitada.
def _set_safe_cost(graph, beta: float) -> None:
    graph.graph["safe_beta"] = beta
    for _, _, _, data in _iter_edges(graph):
        data["cost_safe"] = field_edge_cost(data, beta) if data.get("risk_field_version") else _edge_cost(
            float(data["distance_m"]),
            float(data["risk_segment_normalized"]),
            beta,
        )


def _edge_cost(distance_m: float, risk_score: float, beta: float) -> float:
    return distance_m * (1 + beta * risk_score)


# Ejecuta A* con una heurística de distancia y el costo seleccionado para cada arista.
def _compute_path(graph, start_node, end_node, node_to_latlng, weight):
    use_distance = weight != "cost_exposure" and (weight != "cost_hotspot" or graph.graph.get("risk_field_version"))
    distances = (getattr(node_to_latlng, "destination_distances", None)
                 if getattr(node_to_latlng, "route_destination", None) == end_node else None)
    if not use_distance:
        heuristic = lambda a, b: 0.0
    elif distances is not None:
        heuristic = lambda a, b: distances[a]
    else:
        heuristic = lambda a, b: haversine_m(*node_to_latlng(a), *node_to_latlng(b))
    return nx.astar_path(
        graph,
        start_node,
        end_node,
        heuristic=heuristic,
        weight=weight,
    )


# Acumula distancia, exposición, cruces y segmentos de alto riesgo a lo largo del recorrido.
def _route_metrics(
    graph,
    path,
    origin,
    destination,
    node_to_latlng,
    weight,
) -> dict:
    segments = []
    path_coords = []
    for index, (u, v) in enumerate(zip(path, path[1:]), start=1):
        data = _best_edge_data(graph, u, v, weight)
        coordinates = _edge_route_points(data, node_to_latlng(u), node_to_latlng(v))
        path_coords.extend(coordinates if not path_coords else coordinates[1:])
        segments.append(
            {
                "id_segmento": str(data["id_segmento"]),
                "nodo_origen": str(u),
                "nodo_destino": str(v),
                "clave_arista": str(data.get("edge_key", "0")),
                "distancia_metros": round(float(data["distance_m"]), 2),
                "tiempo_min": round(float(data["time_min"]), 3),
                "cantidad_delitos_cercanos": int(data["nearby_crime_count"]),
                "peso_delito_acumulado": round(float(data["crime_weight_sum"]), 3),
                "riesgo_segmento": round(float(data["risk_segment_raw"]), 6),
                "riesgo_historico": round(
                    float(data["risk_historical_normalized"]), 6
                ),
                "riesgo_predicho": round(float(data["risk_predicted"]), 6),
                "nivel_modelo": str(data.get("risk_model_level", data["risk_level"])),
                "riesgo_mapa_maximo": round(float(data.get("risk_map_max", data["risk_segment_normalized"])), 6),
                "metros_rojos": round(float(data.get("red_m", 0)), 3),
                "metros_naranjas": round(float(data.get("orange_m", 0)), 3),
                "metros_rojos_evitables": round(float(data.get("red_avoidable_m", 0)), 3),
                "metros_naranjas_evitables": round(float(data.get("orange_avoidable_m", 0)), 3),
                "riesgo_segmento_normalizado": round(
                    float(data["risk_segment_normalized"]), 6
                ),
                "nivel_riesgo": str(data["risk_level"]),
                "coordenadas": [
                    {"lat": round(lat, 6), "lng": round(lng, 6)}
                    for lat, lng in coordinates
                ],
                "orden": index,
            }
        )
    if not path_coords:
        path_coords = [node_to_latlng(node) for node in path]
    connector_distance = haversine_m(*origin, *path_coords[0]) + haversine_m(
        *path_coords[-1], *destination
    )
    segment_distance = sum(segment["distancia_metros"] for segment in segments)
    distance = connector_distance + segment_distance
    distance_by_level = {
        level: sum(segment["distancia_metros"] for segment in segments if segment["nivel_riesgo"] == level)
        for level in ("bajo", "medio", "alto")
    }
    weighted_risk_m = sum(
        segment["riesgo_segmento_normalizado"] * segment["distancia_metros"]
        for segment in segments
    )
    risk_total = weighted_risk_m / 1000.0
    risk_average = weighted_risk_m / segment_distance if segment_distance else 0.0
    field_version = next((data.get("risk_field_version") for _, _, _, data in _iter_edges(graph)
                          if data.get("risk_field_version")), None)
    return {
        "route": [
            {"lat": round(lat, 6), "lng": round(lng, 6)}
            for lat, lng in path_coords
        ],
        "distance_km": round(distance / 1000, 3),
        "time_min": round(distance / (DEFAULT_SPEED_KMH * 1000 / 60), 1),
        "risk_total": round(risk_total, 6),
        "risk_score": round(risk_average, 6),
        "risk_average": round(risk_average, 6),
        "risk_score_source": "shared_heatmap_field" if field_version else "segment_prediction",
        "risk_field_version": field_version,
        "model_risk_average": round(sum(segment["riesgo_predicho"] * segment["distancia_metros"] for segment in segments)
                                     / segment_distance, 6) if segment_distance else 0,
        "red_distance_m": round(sum(segment["metros_rojos"] for segment in segments), 3),
        "orange_distance_m": round(sum(segment["metros_naranjas"] for segment in segments), 3),
        "red_avoidable_m": round(sum(segment["metros_rojos_evitables"] for segment in segments), 3),
        "orange_avoidable_m": round(sum(segment["metros_naranjas_evitables"] for segment in segments), 3),
        "risk_level": level_from_score(risk_average),
        "high_risk_segments": sum(segment["nivel_riesgo"] == "alto" for segment in segments),
        "risk_distance_by_level_m": {level: round(value, 2) for level, value in distance_by_level.items()},
        "access_distance_m": round(connector_distance, 2),
        "access_connectors": [
            [{"lat": round(lat, 6), "lng": round(lng, 6)} for lat, lng in connection]
            for connection in ((origin, path_coords[0]), (path_coords[-1], destination))
            if haversine_m(*connection[0], *connection[1]) > 2
        ],
        "segments": segments,
        "node_path": [str(node) for node in path],
        "edge_path": [f'{segment["nodo_origen"]}|{segment["nodo_destino"]}|{segment["clave_arista"]}' for segment in segments],
    }


def _edge_route_points(data, start, end) -> list[tuple[float, float]]:
    """Conserva la geometría vial y la orienta en el sentido del recorrido."""
    geometry = data.get("geometry")
    points = [(float(y), float(x)) for x, y, *_ in geometry.coords] if geometry is not None and hasattr(geometry, "coords") else []
    if points and haversine_m(*start, *points[-1]) < haversine_m(*start, *points[0]):
        points.reverse()
    coordinates = []
    for point in (start, *points, end):
        if not coordinates or haversine_m(*coordinates[-1], *point) > 0.1:
            coordinates.append(point)
    return coordinates


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


# Expresa la diferencia de exposición acumulada respecto de la ruta de referencia.
def _risk_reduction(fast_risk: float, safe_risk: float) -> float:
    if fast_risk <= 1e-12:
        return 0.0
    return round(max(0.0, (fast_risk - safe_risk) / fast_risk * 100), 2)


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
