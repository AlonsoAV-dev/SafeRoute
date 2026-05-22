from __future__ import annotations

from math import ceil
from typing import Callable

import networkx as nx
import osmnx as ox

from app.services.risk_model import RiskModel, haversine_m, level_from_score


def generate_safe_route(
    origin: tuple[float, float],
    destination: tuple[float, float],
    turno: str,
    risk_model: RiskModel,
    safety_weight: float,
    alpha: float | None = None,
) -> dict:
    def compute_path(graph, start_node, end_node, node_to_latlng):
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
            weight="cost",
        )

    try:
        graph, start_node, end_node, node_to_latlng = _build_osm_graph(
            origin, destination, turno, risk_model, safety_weight, alpha
        )
        path = compute_path(graph, start_node, end_node, node_to_latlng)
    except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError):
        graph, start_node, end_node, node_to_latlng = _build_grid_graph(
            origin, destination, turno, risk_model, safety_weight, alpha
        )
        try:
            path = compute_path(graph, start_node, end_node, node_to_latlng)
        except (nx.NetworkXNoPath, nx.NodeNotFound) as error:
            raise ValueError("No se encontró una ruta entre el origen y destino.") from error

    path_coords = [node_to_latlng(node) for node in path]
    route = [origin, *path_coords, destination]
    distance = sum(
        haversine_m(a[0], a[1], b[0], b[1])
        for a, b in zip(route, route[1:])
    )
    risk_score = _average_route_risk(route, turno, risk_model)

    return {
        "route": [{"lat": round(lat, 6), "lng": round(lng, 6)} for lat, lng in route],
        "distance_km": round(distance / 1000, 2),
        "risk_score": round(risk_score, 3),
        "risk_level": level_from_score(risk_score),
    }


def _build_osm_graph(
    origin: tuple[float, float],
    destination: tuple[float, float],
    turno: str,
    risk_model: RiskModel,
    safety_weight: float,
    alpha: float | None,
) -> tuple[nx.MultiDiGraph, int, int, Callable[[int], tuple[float, float]]]:
    ox.settings.use_cache = True
    ox.settings.log_console = False

    mid_lat = (origin[0] + destination[0]) / 2
    mid_lng = (origin[1] + destination[1]) / 2
    straight_distance = haversine_m(origin[0], origin[1], destination[0], destination[1])
    min_radius = 2000
    max_radius = 6000
    radius = min(max_radius, max(min_radius, straight_distance / 2 + 1000))

    graph = ox.graph_from_point(
        center_point=(mid_lat, mid_lng),
        dist=radius,
        network_type="drive",
        simplify=True,
    )
    graph = ox.distance.add_edge_lengths(graph)

    node_risk = {
        node: risk_model.predict_point(data["y"], data["x"], turno).score
        for node, data in graph.nodes(data=True)
    }

    for u, v, key, data in graph.edges(keys=True, data=True):
        length = data.get("length")
        if length is None:
            length = haversine_m(graph.nodes[u]["y"], graph.nodes[u]["x"], graph.nodes[v]["y"], graph.nodes[v]["x"])
        risk = (node_risk[u] + node_risk[v]) / 2
        data["cost"] = _edge_cost(length, risk, safety_weight, alpha)

    start_node = ox.distance.nearest_nodes(graph, origin[1], origin[0])
    end_node = ox.distance.nearest_nodes(graph, destination[1], destination[0])

    def node_to_latlng(node: int) -> tuple[float, float]:
        data = graph.nodes[node]
        return (float(data["y"]), float(data["x"]))

    return graph, start_node, end_node, node_to_latlng


def _build_grid_graph(
    origin: tuple[float, float],
    destination: tuple[float, float],
    turno: str,
    risk_model: RiskModel,
    safety_weight: float,
    alpha: float | None,
) -> tuple[nx.Graph, tuple[float, float], tuple[float, float], Callable[[tuple[float, float]], tuple[float, float]]]:
    min_lat, max_lat = sorted([origin[0], destination[0]])
    min_lng, max_lng = sorted([origin[1], destination[1]])
    padding = 0.018
    min_lat -= padding
    max_lat += padding
    min_lng -= padding
    max_lng += padding

    steps = max(8, min(18, ceil(max(max_lat - min_lat, max_lng - min_lng) / 0.004)))
    lat_step = (max_lat - min_lat) / steps
    lng_step = (max_lng - min_lng) / steps

    graph = nx.Graph()
    nodes = [
        (round(min_lat + row * lat_step, 6), round(min_lng + col * lng_step, 6))
        for row in range(steps + 1)
        for col in range(steps + 1)
    ]
    graph.add_nodes_from(nodes)

    node_set = set(nodes)
    for lat, lng in nodes:
        for neighbor in (
            (round(lat + lat_step, 6), lng),
            (lat, round(lng + lng_step, 6)),
            (round(lat + lat_step, 6), round(lng + lng_step, 6)),
            (round(lat + lat_step, 6), round(lng - lng_step, 6)),
        ):
            if neighbor not in node_set:
                continue

            distance = haversine_m(lat, lng, neighbor[0], neighbor[1])
            risk = (
                risk_model.predict_point(lat, lng, turno).score
                + risk_model.predict_point(neighbor[0], neighbor[1], turno).score
            ) / 2
            graph.add_edge((lat, lng), neighbor, cost=_edge_cost(distance, risk, safety_weight, alpha))

    start_node = min(graph.nodes, key=lambda node: haversine_m(origin[0], origin[1], node[0], node[1]))
    end_node = min(graph.nodes, key=lambda node: haversine_m(destination[0], destination[1], node[0], node[1]))

    def node_to_latlng(node: tuple[float, float]) -> tuple[float, float]:
        return node

    return graph, start_node, end_node, node_to_latlng


def _average_route_risk(
    route: list[tuple[float, float]],
    turno: str,
    risk_model: RiskModel,
) -> float:
    scores = [risk_model.predict_point(lat, lng, turno).score for lat, lng in route]
    return sum(scores) / len(scores)


def _edge_cost(distance_m: float, risk_score: float, safety_weight: float, alpha: float | None) -> float:
    if alpha is not None:
        return distance_m * ((1 - alpha) + alpha * risk_score)
    return distance_m * (1 + safety_weight * risk_score)

