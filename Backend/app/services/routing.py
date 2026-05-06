from __future__ import annotations

from math import ceil

import networkx as nx

from app.services.risk_model import RiskModel, haversine_m, level_from_score


def generate_safe_route(
    origin: tuple[float, float],
    destination: tuple[float, float],
    turno: str,
    risk_model: RiskModel,
    safety_weight: float,
) -> dict:
    graph = _build_grid_graph(origin, destination, turno, risk_model, safety_weight)
    start_node = min(graph.nodes, key=lambda node: haversine_m(origin[0], origin[1], node[0], node[1]))
    end_node = min(graph.nodes, key=lambda node: haversine_m(destination[0], destination[1], node[0], node[1]))

    path = nx.astar_path(
        graph,
        start_node,
        end_node,
        heuristic=lambda a, b: haversine_m(a[0], a[1], b[0], b[1]),
        weight="cost",
    )

    route = [origin, *path[1:-1], destination]
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


def _build_grid_graph(
    origin: tuple[float, float],
    destination: tuple[float, float],
    turno: str,
    risk_model: RiskModel,
    safety_weight: float,
) -> nx.Graph:
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
            graph.add_edge((lat, lng), neighbor, cost=distance * (1 + safety_weight * risk))

    return graph


def _average_route_risk(
    route: list[tuple[float, float]],
    turno: str,
    risk_model: RiskModel,
) -> float:
    scores = [risk_model.predict_point(lat, lng, turno).score for lat, lng in route]
    return sum(scores) / len(scores)

