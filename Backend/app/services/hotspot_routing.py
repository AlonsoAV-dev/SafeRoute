from __future__ import annotations

import networkx as nx
import shapely
from shapely import LineString

from app.services.risk_model import level_from_score


# Fija el paso de integración vial y las penalizaciones por metro en zonas rojas y naranjas.
ROAD_INTEGRATION_STEP_M = 10.0
RED_PENALTY = 200.0
ORANGE_PENALTY = 20.0


# Integra el campo compartido sobre las calles y conserva por separado el score original del modelo.
def assign_heatmap_scores(graph, node_to_latlng, surface, model, endpoints):
    """Integra el campo que dibuja el mapa a lo largo de la geometría completa.

    Conserva el score original del modelo en risk_predicted. RISK_SCORE para
    ruteo es la media espacial por tramo; el cruce de rojo/naranja se obtiene
    por intersección geométrica, sin depender de esa media ni del punto medio.
    """
    edges = list(graph.edges(keys=True, data=True))
    roads = {}
    for u, v, _, data in edges:
        segment_id = data["id_segmento"]
        if segment_id not in roads:
            geometry = data.get("geometry")
            if geometry is None:
                start, end = node_to_latlng(u), node_to_latlng(v)
                geometry = LineString([(start[1], start[0]), (end[1], end[0])])
            roads[segment_id] = geometry
    ids = tuple(roads)
    road_info = surface.routing_road_info(roads, model, ROAD_INTEGRATION_STEP_M)
    exempt = surface.endpoint_zones(endpoints, model)
    field = surface.field_index(model)
    info = {}
    for segment_id in ids:
        cached = road_info[segment_id]
        geometry = cached["geometry"]
        zones = cached["zones"]
        avoid_ids = {color: tuple(zone for zone in zones[f"{color}_ids"] if zone not in exempt)
                     for color in ("red", "orange")}
        # Solo las calles próximas a un extremo necesitan recalcular excepciones.
        if exempt.intersection((*zones["red_ids"], *zones["orange_ids"])):
            avoid_pieces = {color: shapely.intersection(geometry, shapely.union_all(field.zone_polygons[list(zone_ids)]))
                            if zone_ids else LineString() for color, zone_ids in avoid_ids.items()}
            red_avoidable = float(shapely.length(avoid_pieces["red"]))
            orange_avoidable = float(shapely.length(shapely.difference(avoid_pieces["orange"], avoid_pieces["red"])))
        else:
            red_avoidable, orange_avoidable = zones["red_m"], zones["orange_m"]
        info[segment_id] = {
            "mean": cached["mean"], "peak": cached["peak"], "projected_length": cached["projected_length"],
            **zones, "red_clearance_blocked": bool(avoid_ids["red"]),
            "orange_clearance_blocked": bool(avoid_ids["orange"]),
            "red_avoidable_m": red_avoidable,
            "orange_avoidable_m": orange_avoidable,
        }
    field_config = surface.field_config()
    graph.graph["risk_field_version"] = field_config["version"]
    for _, _, _, data in edges:
        row = info[data["id_segmento"]]
        ratio = float(data["distance_m"]) / max(row["projected_length"], 1e-9)
        data["risk_model_level"] = data["risk_level"]
        data["risk_segment_normalized"] = row["mean"]
        data["risk_map_max"] = row["peak"]
        data["risk_field_version"] = field_config["version"]
        data["risk_level"] = "alto" if row["red_m"] > 0 else level_from_score(row["peak"])
        for key in ("red_m", "orange_m", "red_avoidable_m", "orange_avoidable_m"):
            data[key] = row[key] * ratio
        data["red_blocked"] = row["red_avoidable_m"] > 0.001
        data["orange_blocked"] = row["orange_avoidable_m"] > 0.001
        data["red_clearance_blocked"] = row["red_clearance_blocked"]
        data["orange_clearance_blocked"] = row["orange_clearance_blocked"]
        data["cost_exposure"] = data["distance_m"] * (row["mean"] + 1e-6)
        data["cost_hotspot"] = field_edge_cost(data, 10)
        data["cost_safe"] = data["cost_hotspot"]
    return {"endpoint_red": any(field.zone_colors[index] == "red" for index in exempt),
            "endpoint_orange": any(field.zone_colors[index] == "orange" for index in exempt),
            "field": field_config}


# Combina longitud, riesgo medio y metros de cruce para calcular el costo de una arista.
def field_edge_cost(data, beta):
    return (data["distance_m"] * (1 + beta * data["risk_segment_normalized"])
            + RED_PENALTY * data["red_m"] + ORANGE_PENALTY * data["orange_m"])


# Busca primero calles sin rojo y luego sin naranja, respetando las excepciones de los extremos y el desvío máximo.
def choose_avoidance_graph(graph, *, solve, summarize, distance_limit_km, endpoint_info):
    """Demuestra viabilidad con el camino de menor distancia del grafo filtrado.

    Si el menor camino sin rojo cumple el presupuesto, todas las búsquedas
    siguientes se realizan en ese grafo. El mismo procedimiento intenta
    excluir naranja. No se exige reducir el score promedio de la calle.
    """
    policy = {**endpoint_info, "distance_limit_km": distance_limit_km,
              "red": "not_evaluated", "orange": "not_evaluated"}
    active = graph

    def exclude(target, *flags):
        # Materializar evita que filtros anidados se llamen entre sí en cada
        # acceso a una arista durante las decenas de búsquedas de A*.
        blocked = [(u, v, key) for u, v, key, data in target.edges(keys=True, data=True)
                   if any(data.get(flag, False) for flag in flags)]
        if not blocked:
            return target
        filtered = target.copy()
        filtered.remove_edges_from(blocked)
        return filtered

    baseline = None
    for color in ("red", "orange"):
        filtered = exclude(active, f"{color}_blocked")
        try:
            path = solve(filtered, "cost_fast")
            route = summarize(filtered, path, "cost_fast")
        except nx.NetworkXNoPath:
            policy[color] = "no_viable_path"
            continue
        policy[f"{color}_free_distance_km"] = route["distance_km"]
        if route["distance_km"] <= distance_limit_km + 1e-9:
            active = filtered
            baseline = (path, route)
            policy[color] = "avoided_except_endpoints" if policy[f"endpoint_{color}"] else "avoided"
        else:
            policy[color] = "detour_limit"
    if baseline is None:
        path = solve(active, "cost_fast")
        baseline = (path, summarize(active, path, "cost_fast"))
    # El margen visual es secundario: nunca invalida una alternativa que
    # evita los colores reales ni causa un falso "sin alternativa".
    avoided = [color for color in ("red", "orange") if policy[color].startswith("avoided")]
    policy["clearance_applied"] = False
    if avoided:
        clearance_graph = exclude(active, *(f"{color}_clearance_blocked" for color in avoided))
        try:
            path = solve(clearance_graph, "cost_fast")
            route = summarize(clearance_graph, path, "cost_fast")
            if route["distance_km"] <= distance_limit_km + 1e-9:
                active, baseline = clearance_graph, (path, route)
                policy["clearance_applied"] = True
        except nx.NetworkXNoPath:
            pass
    return active, baseline[0], baseline[1], policy


# Explica al usuario por qué un recorrido mantiene cruces con zonas de riesgo.
def crossing_reasons(route, policy):
    reasons = []
    labels = {"red": "rojas", "orange": "naranjas"}
    for color in ("red", "orange"):
        if route.get(f"{color}_distance_m", 0) < 0.01:
            continue
        if policy.get(f"endpoint_{color}"):
            reasons.append(f"El inicio, destino o su acceso a la red vial está dentro de una zona {labels[color]}.")
        status = policy.get(color)
        if status == "detour_limit":
            reasons.append(f"El desvío encontrado para evitar las zonas {labels[color]} supera el límite de +50 % de distancia.")
        elif status == "no_viable_path":
            reasons.append(f"No existe un recorrido vial sin zonas {labels[color]} dentro de la red admitida por el límite de +50 %.")
    return reasons
