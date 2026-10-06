from __future__ import annotations

import networkx as nx


MAX_ADAPTIVE_STEPS = 6
MAX_BRANCH_SEARCHES = 6


# Identifica el recorrido por sus segmentos para distinguir alternativas geométricas reales.
def route_signature(route: dict) -> tuple:
    return tuple(route.get("edge_path") or route.get("node_path", ()))


# Explora costos y penalizaciones, reutilizando soluciones y respetando el presupuesto de distancia.
def search_alternatives(graph, fast_path, fast_route, *, solve, summarize,
                        configure_beta, beta_values, requested_beta, distance_limit_km):
    """Amplía A* con mínimo de exposición, pesos intermedios y desvíos reales.

    Las exclusiones de aristas generan candidatos; sus riesgos se vuelven a
    medir con los scores asignados al grafo. La selección conserva el límite de desvío.
    La búsqueda es acotada y no afirma enumerar todas las rutas posibles.
    """
    candidates = {}
    diagnostics = []
    fast_signature = route_signature(fast_route)

    def evaluate(target_graph, weight, strategy, beta):
        try:
            path = solve(target_graph, weight)
        except nx.NetworkXNoPath:
            return None
        route = summarize(target_graph, path, weight)
        signature = route_signature(route)
        diagnostic = {
            "beta": beta, "strategy": strategy, "distance_km": route["distance_km"],
            "risk_total": route["risk_total"], "risk_average": route["risk_score"],
            "high_risk_distance_m": route["risk_distance_by_level_m"]["alto"],
            "geometry_changed": signature != fast_signature,
            "within_distance_limit": route["distance_km"] <= distance_limit_km + 1e-9,
        }
        diagnostics.append(diagnostic)
        candidates.setdefault(signature, {"beta": beta, "strategy": strategy, "route": route, "weight": weight})
        return route

    anchors = []
    for beta in sorted({*beta_values, float(requested_beta)}):
        configure_beta(beta)
        route = evaluate(graph, "cost_safe", "ponderacion_riesgo", beta)
        if route:
            anchors.append((beta, route))

    minimum_exposure = evaluate(graph, "cost_exposure", "minima_exposicion", None)
    hotspot_avoidance = evaluate(graph, "cost_hotspot", "evitar_focos_altos", None)
    # Si el mínimo de exposición ya coincide con la base, un desvío no la reduce.
    if (minimum_exposure and route_signature(minimum_exposure) == fast_signature
            and (not hotspot_avoidance or route_signature(hotspot_avoidance) == fast_signature)):
        return list(candidates.values()), diagnostics

    if minimum_exposure and minimum_exposure["distance_km"] > distance_limit_km and anchors:
        upper_beta, upper_route = anchors[-1]
        for _ in range(4):
            if upper_route["distance_km"] > distance_limit_km:
                break
            upper_beta *= 2
            configure_beta(upper_beta)
            upper_route = evaluate(graph, "cost_safe", "ponderacion_ampliada", upper_beta)
            if upper_route is None:
                break
            anchors.append((upper_beta, upper_route))

    for (low, low_route), (high, high_route) in zip(anchors, anchors[1:]):
        if low_route["distance_km"] <= distance_limit_km < high_route["distance_km"]:
            for _ in range(MAX_ADAPTIVE_STEPS):
                middle = (low + high) / 2
                configure_beta(middle)
                route = evaluate(graph, "cost_safe", "ponderacion_intermedia", middle)
                if route is None:
                    break
                if route["distance_km"] <= distance_limit_km:
                    low = middle
                else:
                    high = middle
            break

    branch_edges = []
    for index, (u, v) in enumerate(zip(fast_path, fast_path[1:])):
        if len(graph[u]) < 2:
            continue
        data = graph.get_edge_data(u, v)
        data = min(data.values(), key=lambda item: item["cost_fast"]) if graph.is_multigraph() else data
        exposure = data["distance_m"] * data["risk_segment_normalized"]
        branch_edges.append((data["risk_level"] == "alto", exposure, index, u, v))
    chosen = []
    spacing = max(1, len(fast_path) // (MAX_BRANCH_SEARCHES + 1))
    for _, _, index, u, v in sorted(branch_edges, reverse=True):
        if any(abs(index - previous[0]) < spacing for previous in chosen):
            continue
        chosen.append((index, u, v))
        if len(chosen) >= MAX_BRANCH_SEARCHES:
            break
    configure_beta(float(requested_beta))
    for _, blocked_u, blocked_v in chosen:
        view = graph.copy()
        if graph.is_multigraph():
            view.remove_edges_from([(blocked_u, blocked_v, key) for key in graph[blocked_u][blocked_v]])
        else:
            view.remove_edge(blocked_u, blocked_v)
        evaluate(view, "cost_safe", "desvio_vial", float(requested_beta))
        evaluate(view, "cost_exposure", "desvio_minima_exposicion", None)
        evaluate(view, "cost_hotspot", "desvio_evitar_focos", None)
    return list(candidates.values()), diagnostics
