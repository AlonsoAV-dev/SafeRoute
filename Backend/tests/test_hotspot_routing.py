from __future__ import annotations

import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
from shapely import LineString

from app.services.hotspot_routing import assign_heatmap_scores, choose_avoidance_graph
from app.services.risk_surface import RiskSurface, TO_COORDINATES, TO_METERS


class FieldModel:
    def model_info(self, model):
        return {"key": model, "prediction_period": "2026-09"}


class RoutingFieldTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y = TO_METERS.transform(-77.0, -12.0)
        self.surface = RiskSurface(FieldModel())
        self.samples = patch("app.services.risk_surface._sample_road_geometry", return_value=object())
        self.hotspots = patch.object(self.surface, "_hotspots", side_effect=lambda samples, model:
            np.array([[self.x, self.y, .95 if model == "rf" else .74]]))
        self.samples.start()
        self.hotspots.start()
        self.addCleanup(self.samples.stop)
        self.addCleanup(self.hotspots.stop)

    def coordinate(self, x, y):
        lng, lat = TO_COORDINATES.transform(x, y)
        return lat, lng

    def road(self):
        positions = {1: self.coordinate(self.x-200, self.y), 2: self.coordinate(self.x+200, self.y)}
        line = LineString([(lng, lat) for lat, lng in positions.values()])
        graph = nx.MultiDiGraph()
        graph.add_edge(1, 2, id_segmento="road", geometry=line, distance_m=400,
                       risk_level="alto", risk_predicted=.95)
        return graph, positions

    def test_cached_scores_keep_endpoint_exceptions_request_specific(self):
        graph, positions = self.road()
        with patch.object(self.surface, "scores_at_xy", wraps=self.surface.scores_at_xy) as sampled:
            assign_heatmap_scores(graph, positions.__getitem__, self.surface, "rf", list(positions.values()))
            original = dict(graph[1][2][0])
            self.assertTrue(original["red_blocked"])
            self.assertGreater(original["red_m"], 0)
            second, positions = self.road()
            info = assign_heatmap_scores(second, positions.__getitem__, self.surface, "rf",
                                        [self.coordinate(self.x, self.y), positions[2]])
            self.assertTrue(info["endpoint_red"])
            self.assertFalse(second[1][2][0]["red_blocked"])
            self.assertEqual(second[1][2][0]["red_m"], original["red_m"])
            self.assertEqual(second[1][2][0]["risk_segment_normalized"], original["risk_segment_normalized"])
            self.assertEqual(sampled.call_count, 1)

    def test_cache_does_not_mix_model_fields(self):
        first, positions = self.road()
        second, _ = self.road()
        assign_heatmap_scores(first, positions.__getitem__, self.surface, "rf", list(positions.values()))
        assign_heatmap_scores(second, positions.__getitem__, self.surface, "xgb", list(positions.values()))
        self.assertGreater(first[1][2][0]["risk_segment_normalized"], second[1][2][0]["risk_segment_normalized"])
        self.assertTrue(first[1][2][0]["red_blocked"])
        self.assertFalse(second[1][2][0]["red_blocked"])

    def test_batch_intersections_match_individual_polygon_unions(self):
        geometries = np.array([
            LineString([(self.x-200, self.y), (self.x+700, self.y)]),
            LineString([(self.x, self.y-200), (self.x, self.y+200)]),
            LineString([(self.x-200, self.y+100), (self.x+700, self.y+100)]),
            LineString([(self.x-200, self.y+400), (self.x+700, self.y+400)]),
        ], dtype=object)
        with patch.object(self.surface, "_hotspots", return_value=np.array([
                [self.x, self.y, .95], [self.x+500, self.y, .85]])):
            batch = self.surface.road_zones_many(geometries, "rf")
            for geometry, result in zip(geometries, batch):
                exact = self.surface.road_zones(geometry, "rf")
                self.assertEqual(set(result["red_ids"]), set(exact["red_ids"]))
                self.assertEqual(set(result["orange_ids"]), set(exact["orange_ids"]))
                self.assertAlmostEqual(result["red_m"], exact["red_m"], places=6)
                self.assertAlmostEqual(result["orange_m"], exact["orange_m"], places=6)


class AvoidanceTests(unittest.TestCase):
    def graph(self, detour):
        graph = nx.MultiDiGraph()
        for u, v, length, red in (("a", "b", 100, True), ("b", "d", 100, True),
                                   ("a", "c", detour/2, False), ("c", "d", detour/2, False)):
            graph.add_edge(u, v, cost_fast=length, red_blocked=red, orange_blocked=False,
                           red_clearance_blocked=red, orange_clearance_blocked=False)
        return graph

    def choose(self, graph):
        solve = lambda target, weight: nx.astar_path(target, "a", "d", weight=weight)
        summarize = lambda target, path, weight: {"distance_km":
            sum(min(edge[weight] for edge in target[u][v].values()) for u, v in zip(path, path[1:]))/1000}
        return choose_avoidance_graph(graph, solve=solve, summarize=summarize,
            distance_limit_km=.3, endpoint_info={"endpoint_red": False, "endpoint_orange": False})

    def test_red_is_avoided_when_a_route_fits_the_budget(self):
        graph = self.graph(240)
        active, path, route, policy = self.choose(graph)
        self.assertEqual(path, ["a", "c", "d"])
        self.assertEqual(policy["red"], "avoided")
        self.assertLessEqual(route["distance_km"], .3)
        self.assertEqual(graph.number_of_edges(), 4)
        self.assertFalse(active.has_edge("a", "b"))

    def test_red_exception_keeps_the_detour_limit(self):
        graph = self.graph(360)
        _, path, _, policy = self.choose(graph)
        self.assertEqual(path, ["a", "b", "d"])
        self.assertEqual(policy["red"], "detour_limit")

    def test_red_exception_handles_a_disconnected_filtered_network(self):
        graph = self.graph(240)
        graph.remove_node("c")
        _, path, _, policy = self.choose(graph)
        self.assertEqual(path, ["a", "b", "d"])
        self.assertEqual(policy["red"], "no_viable_path")


if __name__ == "__main__":
    unittest.main()
