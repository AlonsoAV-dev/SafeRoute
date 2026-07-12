from __future__ import annotations

import unittest
from unittest.mock import patch

import networkx as nx

from app.services.risk_model import RiskPrediction
from app.services.routing import _edge_cost, generate_route_comparison


class FakeRiskModel:
    model_name = "Random Forest"
    prediction_period = "2025-07"

    def resolve_model(self, requested):
        del requested
        return self.model_name

    def metrics_for_model(self, requested):
        return {"accuracy": 0.9, "f1_score": 0.88}

    def predict_point(self, lat, lng, turno=None, modelo_riesgo="auto"):
        del lat, lng, turno, modelo_riesgo
        return RiskPrediction(score=0.2, level="bajo")

    def predict_segment(self, tramo_id, midpoint):
        del tramo_id
        risky = abs(midpoint[0]) < 0.0002
        return RiskPrediction(
            score=0.9 if risky else 0.1,
            level="alto" if risky else "bajo",
        )

    def nearby_crime_stats(self, sample_points, radius_m):
        del radius_m
        midpoint = sample_points[1]
        risky = abs(midpoint[0]) < 0.0002
        return {
            "count": 5 if risky else 1,
            "weight_sum": 25.0 if risky else 1.0,
            "weight_avg": 5.0 if risky else 1.0,
        }


def build_alternative_graph():
    graph = nx.Graph()
    positions = {
        "A": (0.0, 0.0),
        "B": (0.0, 0.001),
        "C": (0.001, 0.001),
        "D": (0.0, 0.002),
    }
    graph.add_edge("A", "B", length=100.0)
    graph.add_edge("B", "D", length=100.0)
    graph.add_edge("A", "C", length=150.0)
    graph.add_edge("C", "D", length=150.0)
    return graph, "A", "D", positions.__getitem__


class RoutingTests(unittest.TestCase):
    def test_fast_cost_has_no_risk_penalty(self):
        self.assertEqual(_edge_cost(100, 0.8, 0), 100)

    def test_safe_cost_uses_normalized_segment_risk(self):
        self.assertEqual(_edge_cost(100, 0.8, 10), 900)

    def test_beta_can_change_route_and_reduce_accumulated_risk(self):
        with patch(
            "app.services.routing._build_osm_graph_base",
            return_value=build_alternative_graph(),
        ):
            result = generate_route_comparison(
                origin=(0.0, 0.0),
                destination=(0.0, 0.002),
                risk_model=FakeRiskModel(),
                beta=10,
            )
        self.assertGreater(result["risk_reduction"], 0)
        self.assertNotEqual(
            result["safe_route"]["route"],
            result["traditional_route"]["route"],
        )
        self.assertEqual(result["parametros_a_star"]["beta_ruta_rapida"], 0)
        self.assertGreater(result["parametros_a_star"]["beta_ruta_segura"], 0)

    def test_every_segment_has_variable_normalized_risk(self):
        with patch(
            "app.services.routing._build_osm_graph_base",
            return_value=build_alternative_graph(),
        ):
            result = generate_route_comparison(
                origin=(0.0, 0.0),
                destination=(0.0, 0.002),
                risk_model=FakeRiskModel(),
            )
        values = {
            segment["riesgo_segmento_normalizado"]
            for route in (result["safe_route"], result["traditional_route"])
            for segment in route["segments"]
        }
        self.assertTrue(all(0 <= value <= 1 for value in values))
        self.assertGreater(len(values), 1)

    def test_route_uses_requested_random_forest(self):
        with patch(
            "app.services.routing._build_osm_graph_base",
            return_value=build_alternative_graph(),
        ):
            result = generate_route_comparison(
                origin=(0.0, 0.0),
                destination=(0.0, 0.002),
                risk_model=FakeRiskModel(),
                modelo_riesgo="random_forest",
            )
        self.assertEqual(result["modelo_usado"], "Random Forest")
        self.assertEqual(result["periodo_prediccion"], "2025-07")

    def test_risk_mode_selects_the_expected_segment_score(self):
        results = {}
        with patch(
            "app.services.routing._build_osm_graph_base",
            return_value=build_alternative_graph(),
        ):
            for mode in ("predicted", "historical", "hybrid"):
                results[mode] = generate_route_comparison(
                    origin=(0.0, 0.0),
                    destination=(0.0, 0.002),
                    risk_model=FakeRiskModel(),
                    risk_mode=mode,
                )

        for mode, result in results.items():
            self.assertEqual(result["modo_riesgo"], mode)
            segment = result["traditional_route"]["segments"][0]
            if mode == "predicted":
                expected = segment["riesgo_predicho"]
            elif mode == "historical":
                expected = segment["riesgo_historico"]
            else:
                expected = (
                    0.7 * segment["riesgo_historico"]
                    + 0.3 * segment["riesgo_predicho"]
                )
            self.assertAlmostEqual(
                segment["riesgo_segmento_normalizado"], expected, places=5
            )

    def test_same_route_returns_explanatory_message(self):
        graph = nx.Graph()
        positions = {"A": (0.0, 0.0), "B": (0.0, 0.001)}
        graph.add_edge("A", "B", length=100.0)
        with patch(
            "app.services.routing._build_osm_graph_base",
            return_value=(graph, "A", "B", positions.__getitem__),
        ):
            result = generate_route_comparison(
                origin=(0.0, 0.0),
                destination=(0.0, 0.001),
                risk_model=FakeRiskModel(),
            )
        self.assertEqual(result["risk_reduction"], 0)
        self.assertIn("coincide con la ruta más rápida", result["mensaje"])


if __name__ == "__main__":
    unittest.main()
