from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.services.preprocessing import CrimeRecord
from app.services.risk_model import RiskModel


def record(lat, lng, weight, modality):
    return CrimeRecord(
        lat=lat,
        lng=lng,
        turno="noche",
        tipo="PATRIMONIO",
        subtipo="ROBO",
        modalidad=modality,
        peso_delito=weight,
        distrito="LIMA",
        fecha="2026-01-01",
        dia_semana="jueves",
    )


class RiskModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RiskModel(
            [
                record(-12.00, -77.00, 1, "HURTO"),
                record(-12.001, -77.001, 1, "HURTO"),
                record(-12.002, -77.002, 2, "ROBO FRUSTRADO"),
                record(-12.04, -77.04, 5, "SICARIATO"),
                record(-12.041, -77.041, 5, "FEMINICIDIO"),
                record(-12.042, -77.042, 5, "ROBO AGRAVADO"),
            ],
            grid_size_m=100,
        )

    def test_heatmap_values_are_normalized_and_not_uniform(self):
        values = [point[2] for point in self.model.get_heatmap_points()]
        self.assertTrue(values)
        self.assertTrue(all(0 <= value <= 1 for value in values))
        self.assertGreater(max(values), min(values))

    def test_route_risk_does_not_change_with_day_or_shift(self):
        morning = self.model.predict_point(-12.04, -77.04, "manana")
        night = self.model.predict_point(-12.04, -77.04, "noche")
        self.assertEqual(morning, night)

    def test_model_exposes_no_cluster_zones(self):
        self.assertEqual(self.model.get_segment_count(), 0)
        self.assertEqual(self.model.resolve_model("auto"), "Random Forest")

    def test_prediction_layers_expose_random_forest_segment_scores(self):
        with TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "predicciones_tramos.csv").write_text(
                "tramo_id,latitud,longitud,riesgo_score,nivel_riesgo\n"
                "OSM-1-2-10,-12.01,-77.01,0.82,alto\n"
                "OSM-2-3-11,-12.02,-77.02,0.45,medio\n",
                encoding="utf-8",
            )
            (model_dir / "metadata_modelo.json").write_text(
                '{"periodo_prediccion": "2026-01"}', encoding="utf-8"
            )
            model = RiskModel(self.model.records, model_dir=model_dir)

        points = model.get_prediction_points(min_score=0.34)
        self.assertEqual([point["risk_score"] for point in points], [0.82, 0.45])
        self.assertEqual(model.prediction_period, "2026-01")
        self.assertTrue(model.get_prediction_heatmap_points())


if __name__ == "__main__":
    unittest.main()
