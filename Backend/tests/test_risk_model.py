from __future__ import annotations

import unittest

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
            clusters=2,
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


if __name__ == "__main__":
    unittest.main()
