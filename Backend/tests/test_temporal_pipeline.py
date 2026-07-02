from __future__ import annotations

import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from app.flujo_entrenamiento.riesgo import VARIABLES_MODELO, crear_panel_temporal


class TemporalPipelineTests(unittest.TestCase):
    def test_panel_uses_complete_months_and_predicts_next_one(self):
        fechas = pd.to_datetime(
            ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31", "2025-06-30", "2025-07-17"]
        )
        delitos = pd.DataFrame(
            {
                "id_hecho": [f"D{i}" for i in range(len(fechas))],
                "fecha": fechas,
                "tramo_id": ["T1"] * len(fechas),
                "asociado": [True] * len(fechas),
                "peso_delito": [1, 2, 3, 4, 1, 3, 5],
                "es_delito_grave": [0, 0, 0, 1, 0, 0, 1],
                "categoria_delito": ["hurtos", "hurtos", "robos", "robos", "hurtos", "robos", "homicidios"],
                "turno": ["noche"] * len(fechas),
                "latitud": [-12.0] * len(fechas),
                "longitud": [-77.0] * len(fechas),
            }
        )
        tramos = pd.DataFrame(
            {
                "tramo_id": ["T1", "T2"],
                "latitud": [-12.0, -12.01],
                "longitud": [-77.0, -77.01],
                "longitud_m": [100.0, 200.0],
            }
        )
        tramos_geo = gpd.GeoDataFrame(
            {
                "tramo_id": ["T1", "T2"],
                "geometry": [
                    LineString([(-77.001, -12.0), (-76.999, -12.0)]),
                    LineString([(-77.011, -12.01), (-77.009, -12.01)]),
                ],
            },
            crs="EPSG:4326",
        )

        panel, futuro, metadata = crear_panel_temporal(
            delitos,
            tramos,
            tramos_geo,
            ventana_meses=3,
            radio_m=100,
        )

        self.assertEqual(sorted(panel["periodo_objetivo"].unique()), ["2025-04", "2025-05", "2025-06"])
        self.assertEqual(metadata["periodos_excluidos"], ["2025-07"])
        self.assertEqual(metadata["periodo_prediccion"], "2025-07")
        self.assertEqual(metadata["radio_buffer_m"], 100)
        self.assertEqual(len(futuro), 2)
        self.assertEqual(len(VARIABLES_MODELO), 16)
        self.assertNotIn("frecuencia_ultimo_mes", panel.columns)
        self.assertNotIn("mes_objetivo_sin", panel.columns)
        self.assertEqual(
            futuro.loc[futuro["tramo_id"].eq("T1"), "frecuencia_delitos_hist"].iloc[0],
            3,
        )


if __name__ == "__main__":
    unittest.main()
