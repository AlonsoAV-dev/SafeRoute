from __future__ import annotations

import unittest

import pandas as pd

from app.flujo_entrenamiento.limpieza import limpiar_delitos


class LimpiezaTests(unittest.TestCase):
    def test_normalizes_english_month_to_spanish(self):
        datos = pd.DataFrame(
            {
                "GlobalID": ["A", "B"],
                "lat_hecho": [-12.04, -12.05],
                "long_hecho": [-77.04, -77.05],
                "fecha_hora_hecho": ["October 31, 2025", "diciembre 1, 2025"],
                "turno_hecho": ["noche", "mañana"],
                "tipo_hecho": ["PATRIMONIO", "PATRIMONIO"],
                "subtipo_hecho": ["ROBO", "HURTO"],
                "modalidad_hecho": ["ROBO", "HURTO"],
            }
        )

        resultado = limpiar_delitos(datos)

        self.assertEqual(resultado["periodo"].tolist(), ["2025-10", "2025-12"])
        self.assertEqual(resultado["mes_nombre"].tolist(), ["octubre", "diciembre"])


if __name__ == "__main__":
    unittest.main()
