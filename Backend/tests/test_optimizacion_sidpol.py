from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from app.flujo_entrenamiento_sidpol.optimizacion.evaluacion import metrics, tune_joint_thresholds
from app.flujo_entrenamiento_sidpol.optimizacion.muestreo import select_training_rows
from app.flujo_entrenamiento_sidpol.optimizacion.variables import WindowFeatures
from app.flujo_entrenamiento_sidpol.segmentacion import MONTHS


class WindowFeatureTests(unittest.TestCase):
    def builder(self, missing_month=None, future_value=0):
        values = np.zeros((len(MONTHS) * 4, 3), dtype=np.float32)
        values[0, 0] = 2
        values[4, 0] = 4
        values[8, 1] = 1
        values[12, 2] = future_value
        values[16, 2] = future_value
        matrices = {name: sparse.csr_matrix(values) for name in ("count", "weight", "grave", "robo", "hurto", "extorsion", "homicidio")}
        tramos = pd.DataFrame({"latitud": [-12., -12.1, -12.2], "longitud": [-77., -77.1, -77.2], "longitud_m": [100., 100., 100.]})
        audit = {"monthly": {m: {"cobertura": 0.2 if i == missing_month else 1.} for i, m in enumerate(MONTHS)}}
        return WindowFeatures(tramos, matrices, SimpleNamespace(history_months=3), audit)

    def test_current_and_future_events_do_not_change_predictors(self):
        before = self.builder(future_value=0)
        after = self.builder(future_value=10000)
        np.testing.assert_array_equal(before.block(3, 0), after.block(3, 0))
        self.assertNotEqual(before.target(3, 0)[2], after.target(3, 0)[2])

    def test_missing_history_is_excluded_and_reported(self):
        complete = self.builder()
        missing = self.builder(missing_month=1)
        count = complete.groups["3m"][7]
        observed = complete.names.index("ventana_3m_fraccion_meses_observados")
        self.assertEqual(complete.block(3, 0)[0, count], 2)
        self.assertEqual(missing.block(3, 0)[0, count], 1)
        self.assertAlmostEqual(float(missing.block(3, 0)[0, observed]), 2 / 3, places=6)


class SamplingAndDecisionTests(unittest.TestCase):
    def test_all_medium_high_examples_survive_and_weights_recover_population(self):
        y = np.r_[np.zeros(1000, dtype=np.int8), np.ones(20, dtype=np.int8), np.full(10, 2, dtype=np.int8)]
        history = np.arange(len(y)) % 3 == 0
        indices, weights = select_training_rows(y, history, np.random.default_rng(4), negative_ratio=2, minimum_negatives=1)
        self.assertEqual(len(indices), len(np.unique(indices)))
        self.assertEqual(set(indices[y[indices] > 0]), set(np.flatnonzero(y > 0)))
        self.assertAlmostEqual(float(weights.sum()), len(y), places=2)

    def test_joint_thresholds_find_separable_medium_and_high(self):
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int8)
        p = np.array([[.9, .05, .05], [.7, .2, .1], [.6, .3, .1],
                      [.3, .6, .1], [.2, .6, .2], [.2, .7, .1],
                      [.1, .2, .7], [.05, .15, .8], [.1, .1, .8]])
        cutoffs = tune_joint_thresholds(y, p)
        result = metrics(y, p, cutoffs)
        self.assertEqual(result["f1_medio_alto"], 1.0)
        self.assertEqual(result["matriz_confusion"], [[3, 0, 0], [0, 3, 0], [0, 0, 3]])


if __name__ == "__main__":
    unittest.main()
