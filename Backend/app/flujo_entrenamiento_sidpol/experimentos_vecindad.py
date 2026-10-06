"""Compara RF y XGBoost con historia de tramos cercanos, sin alterar el modelo activo.

Ejecutar desde Backend: python -m app.flujo_entrenamiento_sidpol.experimentos_vecindad
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import sparse
from scipy.spatial import cKDTree

from app.flujo_entrenamiento_sidpol.config import TrainingConfig
from app.flujo_entrenamiento_sidpol.modelado import (
    _collect_probabilities,
    _decision_thresholds,
    _evaluate,
    _fit,
    _models,
    _sample,
    _threshold,
    _write_metrics,
)
from app.flujo_entrenamiento_sidpol.segmentacion import cargar_matrices
from app.flujo_entrenamiento_sidpol.variables import FEATURE_NAMES, FeatureBuilder, eligible_months


NEIGHBOR_FEATURE_NAMES = (
    "vecinos_frecuencia_turno_3m",
    "vecinos_gravedad_turno_3m",
    "vecinos_frecuencia_total_3m",
    "vecinos_gravedad_total_3m",
)


class SpatialFeatureBuilder(FeatureBuilder):
    """Añade actividad previa de hasta 24 tramos a 300 m del centroide."""

    def __post_init__(self) -> None:
        super().__post_init__()
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)
        easting, northing = transformer.transform(
            self.coordinates[:, 1].astype(float), self.coordinates[:, 0].astype(float)
        )
        points = np.column_stack((easting, northing))
        distances, indices = cKDTree(points).query(
            points, k=25, distance_upper_bound=300.0, workers=-1
        )
        valid = np.isfinite(distances) & (indices < self.n_segments)
        valid &= indices != np.arange(self.n_segments)[:, None]
        rows = np.broadcast_to(np.arange(self.n_segments)[:, None], indices.shape)[valid]
        cols = indices[valid]
        weights = np.exp(-distances[valid] / 125.0).astype(np.float32)
        self.neighbors = sparse.csr_matrix(
            (weights, (rows, cols)), shape=(self.n_segments, self.n_segments)
        )
        print(f"Vecinos espaciales: {self.neighbors.nnz:,} enlaces", flush=True)

    def features(self, month_index: int, turn_index: int) -> np.ndarray:
        original = super().features(month_index, turn_index)
        historical = original[:, [7, 8, 9, 10]]
        nearby = self.neighbors @ historical
        return np.column_stack((original, nearby)).astype(np.float32, copy=False)


def _high_scores(evaluation: dict) -> dict[str, float]:
    matrix = np.asarray(evaluation["matriz_confusion"])
    tp = int(matrix[2, 2])
    actual = int(matrix[2].sum())
    predicted = int(matrix[:, 2].sum())
    precision = tp / predicted if predicted else 0.0
    recall = tp / actual if actual else 0.0
    return {
        "precision_alto": precision,
        "recall_alto": recall,
        "f1_alto": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def run() -> Path:
    config = TrainingConfig.load()
    source_audit = json.loads((config.output / "auditoria_fuente.json").read_text(encoding="utf-8"))
    splits = eligible_months(config, source_audit)
    tramos = pd.read_csv(config.output / "tramos_osm.csv")
    builder = SpatialFeatureBuilder(tramos, cargar_matrices(config.output), config)
    high_from = _threshold(builder, splits["train"])
    x_train, y_train = _sample(builder, splits["train"], high_from, config.random_state)
    output = config.output / "experimentos_vecindad"
    output.mkdir(exist_ok=True)
    baseline = json.loads((config.output / "resumen_entrenamiento.json").read_text(encoding="utf-8"))
    results = {}
    for name, model in _models(config).items():
        print(f"Entrenando {name} con vecindad", flush=True)
        _fit(model, x_train, y_train)
        truth, probability = _collect_probabilities(model, builder, splits["validation"], high_from)
        cutoffs = _decision_thresholds(truth, probability)
        validation = _evaluate(truth, probability, splits["validation"], cutoffs)
        validation.update(_high_scores(validation))
        del truth, probability
        print(f"Validación {name}: {validation['f1_alto']:.4f} F1 alto", flush=True)
        truth, probability = _collect_probabilities(model, builder, splits["test"], high_from)
        test = _evaluate(truth, probability, splits["test"], cutoffs)
        test.update(_high_scores(test))
        del truth, probability
        print(f"Prueba {name}: {test['f1_alto']:.4f} F1 alto", flush=True)
        _write_metrics(output, name, test)
        joblib.dump(model, output / f"modelo_entrenamiento_{name}.joblib", compress=3)
        results[name] = {
            "validation": validation,
            "test": test,
            "decision_thresholds": cutoffs,
        }
    summary = {
        "created_at": datetime.now().isoformat(),
        "description": "Historia de 24 tramos vecinos como máximo, radio 300 m, decaimiento 125 m",
        "feature_names": [*FEATURE_NAMES, *NEIGHBOR_FEATURE_NAMES],
        "high_threshold": high_from,
        "baseline": {
            name: {
                period: {
                    **baseline["assessments"][name][period],
                    **_high_scores(baseline["assessments"][name][period]),
                }
                for period in ("validation", "test")
            }
            for name in ("random_forest", "xgboost")
        },
        "results": results,
    }
    path = output / "comparacion.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(run(), flush=True)
