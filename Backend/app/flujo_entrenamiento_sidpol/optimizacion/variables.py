from __future__ import annotations

import numpy as np
from scipy import sparse

from app.flujo_entrenamiento_sidpol.segmentacion import MONTHS
from app.flujo_entrenamiento_sidpol.variables import FEATURE_NAMES, FeatureBuilder


WINDOW_FIELDS = (
    "frecuencia_turno", "gravedad_turno", "frecuencia_total", "gravedad_total",
    "graves_turno", "robos_turno", "hurtos_turno", "extorsiones_turno", "homicidios_turno",
    "meses_activos_turno", "meses_activos_total", "densidad_turno", "densidad_total",
    "fraccion_meses_observados", "cobertura_media",
)
LONG_FIELDS = (
    "historico_frecuencia_turno", "historico_frecuencia_total", "historico_gravedad_turno",
    "historico_fraccion_medio", "historico_fraccion_alto", "historico_fraccion_activo",
    "historico_cuota_turno", "meses_desde_ultimo_turno", "meses_desde_ultimo_total",
    "meses_observados_historico",
)


class WindowFeatures:
    """Todo predictor usa meses anteriores al destino; ignora historia de baja cobertura.

    Las frecuencias se expresan por mes observado. La fracción observada permanece
    como predictor para distinguir historia incompleta de actividad nula.
    """

    def __init__(self, tramos, matrices, source_config, audit, windows=(1, 3, 6, 12), minimum_coverage=0.8, high_threshold=3.0):
        self.legacy = FeatureBuilder(tramos, matrices, source_config)
        self.tramos, self.matrices = tramos, matrices
        self.n_segments = len(tramos)
        self.factor = self.legacy.length_factor
        self.windows = tuple(windows)
        self.coverage = np.array([audit.get("monthly", {}).get(m, {}).get("cobertura", 0) for m in MONTHS])
        self.observed = self.coverage >= minimum_coverage
        active = matrices["count"].copy()
        active.data[:] = 1
        self.active = active
        high = matrices["weight"].copy()
        high.data = (high.data / self.factor[high.indices] >= high_threshold).astype(np.float32)
        high.eliminate_zeros()
        self.high = high
        self.names = list(FEATURE_NAMES)
        self.groups = {"legacy": list(range(len(FEATURE_NAMES)))}
        for w in self.windows:
            start = len(self.names)
            self.names += [f"ventana_{w}m_{name}" for name in WINDOW_FIELDS]
            self.groups[f"{w}m"] = list(range(7)) + list(range(start, len(self.names)))
        self.names += list(LONG_FIELDS)
        self.groups["multi"] = list(range(7)) + list(range(len(FEATURE_NAMES), len(self.names)))

    def months(self, target, window=None):
        start = 0 if window is None else max(0, target - window)
        return [m for m in range(start, target) if self.observed[m]]

    def sum_rows(self, matrix, months, turn=None):
        rows = [m * 4 + t for m in months for t in (range(4) if turn is None else (turn,))]
        if not rows:
            return np.zeros(self.n_segments, dtype=np.float32)
        return np.asarray(matrix[rows].sum(axis=0)).ravel().astype(np.float32)

    def target(self, month, turn, high_threshold=3):
        risk = self.legacy.target_risk(month, turn)
        return np.where(risk >= high_threshold, 2, np.where(risk > 0, 1, 0)).astype(np.int8)

    def block(self, month, turn):
        columns = [self.legacy.features(month, turn)]
        for w in self.windows:
            months = self.months(month, w)
            denominator = max(len(months), 1)
            counts = [
                self.sum_rows(self.matrices[name], months, t) / denominator
                for name, t in (("count", turn), ("weight", turn), ("count", None), ("weight", None),
                                ("grave", turn), ("robo", turn), ("hurto", turn), ("extorsion", turn), ("homicidio", turn))
            ]
            active_turn = self.sum_rows(self.active, months, turn) / denominator
            active_all = np.zeros(self.n_segments, dtype=np.float32)
            for m in months:
                active_all += self.sum_rows(self.matrices["count"], [m]) > 0
            active_all /= denominator
            constants = np.empty((self.n_segments, 2), dtype=np.float32)
            constants[:, 0] = len(months) / w
            constants[:, 1] = float(self.coverage[months].mean()) if months else 0
            columns.append(np.column_stack((*counts, active_turn, active_all, counts[0] / self.factor, counts[2] / self.factor, constants)))
        months = self.months(month)
        denominator = max(len(months), 1)
        count_turn = self.sum_rows(self.matrices["count"], months, turn)
        count_all = self.sum_rows(self.matrices["count"], months)
        active_turn = self.sum_rows(self.active, months, turn)
        high_turn = self.sum_rows(self.high, months, turn)
        recency_turn = np.full(self.n_segments, month + 1, dtype=np.float32)
        recency_all = recency_turn.copy()
        for m in months:
            recency_turn[self.matrices["count"].getrow(m * 4 + turn).indices] = month - m
            for t in range(4):
                recency_all[self.matrices["count"].getrow(m * 4 + t).indices] = month - m
        columns.append(np.column_stack((
            count_turn / denominator, count_all / denominator,
            self.sum_rows(self.matrices["weight"], months, turn) / denominator,
            (active_turn - high_turn) / denominator, high_turn / denominator, active_turn / denominator,
            count_turn / np.maximum(count_all, 1), recency_turn, recency_all,
            np.full(self.n_segments, len(months), dtype=np.float32),
        )))
        return np.column_stack(columns).astype(np.float32, copy=False)
