from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np
import pandas as pd
from scipy import sparse

from app.flujo_entrenamiento_sidpol.config import TURNOS, TrainingConfig
from app.flujo_entrenamiento_sidpol.segmentacion import MONTHS


FEATURE_NAMES = (
    "latitud", "longitud", "longitud_m",
    "turno_sin", "turno_cos", "mes_sin", "mes_cos",
    "frecuencia_turno_3m", "gravedad_turno_3m",
    "frecuencia_total_3m", "gravedad_total_3m",
    "graves_turno_3m", "robos_turno_3m", "hurtos_turno_3m",
    "extorsiones_turno_3m", "homicidios_turno_3m",
    "densidad_turno_100m", "densidad_gravedad_turno_100m",
)


@dataclass
# Construye las variables de cada segmento y turno consultando exclusivamente periodos anteriores.
class FeatureBuilder:
    tramos: pd.DataFrame
    matrices: dict[str, sparse.csr_matrix]
    config: TrainingConfig

    def __post_init__(self) -> None:
        self.n_segments = len(self.tramos)
        self.coordinates = self.tramos[["latitud", "longitud", "longitud_m"]].to_numpy(dtype=np.float32)
        self.length_factor = np.maximum(self.coordinates[:, 2] / 100.0, 1.0)

    def _sum(self, name: str, month_indices: range, turns: tuple[int, ...]) -> np.ndarray:
        indices = [month * len(TURNOS) + turn for month in month_indices for turn in turns]
        return np.asarray(self.matrices[name][indices].sum(axis=0)).ravel().astype(np.float32)

    # Calcula la representación tabular histórica que reciben los modelos predictivos.
    def features(self, month_index: int, turn_index: int) -> np.ndarray:
        # Los primeros meses de la serie tienen historia parcial; nunca se usa
        # información del propio mes objetivo ni de meses posteriores.
        start = max(0, month_index - self.config.history_months)
        months = range(start, month_index)
        selected_turn = (turn_index,)
        all_turns = tuple(range(len(TURNOS)))
        count_turn = self._sum("count", months, selected_turn)
        weight_turn = self._sum("weight", months, selected_turn)
        count_all = self._sum("count", months, all_turns)
        weight_all = self._sum("weight", months, all_turns)
        month_number = int(MONTHS[month_index][5:7])
        x = np.empty((self.n_segments, len(FEATURE_NAMES)), dtype=np.float32)
        x[:, :3] = self.coordinates
        x[:, 3] = sin(2 * pi * turn_index / len(TURNOS))
        x[:, 4] = cos(2 * pi * turn_index / len(TURNOS))
        x[:, 5] = sin(2 * pi * (month_number - 1) / 12)
        x[:, 6] = cos(2 * pi * (month_number - 1) / 12)
        x[:, 7:11] = np.column_stack((count_turn, weight_turn, count_all, weight_all))
        for column, name in enumerate(("grave", "robo", "hurto", "extorsion", "homicidio"), start=11):
            x[:, column] = self._sum(name, months, selected_turn)
        x[:, 16] = count_turn / self.length_factor
        x[:, 17] = weight_turn / self.length_factor
        return x

    # Obtiene el riesgo observado del periodo objetivo para crear las etiquetas de evaluación.
    def target_risk(self, month_index: int, turn_index: int) -> np.ndarray:
        row = month_index * len(TURNOS) + turn_index
        weight = self.matrices["weight"].getrow(row).toarray().ravel().astype(np.float32)
        return weight / self.length_factor


# Selecciona los meses de cada partición considerando los cortes y la cobertura auditada.
def eligible_months(config: TrainingConfig, source_audit: dict) -> dict[str, list[int]]:
    """Admite un objetivo solo si el mes y toda su historia tienen cobertura suficiente."""
    coverage = {
        month: values["cobertura"]
        for month, values in source_audit["monthly"].items()
    }
    result = {"train": [], "validation": [], "test": []}
    bounds = {
        "train": (config.train_start, config.train_end),
        "validation": (config.validation_start, config.validation_end),
        "test": (config.test_start, config.test_end),
    }
    for i, month in enumerate(MONTHS):
        if i < config.history_months:
            continue
        history = MONTHS[i - config.history_months:i]
        if min(coverage.get(period, 0.0) for period in (*history, month)) < config.min_monthly_geo_coverage:
            continue
        for split, (first, last) in bounds.items():
            if first <= month <= last:
                result[split].append(i)
    for split, months in result.items():
        if not months:
            raise ValueError(f"No hay meses aptos para {split}; revise cobertura y cortes")
    return result
