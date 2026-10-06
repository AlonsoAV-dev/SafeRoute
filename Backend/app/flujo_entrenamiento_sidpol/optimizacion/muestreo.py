from __future__ import annotations

import numpy as np


def select_training_rows(y, active_history, rng, negative_ratio=3, minimum_negatives=900, active_fraction=0.6):
    """Conserva todos los medios/altos; muestrea bajos en dos estratos históricos.

    Devuelve pesos inversos de inclusión para poder reconstruir la distribución
    natural o comparar ponderaciones intermedias sin perder la trazabilidad.
    """
    positive = np.flatnonzero(y > 0)
    negative_budget = min(int(np.count_nonzero(y == 0)), max(minimum_negatives, int(len(positive) * negative_ratio)))
    active = np.flatnonzero((y == 0) & active_history)
    inactive = np.flatnonzero((y == 0) & ~active_history)
    n_active = min(len(active), int(round(negative_budget * active_fraction)))
    n_inactive = min(len(inactive), negative_budget - n_active)
    n_active = min(len(active), negative_budget - n_inactive)
    indices, weights = [positive], [np.ones(len(positive), dtype=np.float32)]
    for population, amount in ((active, n_active), (inactive, n_inactive)):
        if amount:
            indices.append(rng.choice(population, size=amount, replace=False))
            weights.append(np.full(amount, len(population) / amount, dtype=np.float32))
    return np.concatenate(indices), np.concatenate(weights)
