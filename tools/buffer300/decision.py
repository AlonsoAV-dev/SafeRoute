"""Decisión por tramo y mes con clase alto protegida."""

import numpy as np


def predict_frozen_high(base_probability, light_probability, blend_lighting, medium_multiplier):
    """Conserva alto del modelo base y decide bajo/medio en el resto."""
    base = np.asarray(base_probability)
    light = np.asarray(light_probability)
    if base.shape != light.shape or base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("Las probabilidades deben tener forma (tramos, 3) en ambos modelos")
    if not 0 <= blend_lighting <= 1 or medium_multiplier <= 0:
        raise ValueError("Parámetros de decisión fuera de rango")
    original = base.argmax(axis=1)
    other = original != 2
    result = original.astype(np.int8, copy=True)
    low = (1-blend_lighting)*base[other, 0] + blend_lighting*light[other, 0]
    medium = (1-blend_lighting)*base[other, 1] + blend_lighting*light[other, 1]
    result[other] = (medium*medium_multiplier > low).astype(np.int8)
    return result
