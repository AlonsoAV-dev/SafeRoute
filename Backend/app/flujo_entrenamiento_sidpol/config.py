from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[3]
BACKEND_DIR = PROJECT_DIR / "Backend"
DEFAULT_CONFIG = BACKEND_DIR / "config_entrenamiento_sidpol.json"
TURNOS = ("madrugada", "manana", "tarde", "noche")


@dataclass(frozen=True)
# Agrupa rutas, cortes temporales e hiperparámetros definidos por la configuración SIDPOL.
class TrainingConfig:
    source: Path
    graph: Path
    output: Path
    weights_file: Path
    match_radius_m: float
    min_monthly_geo_coverage: float
    history_months: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    forecast_period: str
    samples_per_class_per_month_turn: int
    random_state: int
    rf_trees: int
    xgb_trees: int

    @classmethod
    # Resuelve las rutas del JSON y verifica que los periodos sean cronológicos y no se solapen.
    def load(cls, path: Path = DEFAULT_CONFIG) -> "TrainingConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        base = path.resolve().parent
        for key in ("source", "graph", "output", "weights_file"):
            location = Path(raw[key])
            raw[key] = location if location.is_absolute() else (base / location).resolve()
        config = cls(**raw)
        if not (0 < config.min_monthly_geo_coverage <= 1):
            raise ValueError("min_monthly_geo_coverage debe estar entre 0 y 1")
        if config.history_months < 1 or config.match_radius_m <= 0:
            raise ValueError("Ventana histórica y radio deben ser positivos")
        for first, last in (
            (config.train_start, config.train_end),
            (config.validation_start, config.validation_end),
            (config.test_start, config.test_end),
        ):
            if first > last:
                raise ValueError(f"Rango temporal invertido: {first}..{last}")
        if not (
            config.train_end < config.validation_start
            and config.validation_end < config.test_start
            and config.test_end < config.forecast_period
        ):
            raise ValueError("Los cortes deben ser cronológicos y sin solaparse")
        return config
