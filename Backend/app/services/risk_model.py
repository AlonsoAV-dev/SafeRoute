from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from math import atan2, cos, floor, radians, sin, sqrt
from pathlib import Path

import numpy as np
from sklearn.neighbors import BallTree

from app.services.preprocessing import CrimeRecord, normalize_turno

MODEL_ALIASES = {
    "auto": "random_forest",
    "random_forest": "random_forest",
    "rf": "random_forest",
    "xgboost": "xgboost",
    "xgb": "xgboost",
}
MODEL_NAMES = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
DEFAULT_MODEL_KEY = "random_forest"


@dataclass(frozen=True)
class RiskPrediction:
    score: float
    level: str


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6_371_000
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return 2 * radius * atan2(sqrt(a), sqrt(1 - a))


def level_from_score(score: float) -> str:
    if score >= 0.66:
        return "alto"
    if score >= 0.34:
        return "medio"
    return "bajo"


class RiskModel:
    """Consulta predicciones supervisadas por segmento y capas historicas."""

    def __init__(
        self,
        records: list[CrimeRecord],
        grid_size_m: int = 100,
        model_dir: Path | None = None,
    ):
        if not records:
            raise ValueError("Se necesita al menos un registro delictivo válido.")
        self.records = records
        self.grid_size_m = max(40, int(grid_size_m))
        self.model_name = "Random Forest"
        self.model_accuracy = 0.0
        self.model_metrics: dict[str, dict] = {}
        self.prediction_period = "no disponible"
        self.model_version = "no disponible"
        self.feature_count = 0
        self._segment_scores: dict[str, RiskPrediction] = {}
        self._prediction_rows: list[RiskPrediction] = []
        self._prediction_points: list[dict] = []
        self._prediction_heatmap: list[list[float]] = []
        self._prediction_tree: BallTree | None = None
        self._model_keys: set[str] = set()
        self._segment_scores_by_model: dict[str, dict[str, RiskPrediction]] = {}
        self._prediction_rows_by_model: dict[str, list[RiskPrediction]] = {}
        self._prediction_points_by_model: dict[str, list[dict]] = {}
        self._prediction_heatmap_by_model: dict[str, list[list[float]]] = {}
        self._prediction_tree_by_model: dict[str, BallTree] = {}
        self._lat0 = sum(record.lat for record in records) / len(records)
        self._lat_m_per_deg = 111_320.0
        self._lng_m_per_deg = self._lat_m_per_deg * cos(radians(self._lat0))
        coordinates = np.asarray([[record.lat, record.lng] for record in records])
        self._crime_tree = BallTree(np.radians(coordinates), metric="haversine")
        if model_dir:
            self._load_predictions(model_dir)

    def resolve_model(self, requested: str | None = None) -> str:
        key = self._resolve_model_key(requested)
        return MODEL_NAMES.get(key, self.model_name)

    def metrics_for_model(self, requested: str | None = None) -> dict:
        key = self._resolve_model_key(requested)
        return self.model_metrics.get(MODEL_NAMES.get(key, self.model_name), {})

    def predict_segment(
        self,
        tramo_id: str,
        midpoint: tuple[float, float],
        modelo_riesgo: str | None = None,
    ) -> RiskPrediction:
        key = self._resolve_model_key(modelo_riesgo)
        exacta = self._segment_scores_by_model.get(key, {}).get(tramo_id)
        if exacta is not None:
            return exacta
        return self.predict_point(midpoint[0], midpoint[1], modelo_riesgo=modelo_riesgo)

    def predict_point(
        self,
        lat: float,
        lng: float,
        turno: str | None = None,
        modelo_riesgo: str | None = None,
    ) -> RiskPrediction:
        del turno
        key = self._resolve_model_key(modelo_riesgo)
        prediction_tree = self._prediction_tree_by_model.get(key)
        prediction_rows = self._prediction_rows_by_model.get(key, [])
        if prediction_tree is not None:
            distances, indices = prediction_tree.query(
                np.radians(np.asarray([[lat, lng]])), k=1
            )
            distance_m = float(distances[0][0]) * 6_371_000
            if distance_m <= 1_000:
                return prediction_rows[int(indices[0][0])]
        stats = self.nearby_crime_stats([(lat, lng)], radius_m=100)
        score = min(1.0, stats["weight_sum"] / 10.0)
        return RiskPrediction(round(score, 6), level_from_score(score))

    def nearby_crime_stats(
        self,
        sample_points: list[tuple[float, float]],
        radius_m: float,
    ) -> dict:
        if not sample_points:
            return {"count": 0, "weight_sum": 0.0, "weight_avg": 0.0}
        indices: set[int] = set()
        consultas = self._crime_tree.query_radius(
            np.radians(np.asarray(sample_points)),
            r=radius_m / 6_371_000,
        )
        for resultado in consultas:
            indices.update(int(index) for index in resultado)
        pesos = [self.records[index].peso_delito for index in indices]
        return {
            "count": len(pesos),
            "weight_sum": float(sum(pesos)),
            "weight_avg": float(np.mean(pesos)) if pesos else 0.0,
        }

    def get_heatmap_points(
        self,
        turno: str | None = None,
        tipo: str | None = None,
        modalidad: str | None = None,
        dia_semana: str | None = None,
    ) -> list[list[float]]:
        records = self._filter_records(turno, tipo, modalidad, dia_semana)
        cells = self._grid_cells(records)
        if not cells:
            return []
        raw = np.asarray([cell["weight_sum"] for cell in cells], dtype=float)
        p90 = max(float(np.percentile(raw, 90)), 1e-9)
        values = np.clip(raw / p90, 0, 1)
        return [
            [round(cell["center"][0], 6), round(cell["center"][1], 6), round(float(value), 6)]
            for cell, value in zip(cells, values)
        ]

    def get_crime_points(
        self,
        turno: str | None = None,
        tipo: str | None = None,
        modalidad: str | None = None,
        dia_semana: str | None = None,
    ) -> list[dict]:
        records = self._filter_records(turno, tipo, modalidad, dia_semana)
        return [
            {
                "id": index,
                "lat": record.lat,
                "lng": record.lng,
                "turno": record.turno,
                "tipo": record.tipo,
                "subtipo": record.subtipo,
                "modalidad": record.modalidad,
                "peso_delito": record.peso_delito,
                "distrito": record.distrito,
                "dia_semana": record.dia_semana,
            }
            for index, record in enumerate(records)
        ]

    def get_prediction_heatmap_points(
        self,
        modelo_riesgo: str | None = None,
    ) -> list[list[float]]:
        key = self._resolve_model_key(modelo_riesgo)
        return self._prediction_heatmap_by_model.get(key, [])

    def get_prediction_points(
        self,
        min_score: float = 0.34,
        limit: int = 15_000,
        modelo_riesgo: str | None = None,
    ) -> list[dict]:
        key = self._resolve_model_key(modelo_riesgo)
        minimum = min(1.0, max(0.0, float(min_score)))
        maximum = min(25_000, max(1, int(limit)))
        points = []
        for point in self._prediction_points_by_model.get(key, []):
            if point["risk_score"] < minimum:
                break
            points.append(point)
            if len(points) >= maximum:
                break
        return points

    def get_filter_options(self) -> dict:
        return {
            "turnos": ["todos", "manana", "tarde", "noche", "madrugada"],
            "dias_semana": ["todos", *sorted({r.dia_semana for r in self.records})],
            "tipos": ["todos", *sorted({r.tipo for r in self.records})],
            "modalidades": ["todos", *sorted({r.modalidad for r in self.records})],
        }

    def get_segment_count(self) -> int:
        return len(self._segment_scores)

    def _resolve_model_key(self, requested: str | None = None) -> str:
        normalized = MODEL_ALIASES.get((requested or "auto").lower(), DEFAULT_MODEL_KEY)
        if normalized in self._model_keys:
            return normalized
        if DEFAULT_MODEL_KEY in self._model_keys:
            return DEFAULT_MODEL_KEY
        return next(iter(self._model_keys), DEFAULT_MODEL_KEY)

    def _filter_records(
        self,
        turno: str | None,
        tipo: str | None,
        modalidad: str | None,
        dia_semana: str | None,
    ) -> list[CrimeRecord]:
        turno_normalizado = normalize_turno(turno) if turno and turno != "todos" else None
        return [
            record
            for record in self.records
            if (not turno_normalizado or record.turno == turno_normalizado)
            and (not tipo or tipo == "todos" or record.tipo == tipo)
            and (not modalidad or modalidad == "todos" or record.modalidad == modalidad)
            and (
                not dia_semana
                or dia_semana == "todos"
                or record.dia_semana == dia_semana
            )
        ]

    def _grid_cells(self, records: list[CrimeRecord]) -> list[dict]:
        cells: dict[tuple[int, int], dict] = defaultdict(dict)
        for record in records:
            x = floor(record.lng * self._lng_m_per_deg / self.grid_size_m)
            y = floor(record.lat * self._lat_m_per_deg / self.grid_size_m)
            key = (x, y)
            if not cells[key]:
                cells[key] = {
                    "center": (
                        (y + 0.5) * self.grid_size_m / self._lat_m_per_deg,
                        (x + 0.5) * self.grid_size_m / self._lng_m_per_deg,
                    ),
                    "total": 0,
                    "weight_sum": 0,
                }
            cells[key]["total"] += 1
            cells[key]["weight_sum"] += record.peso_delito
        return list(cells.values())

    def _load_predictions(self, model_dir: Path) -> None:
        configs = [
            (
                DEFAULT_MODEL_KEY,
                MODEL_NAMES[DEFAULT_MODEL_KEY],
                model_dir / "predicciones_tramos_random_forest.csv",
                model_dir / "metricas_random_forest.csv",
                model_dir / "metadata_modelo.json",
            ),
            (
                "xgboost",
                MODEL_NAMES["xgboost"],
                model_dir / "predicciones_tramos_xgboost.csv",
                model_dir / "metricas_xgboost.csv",
                model_dir / "metadata_modelo_xgboost.json",
            ),
        ]
        for key, name, predictions_path, metrics_path, metadata_path in configs:
            if key == DEFAULT_MODEL_KEY and not predictions_path.exists():
                predictions_path = model_dir / "predicciones_tramos.csv"
            self._load_model_predictions(
                key,
                name,
                predictions_path,
                metrics_path,
                metadata_path,
            )

    def _load_model_predictions(
        self,
        key: str,
        name: str,
        predictions_path: Path,
        metrics_path: Path,
        metadata_path: Path,
    ) -> None:
        if not predictions_path.exists():
            return
        coordinates = []
        segment_scores: dict[str, RiskPrediction] = {}
        prediction_rows: list[RiskPrediction] = []
        prediction_points: list[dict] = []
        with predictions_path.open("r", encoding="utf-8-sig", newline="") as archivo:
            for row in csv.DictReader(archivo):
                try:
                    score = min(1.0, max(0.0, float(row["riesgo_score"])))
                    prediction = RiskPrediction(
                        score=score,
                        level=row.get("nivel_riesgo") or level_from_score(score),
                    )
                    lat = float(row["latitud"])
                    lng = float(row["longitud"])
                    coordinates.append([radians(lat), radians(lng)])
                except (KeyError, TypeError, ValueError):
                    continue
                segment_scores[str(row["tramo_id"])] = prediction
                prediction_rows.append(prediction)
                prediction_points.append(
                    {
                        "tramo_id": str(row["tramo_id"]),
                        "lat": round(lat, 6),
                        "lng": round(lng, 6),
                        "risk_score": round(score, 6),
                        "risk_level": prediction.level,
                    }
                )
        if coordinates:
            self._model_keys.add(key)
            prediction_tree = BallTree(np.asarray(coordinates), metric="haversine")
            prediction_points.sort(
                key=lambda point: point["risk_score"], reverse=True
            )
            self._segment_scores_by_model[key] = segment_scores
            self._prediction_rows_by_model[key] = prediction_rows
            self._prediction_points_by_model[key] = prediction_points
            self._prediction_tree_by_model[key] = prediction_tree
            self._prediction_heatmap_by_model[key] = self._build_prediction_heatmap(key)
            if key == DEFAULT_MODEL_KEY:
                self._segment_scores = segment_scores
                self._prediction_rows = prediction_rows
                self._prediction_points = prediction_points
                self._prediction_tree = prediction_tree
                self._prediction_heatmap = self._prediction_heatmap_by_model[key]
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8-sig", newline="") as archivo:
                row = next(csv.DictReader(archivo), None)
            if row:
                metricas = {
                    key: float(value)
                    for key, value in row.items()
                    if key not in {"modelo", "periodo_prueba"} and value not in {None, ""}
                }
                metricas["periodo_prueba"] = row.get("periodo_prueba", "")
                self.model_metrics[name] = metricas
                if key == DEFAULT_MODEL_KEY:
                    self.model_accuracy = float(metricas.get("accuracy", 0.0))
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as archivo:
                metadata = json.load(archivo)
            if key == DEFAULT_MODEL_KEY:
                self.prediction_period = str(metadata.get("periodo_prediccion", "no disponible"))
                self.model_version = str(metadata.get("version_variables", "no disponible"))
                self.feature_count = len(metadata.get("variables", []))

    def _build_prediction_heatmap(
        self,
        modelo_riesgo: str | None = None,
        cell_size_m: int = 800,
        min_score: float = 0.66,
    ) -> list[list[float]]:
        key = self._resolve_model_key(modelo_riesgo)
        cells: dict[tuple[int, int], dict] = {}
        for point in self._prediction_points_by_model.get(key, []):
            if point["risk_score"] < min_score:
                continue
            x = floor(point["lng"] * self._lng_m_per_deg / cell_size_m)
            y = floor(point["lat"] * self._lat_m_per_deg / cell_size_m)
            key = (x, y)
            cell = cells.setdefault(
                key,
                {
                    "lat": (y + 0.5) * cell_size_m / self._lat_m_per_deg,
                    "lng": (x + 0.5) * cell_size_m / self._lng_m_per_deg,
                    "score_sum": 0.0,
                    "score_max": 0.0,
                    "count": 0,
                },
            )
            cell["score_sum"] += point["risk_score"]
            cell["score_max"] = max(cell["score_max"], point["risk_score"])
            cell["count"] += 1

        if not cells:
            return []

        cell_values = list(cells.values())
        raw_values = np.asarray(
            [
                (
                    0.65 * (cell["score_sum"] / cell["count"])
                    + 0.35 * cell["score_max"]
                )
                * np.log1p(cell["count"])
                for cell in cell_values
            ]
        )
        concentration_cutoff = float(np.percentile(raw_values, 60))
        selected = raw_values >= concentration_cutoff
        p95 = max(float(np.percentile(raw_values, 95)), 1e-9)
        intensities = np.clip(raw_values / p95, 0.08, 1.0)
        return [
            [
                round(cell["lat"], 6),
                round(cell["lng"], 6),
                round(float(intensity), 6),
            ]
            for cell, intensity, include in zip(cell_values, intensities, selected)
            if include
        ]
