from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from math import atan2, cos, floor, radians, sin, sqrt
from pathlib import Path
from types import MappingProxyType

import numpy as np
from sklearn.neighbors import BallTree

from app.services.preprocessing import CrimeRecord, normalize_turno

# Relaciona los identificadores recibidos desde la API con los tres modelos del sistema.
MODEL_ALIASES = {
    "auto": "random_forest",
    "random_forest": "random_forest",
    "rf": "random_forest",
    "xgboost": "xgboost",
    "xgb": "xgboost",
    "lstm": "lstm",
}
MODEL_NAMES = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}
DEFAULT_MODEL_KEY = "random_forest"


@dataclass(frozen=True, slots=True)
class RiskPrediction:
    score: float
    level: str


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6_371_000
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return 2 * radius * atan2(sqrt(a), sqrt(1 - a))


# Convierte el índice continuo a niveles usando los cortes 0,34 y 0,66.
def level_from_score(score: float) -> str:
    if score >= 0.66:
        return "alto"
    if score >= 0.34:
        return "medio"
    return "bajo"


# Centraliza las predicciones exportadas y los índices espaciales para consultar el riesgo.
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
        self._model_metadata: dict[str, dict] = {}
        self._segment_scores_by_model: dict[str, dict[str, RiskPrediction]] = {}
        self._segment_scores_by_model_turno: dict[str, dict[str, dict[str, RiskPrediction]]] = {}
        self._prediction_rows_by_model: dict[str, list[RiskPrediction]] = {}
        self._prediction_rows_by_model_turno: dict[str, dict[str, list[RiskPrediction]]] = {}
        self._prediction_points_by_model: dict[str, list[dict]] = {}
        self._prediction_heatmap_by_model: dict[str, list[list[float]]] = {}
        self._prediction_tree_by_model: dict[str, BallTree] = {}
        self.default_model_key = DEFAULT_MODEL_KEY
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

    def segment_predictions(self, requested: str | None = None, turno: str | None = None):
        """Predicciones originales, compartidas por la visualización y el ruteo."""
        key = self._resolve_model_key(requested)
        selected = normalize_turno(turno) if turno else None
        predictions = self._segment_scores_by_model_turno.get(key, {}).get(selected)
        return MappingProxyType(predictions if predictions is not None else self._segment_scores_by_model[key])

    def spatial_risk_surface(self):
        """Campo espacial único para el mapa y el costo de ruteo."""
        if not hasattr(self, "_spatial_risk_surface"):
            # Importación diferida: la superficie utiliza la misma red vial que el ruteo.
            from app.services.risk_surface import RiskSurface
            self._spatial_risk_surface = RiskSurface(self)
        return self._spatial_risk_surface

    def model_info(self, requested: str | None = None) -> dict:
        key = self._resolve_model_key(requested)
        metadata = self._model_metadata.get(key, {})
        return {
            "key": key,
            "name": MODEL_NAMES[key],
            "available": key in self._model_keys,
            "prediction_period": metadata.get("periodo_prediccion", "no disponible"),
            "supports_turns": bool(metadata.get("tramo_turno", False)),
            "unit": "segmento × mes × turno" if metadata.get("tramo_turno") else "segmento × mes",
            "training_end": metadata.get("entrenamiento_hasta"),
        }

    def available_models(self) -> list[dict]:
        return [
            self.model_info(key) if key in self._model_keys else {
                "key": key, "name": name, "available": False,
                "prediction_period": "no disponible", "supports_turns": False,
            }
            for key, name in MODEL_NAMES.items()
        ]

    # Busca la predicción del segmento y utiliza su ubicación cuando no encuentra una coincidencia exacta.
    def predict_segment(
        self,
        tramo_id: str,
        midpoint: tuple[float, float],
        modelo_riesgo: str | None = None,
        turno: str | None = None,
    ) -> RiskPrediction:
        key = self._resolve_model_key(modelo_riesgo)
        selected_turn = normalize_turno(turno) if turno else None
        if selected_turn and selected_turn in self._segment_scores_by_model_turno.get(key, {}):
            by_turn = self._segment_scores_by_model_turno[key][selected_turn]
            exacta = by_turn.get(tramo_id)
            if exacta is not None:
                return exacta
        exacta = self._segment_scores_by_model.get(key, {}).get(tramo_id)
        if exacta is not None:
            return exacta
        raise ValueError(f"No existe una predicción de {MODEL_NAMES[key]} para el tramo {tramo_id} de este recorrido.")

    def predict_point(
        self,
        lat: float,
        lng: float,
        turno: str | None = None,
        modelo_riesgo: str | None = None,
    ) -> RiskPrediction:
        key = self._resolve_model_key(modelo_riesgo)
        prediction_tree = self._prediction_tree_by_model.get(key)
        selected_turn = normalize_turno(turno) if turno else None
        prediction_rows = self._prediction_rows_by_model_turno.get(key, {}).get(
            selected_turn, self._prediction_rows_by_model.get(key, [])
        )
        if prediction_tree is not None:
            distances, indices = prediction_tree.query(
                np.radians(np.asarray([[lat, lng]])), k=1
            )
            distance_m = float(distances[0][0]) * 6_371_000
            if distance_m <= 1_000:
                return prediction_rows[int(indices[0][0])]
        raise ValueError("No hay una predicción cercana del modelo seleccionado para un tramo de este recorrido.")

    # Consulta los delitos próximos mediante el índice espacial de los registros históricos.
    def nearby_crime_stats(
        self,
        sample_points: list[tuple[float, float]],
        radius_m: float,
        turno: str | None = None,
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
        selected_turn = normalize_turno(turno) if turno else None
        pesos = [
            self.records[index].peso_delito for index in indices
            if not selected_turn or self.records[index].turno == selected_turn
        ]
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
        limit: int = 20_000,
    ) -> list[dict]:
        records = self._filter_records(turno, tipo, modalidad, dia_semana)
        maximum = max(1, min(int(limit), 50_000))
        if len(records) > maximum:
            stride = max(1, len(records) // maximum)
            records = records[::stride][:maximum]
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
        min_score: float = 0.0,
        limit: int = 15_000,
        modelo_riesgo: str | None = None,
        balanced: bool = False,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        key = self._resolve_model_key(modelo_riesgo)
        minimum = min(1.0, max(0.0, float(min_score)))
        maximum = min(25_000, max(1, int(limit)))
        source = self._prediction_points_by_model.get(key, [])
        if bounds:
            south, west, north, east = bounds
            source = [point for point in source if south <= point["lat"] <= north and west <= point["lng"] <= east]
        if balanced:
            levels = ("bajo", "medio", "alto")
            quota = max(1, maximum // len(levels))
            buckets = {level: [] for level in levels}
            for point in source:
                level = point["risk_level"]
                if point["risk_score"] >= minimum and level in buckets:
                    buckets[level].append(point)
            if sum(len(candidates) for candidates in buckets.values()) <= maximum:
                return [point for level in levels for point in buckets[level]]
            selected = {}
            for level, candidates in buckets.items():
                if len(candidates) <= quota:
                    selected[level] = candidates
                    continue
                indices = np.linspace(0, len(candidates) - 1, quota, dtype=int)
                selected[level] = [candidates[index] for index in indices]
            return [point for level in levels for point in selected[level]][:maximum]

        points = []
        for point in source:
            if point["risk_score"] < minimum:
                break
            points.append(point)
            if len(points) >= maximum:
                break
        return points

    def get_prediction_counts(
        self,
        modelo_riesgo: str | None = None,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> dict[str, int]:
        key = self._resolve_model_key(modelo_riesgo)
        counts = {"bajo": 0, "medio": 0, "alto": 0}
        for point in self._prediction_points_by_model.get(key, []):
            if bounds:
                south, west, north, east = bounds
                if not (south <= point["lat"] <= north and west <= point["lng"] <= east):
                    continue
            level = point["risk_level"]
            if level in counts:
                counts[level] += 1
        return counts

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
        request = (requested or "auto").lower()
        if request == "auto":
            return self.default_model_key
        normalized = MODEL_ALIASES.get(request)
        if normalized is None:
            raise ValueError(f"Modelo de riesgo no soportado: {requested}")
        if normalized not in self._model_keys:
            raise ValueError(f"No hay predicciones disponibles para {MODEL_NAMES[normalized]}.")
        return normalized

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

    # Carga los artefactos de los modelos disponibles y sus metadatos de procedencia.
    def _load_predictions(self, model_dir: Path) -> None:
        configs = [
            (key, name, model_dir / f"predicciones_tramos_{key}.csv",
             model_dir / f"metricas_{key}.csv",
             model_dir / ("metadata_modelo.json" if key == DEFAULT_MODEL_KEY else f"metadata_modelo_{key}.json"))
            for key, name in MODEL_NAMES.items()
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
        marker = model_dir / "entrenamiento_completo.json"
        if marker.exists():
            selected = json.loads(marker.read_text(encoding="utf-8")).get("selected_for_routing")
            if selected in self._model_keys:
                self.default_model_key = selected
                self.model_name = MODEL_NAMES[selected]
                self.model_accuracy = float(self.model_metrics.get(self.model_name, {}).get("accuracy", 0.0))
                self._segment_scores = self._segment_scores_by_model[selected]
                self._prediction_rows = self._prediction_rows_by_model[selected]
                self._prediction_points = self._prediction_points_by_model[selected]
                self._prediction_tree = self._prediction_tree_by_model[selected]
                self._prediction_heatmap = self._prediction_heatmap_by_model[selected]
                metadata = self._model_metadata.get(selected, {})
                self.prediction_period = str(metadata.get("periodo_prediccion", "no disponible"))
                self.model_version = str(metadata.get("version_variables", "no disponible"))
                self.feature_count = len(metadata.get("variables", []))

    # Lee los scores, niveles y posibles predicciones por turno de un modelo exportado.
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
        turn_rows: dict[str, list[RiskPrediction]] = {
            turn: [] for turn in ("madrugada", "manana", "tarde", "noche")
        }
        turn_scores: dict[str, dict[str, RiskPrediction]] = {
            turn: {} for turn in turn_rows
        }
        prediction_points: list[dict] = []
        supports_turns = False
        prediction_period = "no disponible"
        with predictions_path.open("r", encoding="utf-8-sig", newline="") as archivo:
            reader = csv.DictReader(archivo)
            supports_turns = any(f"riesgo_score_{turn}" in (reader.fieldnames or []) for turn in turn_rows)
            for row in reader:
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
                prediction_period = row.get("periodo_objetivo") or prediction_period
                for turn in turn_rows if supports_turns else ():
                    raw_turn_score = row.get(f"riesgo_score_{turn}")
                    turn_score = score if raw_turn_score in (None, "") else min(
                        1.0, max(0.0, float(raw_turn_score))
                    )
                    turn_level = row.get(f"nivel_riesgo_{turn}") or level_from_score(turn_score)
                    turn_prediction = RiskPrediction(turn_score, turn_level)
                    turn_scores[turn][str(row["tramo_id"])] = turn_prediction
                    turn_rows[turn].append(turn_prediction)
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
            self._segment_scores_by_model_turno[key] = turn_scores if supports_turns else {}
            self._prediction_rows_by_model[key] = prediction_rows
            self._prediction_rows_by_model_turno[key] = turn_rows if supports_turns else {}
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
                metricas = {}
                for metric, value in row.items():
                    if metric == "modelo" or value in (None, ""):
                        continue
                    try:
                        metricas[metric] = float(value)
                    except ValueError:
                        metricas[metric] = value
                metricas["periodo_prueba"] = row.get("periodo_prueba", "")
                self.model_metrics[name] = metricas
                if key == DEFAULT_MODEL_KEY:
                    self.model_accuracy = float(metricas.get("accuracy", 0.0))
        metadata = {"periodo_prediccion": prediction_period, "tramo_turno": supports_turns}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as archivo:
                metadata.update(json.load(archivo))
        self._model_metadata[key] = metadata
        if key == DEFAULT_MODEL_KEY:
            self.prediction_period = str(metadata.get("periodo_prediccion", "no disponible"))
            self.model_version = str(metadata.get("version_variables", "no disponible"))
            self.feature_count = len(metadata.get("variables", []))

    # Resume los puntos de predicción en celdas para la capa de calor de consulta.
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
