from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import atan2, cos, floor, radians, sin, sqrt
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import BallTree

from app.services.preprocessing import CrimeRecord, normalize_turno


RISK_LEVELS = ("bajo", "medio", "alto")


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
    def __init__(
        self,
        records: list[CrimeRecord],
        clusters: int | None = None,
        grid_size_m: int = 100,
        model_dir: Path | None = None,
    ):
        if len(records) < 3:
            raise ValueError("Se necesitan al menos 3 registros delictivos válidos.")

        self.records = records
        if clusters is None:
            clusters = max(6, min(24, int(len(records) ** 0.5)))
        self.cluster_count = min(int(clusters), len(records))
        self.grid_size_m = max(40, int(grid_size_m))
        self._lat0 = sum(record.lat for record in records) / len(records)
        self._lat_m_per_deg = 111_320.0
        self._lng_m_per_deg = self._lat_m_per_deg * cos(radians(self._lat0))
        self.kmeans = KMeans(n_clusters=self.cluster_count, random_state=42, n_init=10)
        self.random_forest = RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            max_depth=7,
            class_weight="balanced",
        )
        self.model_accuracy = 0.0
        self.model_name = "Random Forest"
        self.model_metrics: dict[str, dict] = {}
        self._trained_segments: list[dict] = []
        self._trained_tree: BallTree | None = None
        self._crime_tree = BallTree(
            np.radians(np.asarray([[record.lat, record.lng] for record in records])),
            metric="haversine",
        )
        self._fit()
        if model_dir:
            self._load_trained_segments(model_dir)

    def _fit(self) -> None:
        coordinates = np.asarray([[record.lat, record.lng] for record in self.records])
        clusters = self.kmeans.fit_predict(coordinates)
        cluster_totals = Counter(int(cluster) for cluster in clusters)
        cluster_weights: dict[int, list[int]] = defaultdict(list)
        cluster_distances: dict[int, list[float]] = defaultdict(list)

        for record, cluster in zip(self.records, clusters):
            cluster_id = int(cluster)
            cluster_weights[cluster_id].append(record.peso_delito)
            center = self.kmeans.cluster_centers_[cluster_id]
            cluster_distances[cluster_id].append(
                haversine_m(record.lat, record.lng, float(center[0]), float(center[1]))
            )

        max_total = max(cluster_totals.values())
        self.cluster_stats = {}
        for cluster in range(self.cluster_count):
            total = cluster_totals.get(cluster, 0)
            weights = cluster_weights.get(cluster, [3])
            avg_weight = float(np.mean(weights))
            frequency_score = total / max_total if max_total else 0.0
            severity_score = avg_weight / 10
            risk_score = min(1.0, 0.30 * frequency_score + 0.70 * severity_score)
            distances = cluster_distances.get(cluster, [self.grid_size_m])
            radius_m = float(np.percentile(distances, 75))
            self.cluster_stats[cluster] = {
                "total": total,
                "avg_weight": avg_weight,
                "frequency_score": frequency_score,
                "risk_score": risk_score,
                "radius_m": max(120.0, min(900.0, radius_m)),
            }

        self.cluster_base_risk = {
            cluster: stats["risk_score"] for cluster, stats in self.cluster_stats.items()
        }
        self.record_clusters = [int(cluster) for cluster in clusters]
        features = []
        labels = []
        for record, cluster in zip(self.records, self.record_clusters):
            features.append([record.lat, record.lng, cluster, record.peso_delito])
            labels.append(level_from_score(self.cluster_base_risk[cluster]))
        self.random_forest.fit(features, labels)
        predictions = self.random_forest.predict(features)
        self.model_accuracy = float(np.mean(predictions == np.asarray(labels)))

    def predict_point(
        self,
        lat: float,
        lng: float,
        turno: str | None = None,
        modelo_riesgo: str = "auto",
    ) -> RiskPrediction:
        del turno  # El riesgo de rutas usa todo el historial, no un filtro temporal.
        trained_prediction = self._predict_from_trained_segments(
            lat, lng, self.resolve_model(modelo_riesgo)
        )
        if trained_prediction:
            return trained_prediction

        cluster = int(self.kmeans.predict([[lat, lng]])[0])
        score = self.cluster_base_risk.get(cluster, 0.0)
        return RiskPrediction(score=round(score, 3), level=level_from_score(score))

    def resolve_model(self, requested: str) -> str:
        normalized = (requested or "auto").strip().lower()
        if normalized == "random_forest":
            return "Random Forest"
        if normalized == "xgboost":
            return "XGBoost"
        return self.model_name

    def metrics_for_model(self, requested: str) -> dict:
        return self.model_metrics.get(self.resolve_model(requested), {})

    def nearby_crime_stats(
        self,
        sample_points: list[tuple[float, float]],
        radius_m: float,
    ) -> dict:
        if not sample_points:
            return {"count": 0, "weight_sum": 0.0, "weight_avg": 0.0}
        indices: set[int] = set()
        radius_rad = radius_m / 6_371_000
        query = self._crime_tree.query_radius(
            np.radians(np.asarray(sample_points)),
            r=radius_rad,
        )
        for result in query:
            indices.update(int(index) for index in result)
        weights = [self.records[index].peso_delito for index in indices]
        return {
            "count": len(weights),
            "weight_sum": float(sum(weights)),
            "weight_avg": float(np.mean(weights)) if weights else 0.0,
        }

    def get_zones(self, turno: str | None = None) -> list[dict]:
        del turno
        zones = []
        for cluster, center in enumerate(self.kmeans.cluster_centers_):
            stats = self.cluster_stats[cluster]
            zones.append(
                {
                    "cluster": cluster,
                    "center": {
                        "lat": round(float(center[0]), 6),
                        "lng": round(float(center[1]), 6),
                    },
                    "radius_m": round(stats["radius_m"], 1),
                    "risk_score": round(stats["risk_score"], 3),
                    "risk_level": level_from_score(stats["risk_score"]),
                    "total_crimes": stats["total"],
                    "avg_crime_weight": round(stats["avg_weight"], 3),
                }
            )
        return sorted(zones, key=lambda zone: zone["risk_score"], reverse=True)

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
        max_total = max(cell["total"] for cell in cells)
        raw_values = np.asarray(
            [
                0.30 * (cell["total"] / max_total)
                + 0.70 * (cell["weight_sum"] / cell["total"] / 5)
                for cell in cells
            ],
            dtype=float,
        )
        p95 = float(np.percentile(raw_values, 95)) if len(raw_values) else 1.0
        p95 = max(p95, 1e-9)
        clipped_values = np.clip(raw_values, 0, p95)
        order = np.argsort(clipped_values, kind="stable")
        normalized_values = np.empty(len(clipped_values), dtype=float)
        normalized_values[order] = np.linspace(0, 1, len(clipped_values))
        points = []
        for cell, intensity in zip(cells, normalized_values):
            points.append(
                [
                    round(cell["center"][0], 6),
                    round(cell["center"][1], 6),
                    round(intensity, 3),
                ]
            )
        return points

    def get_zone_count(self, turno: str | None = None) -> int:
        del turno
        return self.cluster_count

    def get_crime_points(
        self,
        turno: str | None = None,
        tipo: str | None = None,
        modalidad: str | None = None,
        dia_semana: str | None = None,
    ) -> list[dict]:
        filters_active = any((turno, tipo, modalidad, dia_semana))
        points = []
        for index, (record, cluster) in enumerate(
            zip(self.records, self.record_clusters), start=1
        ):
            if filters_active and not self._matches_filters(
                record, turno, tipo, modalidad, dia_semana
            ):
                continue
            points.append(
                {
                    "id": index,
                    "lat": record.lat,
                    "lng": record.lng,
                    "cluster": cluster,
                    "turno": record.turno,
                    "tipo": record.tipo,
                    "subtipo": record.subtipo,
                    "modalidad": record.modalidad,
                    "peso_delito": record.peso_delito,
                    "distrito": record.distrito,
                    "dia_semana": record.dia_semana,
                }
            )
        return points

    def get_filter_options(self) -> dict:
        return {
            "turnos": ["todos", "manana", "tarde", "noche", "madrugada"],
            "dias_semana": ["todos", *sorted({record.dia_semana for record in self.records})],
            "tipos": ["todos", *sorted({record.tipo for record in self.records})],
            "modalidades": ["todos", *sorted({record.modalidad for record in self.records})],
        }

    def _filter_records(
        self,
        turno: str | None,
        tipo: str | None,
        modalidad: str | None,
        dia_semana: str | None,
    ) -> list[CrimeRecord]:
        return [
            record
            for record in self.records
            if self._matches_filters(record, turno, tipo, modalidad, dia_semana)
        ]

    @staticmethod
    def _matches_filters(
        record: CrimeRecord,
        turno: str | None,
        tipo: str | None,
        modalidad: str | None,
        dia_semana: str | None,
    ) -> bool:
        normalized_turno = normalize_turno(turno) if turno and turno != "todos" else None
        return (
            (not normalized_turno or record.turno == normalized_turno)
            and (not tipo or tipo == "todos" or record.tipo == tipo)
            and (not modalidad or modalidad == "todos" or record.modalidad == modalidad)
            and (
                not dia_semana
                or dia_semana == "todos"
                or record.dia_semana == dia_semana
            )
        )

    def _load_trained_segments(self, model_dir: Path) -> None:
        predictions_path = model_dir / "predicciones_riesgo.csv"
        metrics_path = model_dir / "metricas_modelos.csv"
        winner_path = model_dir / "modelo_ganador.json"
        if not predictions_path.exists():
            return

        with predictions_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        valid_rows = []
        coordinates = []
        for row in rows:
            try:
                lat = float(row["latitud"])
                lng = float(row["longitud"])
                score = float(row["riesgo_score"])
                probability = float(row["probabilidad_riesgo_alto"])
            except (KeyError, TypeError, ValueError):
                continue
            valid_rows.append(
                {
                    "score": score,
                    "level": row.get("riesgo_predicho") or row.get("nivel_riesgo") or "medio",
                    "probability_high": probability,
                    "models": {
                        "Random Forest": {
                            "score": float(row.get("score_modelo_random_forest", score)),
                            "level": row.get(
                                "riesgo_predicho_random_forest",
                                row.get("riesgo_predicho") or "medio",
                            ),
                            "probability_high": float(
                                row.get("probabilidad_alto_random_forest", probability)
                            ),
                        },
                        "XGBoost": {
                            "score": float(row.get("score_modelo_xgboost", score)),
                            "level": row.get(
                                "riesgo_predicho_xgboost",
                                row.get("riesgo_predicho") or "medio",
                            ),
                            "probability_high": float(
                                row.get("probabilidad_alto_xgboost", probability)
                            ),
                        },
                    },
                }
            )
            coordinates.append([radians(lat), radians(lng)])
        if not valid_rows:
            return

        self._trained_segments = valid_rows
        self._trained_tree = BallTree(np.asarray(coordinates), metric="haversine")
        if winner_path.exists():
            with winner_path.open("r", encoding="utf-8") as file:
                winner = json.load(file)
            self.model_name = str(winner.get("modelo", self.model_name))
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8-sig", newline="") as file:
                metrics = list(csv.DictReader(file))
            for row in metrics:
                try:
                    self.model_metrics[str(row["modelo"])] = {
                        "accuracy": float(row["accuracy"]),
                        "precision": float(row["precision_macro"]),
                        "recall": float(row["recall_macro"]),
                        "f1_score": float(row["f1_macro"]),
                        "recall_riesgo_alto": float(row["recall_riesgo_alto"]),
                        "matriz_confusion": json.loads(row["matriz_confusion"]),
                    }
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
            winner_metrics = next(
                (row for row in metrics if row.get("modelo") == self.model_name),
                None,
            )
            if winner_metrics:
                self.model_accuracy = float(winner_metrics.get("accuracy", self.model_accuracy))

    def _predict_from_trained_segments(
        self,
        lat: float,
        lng: float,
        model_name: str,
    ) -> RiskPrediction | None:
        if self._trained_tree is None:
            return None
        _, indices = self._trained_tree.query(
            np.asarray([[radians(lat), radians(lng)]]),
            k=1,
        )
        segment = self._trained_segments[int(indices[0][0])]
        model_values = segment["models"].get(model_name, segment)
        score = min(1.0, float(model_values["score"]))
        level = max(
            str(model_values["level"]),
            level_from_score(score),
            key=RISK_LEVELS.index,
        )
        return RiskPrediction(score=round(score, 3), level=level)

    def _grid_cell(self, lat: float, lng: float) -> tuple[str, tuple[float, float]]:
        x_m = lng * self._lng_m_per_deg
        y_m = lat * self._lat_m_per_deg
        grid_x = floor(x_m / self.grid_size_m)
        grid_y = floor(y_m / self.grid_size_m)
        center_x = (grid_x + 0.5) * self.grid_size_m
        center_y = (grid_y + 0.5) * self.grid_size_m
        return (
            f"{grid_x}:{grid_y}",
            (center_y / self._lat_m_per_deg, center_x / self._lng_m_per_deg),
        )

    def _grid_cells(self, records: list[CrimeRecord]) -> list[dict]:
        cells: dict[str, dict] = {}
        for record in records:
            cell_id, center = self._grid_cell(record.lat, record.lng)
            cell = cells.setdefault(
                cell_id,
                {"center": center, "total": 0, "weight_sum": 0},
            )
            cell["total"] += 1
            cell["weight_sum"] += record.peso_delito
        return list(cells.values())
