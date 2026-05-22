from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from math import radians, sin, cos, sqrt, atan2, floor

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from app.services.preprocessing import CrimeRecord, normalize_turno


RISK_LEVELS = ("bajo", "medio", "alto")
TURNO_RISK = {
    "manana": 0.00,
    "tarde": 0.08,
    "noche": 0.18,
    "madrugada": 0.24,
}


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
    def __init__(self, records: list[CrimeRecord], clusters: int | None = None, grid_size_m: int = 100):
        if len(records) < 3:
            raise ValueError("Se necesitan al menos 3 registros delictivos validos.")

        self.records = records
        self.turno_encoder = LabelEncoder()
        if clusters is None:
            clusters = max(6, min(24, int(len(records) ** 0.5)))
        self.cluster_count = min(int(clusters), len(records))
        self.grid_size_m = max(40, int(grid_size_m))
        self._lat0 = sum(record.lat for record in records) / len(records)
        self._lat_m_per_deg = 111_320.0
        self._lng_m_per_deg = 111_320.0 * cos(radians(self._lat0))
        self.kmeans = KMeans(n_clusters=self.cluster_count, random_state=42, n_init=10)
        self.random_forest = RandomForestClassifier(n_estimators=80, random_state=42, max_depth=5)
        self.model_accuracy = 0.0
        self._fit()

    def _fit(self) -> None:
        coordinates = [[record.lat, record.lng] for record in self.records]
        clusters = self.kmeans.fit_predict(coordinates)
        cluster_totals = Counter(clusters)
        max_total = max(cluster_totals.values())

        self.cluster_base_risk = {
            cluster: total / max_total for cluster, total in cluster_totals.items()
        }

        self.record_clusters = list(clusters)
        turnos = self.turno_encoder.fit_transform([record.turno for record in self.records])

        features = []
        labels = []
        for record, cluster, turno_code in zip(self.records, clusters, turnos):
            score = min(1.0, self.cluster_base_risk[int(cluster)] * 0.78 + TURNO_RISK[record.turno])
            features.append([record.lat, record.lng, int(cluster), int(turno_code)])
            labels.append(level_from_score(score))

        self.random_forest.fit(features, labels)
        predictions = self.random_forest.predict(features)
        correct = sum(1 for expected, predicted in zip(labels, predictions) if expected == predicted)
        self.model_accuracy = correct / max(len(labels), 1)

    def predict_point(self, lat: float, lng: float, turno: str) -> RiskPrediction:
        normalized_turno = normalize_turno(turno)
        cluster = int(self.kmeans.predict([[lat, lng]])[0])
        turno_code = self._encode_turno(normalized_turno)
        rf_level = self.random_forest.predict([[lat, lng, cluster, turno_code]])[0]
        base_score = self.cluster_base_risk.get(cluster, 0.0)

        nearest_penalty = self._nearest_crime_penalty(lat, lng, normalized_turno)
        score = min(1.0, base_score * 0.62 + TURNO_RISK[normalized_turno] + nearest_penalty)
        return RiskPrediction(score=round(score, 3), level=max(rf_level, level_from_score(score), key=RISK_LEVELS.index))

    def get_zones(self, turno: str) -> list[dict]:
        normalized_turno = normalize_turno(turno)
        cells = self._grid_cells(normalized_turno)
        if not cells:
            return []

        max_total = max(cell["total"] for cell in cells)
        zones = []
        radius_m = max(40.0, self.grid_size_m * 0.55)
        for index, cell in enumerate(cells):
            score = min(1.0, (cell["total"] / max_total) * 0.78 + TURNO_RISK[normalized_turno])
            zones.append(
                {
                    "cluster": index,
                    "center": {
                        "lat": round(cell["center"][0], 6),
                        "lng": round(cell["center"][1], 6),
                    },
                    "radius_m": round(radius_m, 1),
                    "risk_score": round(score, 3),
                    "risk_level": level_from_score(score),
                    "total_crimes": cell["total"],
                }
            )

        return sorted(zones, key=lambda zone: zone["risk_score"], reverse=True)

    def get_heatmap_points(self, turno: str) -> list[list[float]]:
        normalized_turno = normalize_turno(turno)
        cells = self._grid_cells(normalized_turno)
        if not cells:
            return []

        max_total = max(cell["total"] for cell in cells)
        points = []
        for cell in cells:
            intensity = (cell["total"] / max_total) ** 0.75
            points.append([
                round(cell["center"][0], 6),
                round(cell["center"][1], 6),
                round(min(1.0, intensity), 3),
            ])
        return points

    def get_zone_count(self, turno: str) -> int:
        normalized_turno = normalize_turno(turno)
        return len(self._grid_cells(normalized_turno))

    def get_crime_points(self, turno: str | None = None) -> list[dict]:
        normalized_turno = normalize_turno(turno) if turno else None
        points = []

        for index, (record, cluster) in enumerate(zip(self.records, self.record_clusters), start=1):
            if normalized_turno and record.turno != normalized_turno:
                continue

            points.append(
                {
                    "id": index,
                    "location": {"lat": record.lat, "lng": record.lng},
                    "cluster": int(cluster),
                    "turno": record.turno,
                    "tipo": record.tipo,
                    "subtipo": record.subtipo,
                    "distrito": record.distrito,
                }
            )

        return points

    def _nearest_crime_penalty(self, lat: float, lng: float, turno: str) -> float:
        distances = [
            haversine_m(lat, lng, record.lat, record.lng)
            for record in self.records
            if record.turno == turno
        ]
        if not distances:
            return 0.0

        nearest = min(distances)
        if nearest <= 250:
            return 0.22
        if nearest <= 600:
            return 0.12
        if nearest <= 1000:
            return 0.06
        return 0.0

    def _encode_turno(self, turno: str) -> int:
        if turno in self.turno_encoder.classes_:
            return int(self.turno_encoder.transform([turno])[0])
        return int(self.turno_encoder.transform(["noche"])[0])

    def _grid_cell(self, lat: float, lng: float) -> tuple[str, tuple[float, float]]:
        x_m = lng * self._lng_m_per_deg
        y_m = lat * self._lat_m_per_deg
        grid_x = floor(x_m / self.grid_size_m)
        grid_y = floor(y_m / self.grid_size_m)
        center_x = (grid_x + 0.5) * self.grid_size_m
        center_y = (grid_y + 0.5) * self.grid_size_m
        center_lat = center_y / self._lat_m_per_deg
        center_lng = center_x / self._lng_m_per_deg
        return f"{grid_x}:{grid_y}", (center_lat, center_lng)

    def _grid_cells(self, turno: str) -> list[dict]:
        cells: dict[str, dict] = {}
        for record in self.records:
            if record.turno != turno:
                continue
            cell_id, center = self._grid_cell(record.lat, record.lng)
            cell = cells.get(cell_id)
            if not cell:
                cells[cell_id] = {"center": center, "total": 1}
            else:
                cell["total"] += 1
        return list(cells.values())
