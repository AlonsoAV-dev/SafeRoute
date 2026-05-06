from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from math import radians, sin, cos, sqrt, atan2

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
    def __init__(self, records: list[CrimeRecord], clusters: int = 4):
        if len(records) < 3:
            raise ValueError("Se necesitan al menos 3 registros delictivos validos.")

        self.records = records
        self.turno_encoder = LabelEncoder()
        self.cluster_count = min(clusters, len(records))
        self.kmeans = KMeans(n_clusters=self.cluster_count, random_state=42, n_init=10)
        self.random_forest = RandomForestClassifier(n_estimators=80, random_state=42, max_depth=5)
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
        zones = []
        for cluster, center in enumerate(self.kmeans.cluster_centers_):
            cluster_records = [
                record
                for record, assigned_cluster in zip(self.records, self.record_clusters)
                if int(assigned_cluster) == cluster
            ]
            if not cluster_records:
                continue

            radius = max(
                haversine_m(center[0], center[1], record.lat, record.lng)
                for record in cluster_records
            )
            score = min(1.0, self.cluster_base_risk.get(cluster, 0.0) * 0.78 + TURNO_RISK[normalized_turno])
            zones.append(
                {
                    "cluster": cluster,
                    "center": {"lat": round(float(center[0]), 6), "lng": round(float(center[1]), 6)},
                    "radius_m": round(max(radius, 450), 1),
                    "risk_score": round(score, 3),
                    "risk_level": level_from_score(score),
                    "total_crimes": len(cluster_records),
                }
            )
        return sorted(zones, key=lambda zone: zone["risk_score"], reverse=True)

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
