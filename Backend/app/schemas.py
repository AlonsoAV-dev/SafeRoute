from typing import Literal

from pydantic import BaseModel, Field


class Coordinate(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


class RouteRequest(BaseModel):
    origin: Coordinate
    destination: Coordinate
    turno: str = Field(default="noche")
    safety_weight: float = Field(default=4.0, ge=0.0, le=10.0)


class RiskZone(BaseModel):
    cluster: int
    center: Coordinate
    radius_m: float
    risk_score: float
    risk_level: str
    total_crimes: int
    avg_crime_weight: float = 0.0


class CrimePoint(BaseModel):
    id: int
    location: Coordinate
    cluster: int
    turno: str
    tipo: str
    subtipo: str
    modalidad: str
    peso_delito: int
    distrito: str
    dia_semana: str


class RouteResponse(BaseModel):
    route: list[Coordinate]
    distance_km: float
    risk_score: float
    risk_level: str
    turno: str
    zones_considered: list[RiskZone]


class ApiRouteRequest(BaseModel):
    origin: tuple[float, float]
    destination: tuple[float, float]
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    datetime: str | None = None
    routePreference: Literal["safe", "fast"] = "safe"
    modelo_riesgo: Literal["auto", "random_forest", "xgboost"] = "auto"
    beta: float = Field(default=10.0, ge=0.0, le=20.0)
    buffer_m: Literal[50, 100, 150] = 100


class ApiRouteMetrics(BaseModel):
    alpha: float
    calc_time_ms: float


class ApiRouteResponse(BaseModel):
    safe_route: RouteResponse
    traditional_route: RouteResponse
    metrics: ApiRouteMetrics


class ApiHeatmapResponse(BaseModel):
    points: list[list[float]]


class ApiRiskZone(BaseModel):
    center: tuple[float, float]
    radius: float
    risk_level: str
    risk_score: float
    total_crimes: int
    avg_crime_weight: float


class ApiRiskZonesResponse(BaseModel):
    zones: list[ApiRiskZone]


class ApiStatsResponse(BaseModel):
    model_accuracy: float
    zones_count: int
    calc_time_ms: float
