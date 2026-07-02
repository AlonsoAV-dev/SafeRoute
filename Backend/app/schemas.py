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


class CrimePoint(BaseModel):
    id: int
    location: Coordinate
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
    zones_considered: list[dict]


class ApiRouteRequest(BaseModel):
    origin: tuple[float, float]
    destination: tuple[float, float]
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    datetime: str | None = None
    routePreference: Literal["safe", "fast"] = "safe"
    modelo_riesgo: Literal["random_forest"] = "random_forest"
    beta: float = Field(default=10.0, ge=0.0, le=20.0)
    buffer_m: Literal[50, 100, 150, 200] = 200
    risk_mode: Literal["predicted", "historical", "hybrid"] = "predicted"


class ApiRouteMetrics(BaseModel):
    alpha: float
    calc_time_ms: float


class ApiRouteResponse(BaseModel):
    safe_route: RouteResponse
    traditional_route: RouteResponse
    metrics: ApiRouteMetrics


class ApiHeatmapResponse(BaseModel):
    points: list[list[float]]


class ApiStatsResponse(BaseModel):
    model_accuracy: float
    segments_count: int
    prediction_period: str
    calc_time_ms: float
