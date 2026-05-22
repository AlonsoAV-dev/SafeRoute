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


class CrimePoint(BaseModel):
    id: int
    location: Coordinate
    cluster: int
    turno: str
    tipo: str
    subtipo: str
    distrito: str


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


class ApiRiskZonesResponse(BaseModel):
    zones: list[ApiRiskZone]


class ApiStatsResponse(BaseModel):
    model_accuracy: float
    zones_count: int
    calc_time_ms: float
