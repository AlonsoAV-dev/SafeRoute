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
