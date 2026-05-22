from datetime import datetime
import os
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.schemas import (
    ApiHeatmapResponse,
    ApiRiskZonesResponse,
    ApiRouteRequest,
    ApiRouteResponse,
    ApiStatsResponse,
    CrimePoint,
    RouteRequest,
    RouteResponse,
    RiskZone,
)
from app.services.preprocessing import load_crime_records
from app.services.risk_model import RiskModel
from app.services.routing import generate_safe_route


BASE_DIR = Path(__file__).resolve().parent.parent
REAL_DATASET_PATH = BASE_DIR / "data" / "AAV-DATASET.csv"
LEGACY_DATASET_PATH = BASE_DIR / "data" / "DB-1JAN-28MARCH.csv"
SAMPLE_DATASET_PATH = BASE_DIR / "data" / "sample_crimes.csv"
DATASET_PATH = next(
    (path for path in (REAL_DATASET_PATH, LEGACY_DATASET_PATH, SAMPLE_DATASET_PATH) if path.exists()),
    REAL_DATASET_PATH,
)

records = load_crime_records(DATASET_PATH)
RISK_GRID_SIZE_M = int(os.getenv("RISK_GRID_SIZE_M", "100"))
risk_model = RiskModel(records, grid_size_m=RISK_GRID_SIZE_M)

app = FastAPI(
    title="SafeRoute API",
    description="API simple para recomendar rutas priorizando seguridad.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _turno_from_datetime(value: str | None) -> str:
    if not value:
        hour = datetime.now().hour
    else:
        try:
            parsed = datetime.fromisoformat(value)
            hour = parsed.hour
        except ValueError:
            hour = datetime.now().hour
    if 0 <= hour < 6:
        return "madrugada"
    if 6 <= hour < 12:
        return "manana"
    if 12 <= hour < 18:
        return "tarde"
    return "noche"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "records": len(records), "dataset": DATASET_PATH.name}


@app.get("/risk-zones", response_model=list[RiskZone])
def risk_zones(turno: str = "noche") -> list[dict]:
    return risk_model.get_zones(turno)


@app.get("/crime-points", response_model=list[CrimePoint])
def crime_points(turno: str | None = None) -> list[dict]:
    return risk_model.get_crime_points(turno)


@app.post("/route", response_model=RouteResponse)
def route(request: RouteRequest) -> dict:
    try:
        route_data = generate_safe_route(
            origin=(request.origin.lat, request.origin.lng),
            destination=(request.destination.lat, request.destination.lng),
            turno=request.turno,
            risk_model=risk_model,
            safety_weight=request.safety_weight,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {
        **route_data,
        "turno": request.turno,
        "zones_considered": risk_model.get_zones(request.turno),
    }


@app.post("/api/route/calculate", response_model=ApiRouteResponse)
def api_route_calculate(request: ApiRouteRequest) -> dict:
    start = perf_counter()
    turno = _turno_from_datetime(request.datetime)
    try:
        safe_route = generate_safe_route(
            origin=(request.origin[0], request.origin[1]),
            destination=(request.destination[0], request.destination[1]),
            turno=turno,
            risk_model=risk_model,
            safety_weight=0.0,
            alpha=request.alpha,
        )
        traditional_route = generate_safe_route(
            origin=(request.origin[0], request.origin[1]),
            destination=(request.destination[0], request.destination[1]),
            turno=turno,
            risk_model=risk_model,
            safety_weight=0.0,
            alpha=0.0,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    elapsed_ms = (perf_counter() - start) * 1000
    zones = risk_model.get_zones(turno)
    return {
        "safe_route": {**safe_route, "turno": turno, "zones_considered": zones},
        "traditional_route": {**traditional_route, "turno": turno, "zones_considered": zones},
        "metrics": {"alpha": request.alpha, "calc_time_ms": round(elapsed_ms, 2)},
    }


@app.get("/api/risk-zones", response_model=ApiRiskZonesResponse)
def api_risk_zones(turno: str = "noche") -> dict:
    zones = risk_model.get_zones(turno)
    return {
        "zones": [
            {
                "center": (zone["center"]["lat"], zone["center"]["lng"]),
                "radius": zone["radius_m"],
                "risk_level": zone["risk_level"],
            }
            for zone in zones
        ]
    }


@app.get("/api/heatmap", response_model=ApiHeatmapResponse)
def api_heatmap(turno: str = "noche") -> dict:
    return {"points": risk_model.get_heatmap_points(turno)}


@app.get("/api/stats", response_model=ApiStatsResponse)
def api_stats() -> dict:
    return {
        "model_accuracy": round(risk_model.model_accuracy, 3),
        "zones_count": risk_model.get_zone_count("noche"),
        "calc_time_ms": 0.0,
    }
