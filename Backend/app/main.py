from datetime import datetime
import os
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.schemas import (
    ApiHeatmapResponse,
    ApiRouteRequest,
    ApiRouteResponse,
    ApiStatsResponse,
    CrimePoint,
    RouteRequest,
    RouteResponse,
)
from app.services.preprocessing import load_crime_records
from app.services.risk_model import RiskModel
from app.services.routing import (
    generate_route_comparison,
    generate_safe_route,
    preload_road_network,
)


BASE_DIR = Path(__file__).resolve().parent.parent
REAL_DATASET_PATH = BASE_DIR / "data" / "DELITOS TOTAL.csv"
DATASET_PATH = REAL_DATASET_PATH
PROCESSED_MODEL_DIR = BASE_DIR / "data" / "procesados"

records = load_crime_records(DATASET_PATH)
RISK_GRID_SIZE_M = int(os.getenv("RISK_GRID_SIZE_M", "100"))
risk_model = RiskModel(
    records,
    grid_size_m=RISK_GRID_SIZE_M,
    model_dir=PROCESSED_MODEL_DIR,
)
road_network = preload_road_network()

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
    return {
        "status": "ok",
        "records": len(records),
        "dataset": DATASET_PATH.name,
        "risk_model": risk_model.model_name,
        "model_version": risk_model.model_version,
        "feature_count": risk_model.feature_count,
        "prediction_period": risk_model.prediction_period,
        "available_models": sorted(risk_model._model_keys),
        "road_network": road_network,
    }


@app.get("/crime-points", response_model=list[CrimePoint])
def crime_points(
    turno: str | None = None,
    tipo: str | None = None,
    modalidad: str | None = None,
    dia_semana: str | None = None,
) -> list[dict]:
    points = risk_model.get_crime_points(turno, tipo, modalidad, dia_semana)
    return [
        {
            "id": point["id"],
            "location": {"lat": point["lat"], "lng": point["lng"]},
            "turno": point["turno"],
            "tipo": point["tipo"],
            "subtipo": point["subtipo"],
            "modalidad": point["modalidad"],
            "peso_delito": point["peso_delito"],
            "distrito": point["distrito"],
            "dia_semana": point["dia_semana"],
        }
        for point in points
    ]


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
        "zones_considered": [],
    }


@app.post("/api/route/calculate")
def api_route_calculate(request: ApiRouteRequest) -> dict:
    start = perf_counter()
    try:
        comparison = generate_route_comparison(
            origin=(request.origin[0], request.origin[1]),
            destination=(request.destination[0], request.destination[1]),
            risk_model=risk_model,
            modelo_riesgo=request.modelo_riesgo,
            beta=request.beta,
            buffer_m=request.buffer_m,
            risk_mode=request.risk_mode,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    elapsed_ms = (perf_counter() - start) * 1000
    return {
        **comparison,
        "route_preference": request.routePreference,
        "recommended_route": (
            "safe_route" if request.routePreference == "safe" else "traditional_route"
        ),
        "metrics": {
            "alpha": 1,
            "beta": comparison["parametros_a_star"]["beta_ruta_segura"],
            "calc_time_ms": round(elapsed_ms, 2),
        },
    }


@app.get("/api/heatmap", response_model=ApiHeatmapResponse)
def api_heatmap(
    turno: str | None = None,
    tipo: str | None = None,
    modalidad: str | None = None,
    dia_semana: str | None = None,
) -> dict:
    return {
        "points": risk_model.get_heatmap_points(turno, tipo, modalidad, dia_semana)
    }


@app.get("/api/prediction-heatmap", response_model=ApiHeatmapResponse)
def api_prediction_heatmap(modelo_riesgo: str | None = None) -> dict:
    return {"points": risk_model.get_prediction_heatmap_points(modelo_riesgo)}


@app.get("/api/prediction-points")
def api_prediction_points(
    min_score: float = 0.34,
    limit: int = 15_000,
    modelo_riesgo: str | None = None,
) -> dict:
    points = risk_model.get_prediction_points(
        min_score=min_score,
        limit=limit,
        modelo_riesgo=modelo_riesgo,
    )
    return {
        "points": points,
        "total": len(points),
        "prediction_period": risk_model.prediction_period,
        "model": risk_model.resolve_model(modelo_riesgo),
    }


@app.get("/api/crime-points")
def api_crime_points(
    turno: str | None = None,
    tipo: str | None = None,
    modalidad: str | None = None,
    dia_semana: str | None = None,
) -> dict:
    points = risk_model.get_crime_points(turno, tipo, modalidad, dia_semana)
    return {"points": points, "total": len(points)}


@app.get("/api/crime-filters")
def api_crime_filters() -> dict:
    return risk_model.get_filter_options()


@app.get("/api/stats", response_model=ApiStatsResponse)
def api_stats() -> dict:
    return {
        "model_accuracy": round(risk_model.model_accuracy, 3),
        "segments_count": risk_model.get_segment_count(),
        "prediction_period": risk_model.prediction_period,
        "calc_time_ms": 0.0,
    }
