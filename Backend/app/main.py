from datetime import datetime
from math import isfinite
import os
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query
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
from app.services.risk_surface import RiskSurface
from app.services.risk_segments import RiskSegments
from app.services.route_cache import RouteComparisonCache
from app.services.routing import (
    generate_route_comparison,
    generate_safe_route,
    preload_road_network,
)


# Define las rutas de datos y permite seleccionar otros artefactos mediante variables de entorno.
BASE_DIR = Path(__file__).resolve().parent.parent
REAL_DATASET_PATH = BASE_DIR / "data" / "DELITOS TOTAL.csv"
SIDPOL_MODEL_DIR = BASE_DIR / "data" / "procesados_sidpol_v2"
SIDPOL_READY = (SIDPOL_MODEL_DIR / "entrenamiento_completo.json").exists()
DATASET_PATH = Path(os.getenv(
    "SAFEROUTE_DATASET_PATH",
    str(SIDPOL_MODEL_DIR / "delitos_geolocalizados.csv" if SIDPOL_READY else REAL_DATASET_PATH),
))
PROCESSED_MODEL_DIR = Path(os.getenv(
    "SAFEROUTE_MODEL_DIR",
    str(SIDPOL_MODEL_DIR if SIDPOL_READY else BASE_DIR / "data" / "procesados"),
))

# Carga una vez los delitos, las predicciones y la red vial que utilizarán las consultas.
records = load_crime_records(DATASET_PATH)
RISK_GRID_SIZE_M = int(os.getenv("RISK_GRID_SIZE_M", "100"))
risk_model = RiskModel(
    records,
    grid_size_m=RISK_GRID_SIZE_M,
    model_dir=PROCESSED_MODEL_DIR,
)
road_network = preload_road_network()
risk_surface = risk_model.spatial_risk_surface()
risk_segments = RiskSegments(risk_model)
route_comparison_cache = RouteComparisonCache()

# Configura la API y el acceso desde el frontend local mediante CORS.
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


# Obtiene el turno horario en la zona de Lima para las consultas que incluyen una fecha.
def _turno_from_datetime(value: str | None) -> str:
    if not value:
        hour = datetime.now(ZoneInfo("America/Lima")).hour
    else:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(ZoneInfo("America/Lima"))
            hour = parsed.hour
        except ValueError:
            hour = datetime.now(ZoneInfo("America/Lima")).hour
    if 0 <= hour < 6:
        return "madrugada"
    if 6 <= hour < 12:
        return "manana"
    if 12 <= hour < 18:
        return "tarde"
    return "noche"


# Informa el estado de la API, los artefactos cargados y los modelos disponibles.
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


# Entrega al selector web el nombre, periodo y disponibilidad de cada modelo.
@app.get("/api/models")
def api_models() -> dict:
    return {"models": risk_model.available_models(), "default_model": risk_model.default_model_key}


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


# Valida la consulta, reutiliza resultados equivalentes y compara las rutas con sus métricas.
@app.post("/api/route/calculate")
def api_route_calculate(request: ApiRouteRequest) -> dict:
    start = perf_counter()
    try:
        info = risk_model.model_info(request.modelo_riesgo)
        cache_key = (info["key"], info["prediction_period"], risk_model.model_version,
                     risk_surface.field_config()["version"], request.origin, request.destination,
                     request.beta, request.buffer_m, request.risk_mode, request.turno)
        comparison, cache_hit = route_comparison_cache.get_or_compute(cache_key, lambda: generate_route_comparison(
            origin=(request.origin[0], request.origin[1]),
            destination=(request.destination[0], request.destination[1]),
            risk_model=risk_model,
            modelo_riesgo=request.modelo_riesgo,
            beta=request.beta,
            buffer_m=request.buffer_m,
            risk_mode=request.risk_mode,
            turno=request.turno,
        ))
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
            "alpha": comparison["parametros_a_star"]["alpha"],
            "beta": comparison["parametros_a_star"]["beta_ruta_segura"],
            "calc_time_ms": round(elapsed_ms, 2),
            "cache_hit": cache_hit,
        },
    }


# Construye la capa histórica con los filtros solicitados por el usuario.
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
    try:
        risk_model.model_info(modelo_riesgo)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"points": risk_model.get_prediction_heatmap_points(modelo_riesgo)}


# Entrega las geometrías viales evaluadas dentro del área visible del mapa.
@app.get("/api/risk-segments")
def api_risk_segments(bbox: str, zoom: int = Query(default=13, ge=1, le=20),
                      modelo_riesgo: str | None = None) -> dict:
    try:
        bounds = tuple(float(value) for value in bbox.split(","))
        if (len(bounds) != 4 or not all(isfinite(value) for value in bounds)
                or not -85 <= bounds[0] < bounds[2] <= 85
                or not -180 <= bounds[1] < bounds[3] <= 180):
            raise ValueError("El área del mapa debe contener sur, oeste, norte y este válidos.")
        return risk_segments.get_segments(bounds, zoom, modelo_riesgo)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


# Entrega el campo espacial del modelo y periodo seleccionados, compartido con el ruteo.
@app.get("/api/risk-surface")
def api_risk_surface(bbox: str, zoom: int = Query(default=13, ge=1, le=20),
                     modelo_riesgo: str | None = None) -> dict:
    try:
        bounds = tuple(float(value) for value in bbox.split(","))
        if (len(bounds) != 4 or not all(isfinite(value) for value in bounds)
                or not (-85 <= bounds[0] < bounds[2] <= 85 and -180 <= bounds[1] < bounds[3] <= 180)):
            raise ValueError("El área del mapa debe tener límites sur, oeste, norte y este válidos.")
        return risk_surface.get_surface(bounds, zoom, modelo_riesgo)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


# Expone las predicciones por segmento con filtros de nivel y área geográfica.
@app.get("/api/prediction-points")
def api_prediction_points(
    min_score: float = 0.0,
    limit: int = 15_000,
    modelo_riesgo: str | None = None,
    balanced: bool = False,
    bbox: str | None = None,
) -> dict:
    try:
        model_info = risk_model.model_info(modelo_riesgo)
        bounds = tuple(float(value) for value in bbox.split(",")) if bbox else None
        if bounds and (len(bounds) != 4 or not (-90 <= bounds[0] < bounds[2] <= 90 and -180 <= bounds[1] < bounds[3] <= 180)):
            raise ValueError("El área del mapa debe tener los límites sur, oeste, norte y este válidos.")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    points = risk_model.get_prediction_points(
        min_score=min_score,
        limit=limit,
        modelo_riesgo=modelo_riesgo,
        balanced=balanced,
        bounds=bounds,
    )
    return {
        "points": points,
        "total": len(points),
        "counts_by_level": risk_model.get_prediction_counts(modelo_riesgo),
        "visible_count": sum(risk_model.get_prediction_counts(modelo_riesgo, bounds).values()),
        "prediction_period": model_info["prediction_period"],
        "risk_scope": "mensual",
        "model": risk_model.resolve_model(modelo_riesgo),
    }


# Devuelve los registros históricos filtrados para la capa de delitos.
@app.get("/api/crime-points")
def api_crime_points(
    turno: str | None = None,
    tipo: str | None = None,
    modalidad: str | None = None,
    dia_semana: str | None = None,
) -> dict:
    points = risk_model.get_crime_points(turno, tipo, modalidad, dia_semana)
    return {"points": points, "total": len(points)}


# Proporciona los valores disponibles para los controles de filtrado.
@app.get("/api/crime-filters")
def api_crime_filters() -> dict:
    return risk_model.get_filter_options()


# Resume las métricas disponibles del modelo y el periodo de predicción.
@app.get("/api/stats", response_model=ApiStatsResponse)
def api_stats() -> dict:
    return {
        "model_accuracy": round(risk_model.model_accuracy, 3),
        "segments_count": risk_model.get_segment_count(),
        "prediction_period": risk_model.prediction_period,
        "calc_time_ms": 0.0,
    }
