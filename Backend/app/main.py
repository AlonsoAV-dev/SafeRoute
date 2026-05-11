from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.schemas import CrimePoint, RouteRequest, RouteResponse, RiskZone
from app.services.preprocessing import load_crime_records
from app.services.risk_model import RiskModel
from app.services.routing import generate_safe_route


BASE_DIR = Path(__file__).resolve().parent.parent
REAL_DATASET_PATH = BASE_DIR / "data" / "DB-1JAN-28MARCH.csv"
SAMPLE_DATASET_PATH = BASE_DIR / "data" / "sample_crimes.csv"
DATASET_PATH = REAL_DATASET_PATH if REAL_DATASET_PATH.exists() else SAMPLE_DATASET_PATH

records = load_crime_records(DATASET_PATH)
risk_model = RiskModel(records)

app = FastAPI(
    title="SafeRoute API",
    description="API simple para recomendar rutas priorizando seguridad.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
