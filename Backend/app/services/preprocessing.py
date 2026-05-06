from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


VALID_TURNOS = {"manana", "tarde", "noche", "madrugada"}
LIMA_BOUNDS = {
    "min_lat": -13.60,
    "max_lat": -10.20,
    "min_lng": -78.20,
    "max_lng": -76.00,
}


@dataclass(frozen=True)
class CrimeRecord:
    lat: float
    lng: float
    turno: str
    tipo: str
    subtipo: str
    distrito: str
    fecha: str


def normalize_turno(value: str) -> str:
    cleaned = (value or "").strip().lower()
    replacements = {
        "mañana": "manana",
        "manana": "manana",
        "tarde": "tarde",
        "noche": "noche",
        "madrugada": "madrugada",
    }
    return replacements.get(cleaned, "noche")


def is_valid_coordinate(lat: float, lng: float) -> bool:
    return (
        LIMA_BOUNDS["min_lat"] <= lat <= LIMA_BOUNDS["max_lat"]
        and LIMA_BOUNDS["min_lng"] <= lng <= LIMA_BOUNDS["max_lng"]
    )


def load_crime_records(csv_path: Path) -> list[CrimeRecord]:
    records: list[CrimeRecord] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        reader = csv.DictReader(file, dialect=dialect)
        for row in reader:
            estado_coord = (row.get("ESTADO_COORD") or "").strip().upper()
            observacion = (row.get("OBSERVACION") or "").strip().upper()
            if estado_coord == "SIN COORDENADA":
                continue
            if "GEO FORZADA" in observacion:
                continue

            try:
                lat = float(row.get("lat_hecho", ""))
                lng = float(row.get("long_hecho", ""))
            except ValueError:
                continue

            if not is_valid_coordinate(lat, lng):
                continue

            records.append(
                CrimeRecord(
                    lat=lat,
                    lng=lng,
                    turno=normalize_turno(row.get("turno_hecho", "")),
                    tipo=(row.get("tipo_hecho") or "NO ESPECIFICADO").strip().upper(),
                    subtipo=(row.get("subtipo_hecho") or "NO ESPECIFICADO").strip().upper(),
                    distrito=(row.get("distrito_hecho") or "NO ESPECIFICADO").strip().upper(),
                    fecha=(row.get("fecha_hora_hecho") or "").strip(),
                )
            )

    return records
