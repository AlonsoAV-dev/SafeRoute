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


def _parse_float(value: str | None) -> float:
    cleaned = (value or "").strip().replace(",", ".")
    if not cleaned:
        raise ValueError("empty numeric value")
    return float(cleaned)


def _first_non_empty(row: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def load_crime_records(csv_path: Path) -> list[CrimeRecord]:
    records: list[CrimeRecord] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        except csv.Error:
            dialect = csv.excel
            dialect.delimiter = ";"
        reader = csv.DictReader(file, dialect=dialect)
        for row in reader:
            try:
                lat = _parse_float(_first_non_empty(row, ("lat_hecho", "y")))
                lng = _parse_float(_first_non_empty(row, ("long_hecho", "x")))
            except ValueError:
                continue

            if not is_valid_coordinate(lat, lng):
                continue

            records.append(
                CrimeRecord(
                    lat=lat,
                    lng=lng,
                    turno=normalize_turno(_first_non_empty(row, ("turno_hecho", "turno"))),
                    tipo=(_first_non_empty(row, ("tipo_hecho", "tipo")) or "NO ESPECIFICADO").upper(),
                    subtipo=(_first_non_empty(row, ("subtipo_hecho", "subtipo")) or "NO ESPECIFICADO").upper(),
                    distrito=(_first_non_empty(row, ("distrito_hecho", "distrito")) or "NO ESPECIFICADO").upper(),
                    fecha=_first_non_empty(row, ("fecha_hora_hecho", "fecha_hora_registro_hecho", "fecha")),
                )
            )

    return records
