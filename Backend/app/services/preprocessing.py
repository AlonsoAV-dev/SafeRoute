from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.services.pesos_delito import normalizar_modalidad, obtener_peso_delito


VALID_TURNOS = {"manana", "tarde", "noche", "madrugada"}
LIMA_BOUNDS = {
    "min_lat": -13.60,
    "max_lat": -10.20,
    "min_lng": -78.20,
    "max_lng": -76.00,
}
DAYS_ES = (
    "lunes",
    "martes",
    "miercoles",
    "jueves",
    "viernes",
    "sabado",
    "domingo",
)


@dataclass(frozen=True)
class CrimeRecord:
    lat: float
    lng: float
    turno: str
    tipo: str
    subtipo: str
    modalidad: str
    peso_delito: int
    distrito: str
    fecha: str
    dia_semana: str


def normalize_turno(value: str) -> str:
    cleaned = normalizar_modalidad(value).lower()
    return cleaned if cleaned in VALID_TURNOS else "noche"


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


def _day_of_week(row: dict[str, str]) -> str:
    try:
        year_value = _first_non_empty(row, ("año_hecho", "anio_hecho")).replace(",", "")
        year = int(float(year_value))
        month = int(float(_first_non_empty(row, ("mes_hecho", "mes"))))
        day = int(float(_first_non_empty(row, ("dia_hecho", "dia"))))
        return DAYS_ES[datetime(year, month, day).weekday()]
    except (TypeError, ValueError):
        return "desconocido"


def load_crime_records(csv_path: Path) -> list[CrimeRecord]:
    records: list[CrimeRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        first_line = file.readline()
        file.seek(0)
        delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
        reader = csv.DictReader(file, delimiter=delimiter)
        for row in reader:
            try:
                lat = _parse_float(_first_non_empty(row, ("lat_hecho", "y")))
                lng = _parse_float(_first_non_empty(row, ("long_hecho", "x")))
            except ValueError:
                continue
            if not is_valid_coordinate(lat, lng):
                continue

            modalidad = normalizar_modalidad(
                _first_non_empty(row, ("modalidad_hecho", "modalidad_he", "modalidad"))
            ) or "NO ESPECIFICADO"
            records.append(
                CrimeRecord(
                    lat=lat,
                    lng=lng,
                    turno=normalize_turno(_first_non_empty(row, ("turno_hecho", "turno"))),
                    tipo=normalizar_modalidad(
                        _first_non_empty(row, ("tipo_hecho", "tipo"))
                    )
                    or "NO ESPECIFICADO",
                    subtipo=normalizar_modalidad(
                        _first_non_empty(row, ("subtipo_hecho", "subtipo"))
                    )
                    or "NO ESPECIFICADO",
                    modalidad=modalidad,
                    peso_delito=obtener_peso_delito(modalidad),
                    distrito=normalizar_modalidad(
                        _first_non_empty(row, ("distrito_hecho", "distrito"))
                    )
                    or "NO ESPECIFICADO",
                    fecha=_first_non_empty(
                        row, ("fecha_hora_hecho", "fecha_hora_registro_hecho", "fecha")
                    ),
                    dia_semana=_day_of_week(row),
                )
            )
    return records
