from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.services.pesos_delito import normalizar_modalidad, obtener_peso_desde_campos


VALID_TURNOS = {"manana", "tarde", "noche", "madrugada"}
LIMA_BOUNDS = {
    "min_lat": -13.50,
    "max_lat": -10.50,
    "min_lng": -78.00,
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


@dataclass(frozen=True, slots=True)
# Representa un delito normalizado que puede consultar el servicio de riesgo.
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


# Homogeneiza las etiquetas de turno antes de aplicar filtros.
def normalize_turno(value: str) -> str:
    cleaned = normalizar_modalidad(value).lower()
    return cleaned if cleaned in VALID_TURNOS else "noche"


# Descarta coordenadas fuera de los límites geográficos utilizados por el proyecto.
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
        event_date = _first_non_empty(row, ("fecha_hora_hecho", "fecha"))
        if event_date:
            return DAYS_ES[datetime.fromisoformat(event_date.replace("/", "-")).weekday()]
        year_value = _first_non_empty(
            row, ("año_hecho", "anio_hecho", "aÃ±o_hecho")
        ).replace(",", "")
        year = int(float(year_value))
        month = int(float(_first_non_empty(row, ("mes_hecho", "mes"))))
        day = int(float(_first_non_empty(row, ("dia_hecho", "dia"))))
        return DAYS_ES[datetime(year, month, day).weekday()]
    except (TypeError, ValueError):
        return "desconocido"


# Reconstruye la fecha del hecho y utiliza los campos alternativos si faltan sus componentes.
def _normalized_date(row: dict[str, str]) -> str:
    try:
        year_value = _first_non_empty(
            row, ("año_hecho", "anio_hecho", "aÃ±o_hecho")
        ).replace(",", "")
        year = int(float(year_value))
        month = int(float(_first_non_empty(row, ("mes_hecho", "mes"))))
        day = int(float(_first_non_empty(row, ("dia_hecho", "dia"))))
        return datetime(year, month, day).date().isoformat()
    except (TypeError, ValueError):
        return _first_non_empty(
            row, ("fecha_hora_hecho", "fecha_hora_registro_hecho", "fecha")
        )


# Lee los formatos CSV admitidos, valida coordenadas y conserva la información delictiva normalizada.
def load_crime_records(csv_path: Path) -> list[CrimeRecord]:
    records: list[CrimeRecord] = []
    identifiers: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        first_line = file.readline()
        file.seek(0)
        delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
        reader = csv.DictReader(file, delimiter=delimiter)
        deduplicate_ids = "id_hecho" not in (reader.fieldnames or [])
        for row in reader:
            identifier = _first_non_empty(
                row, ("GlobalID", "globalid", "id_dgc", "ID_DGC_03", "OBJECTID")
            )
            if deduplicate_ids and identifier and identifier in identifiers:
                continue
            try:
                lat = _parse_float(_first_non_empty(row, ("lat_hecho", "y", "latitud")))
                lng = _parse_float(_first_non_empty(row, ("long_hecho", "x", "longitud")))
            except ValueError:
                continue
            if not is_valid_coordinate(lat, lng):
                continue
            if deduplicate_ids and identifier:
                identifiers.add(identifier)

            modalidad = normalizar_modalidad(
                _first_non_empty(row, ("modalidad_hecho", "modalidad_he", "modalidad"))
            ) or "NO ESPECIFICADO"
            subtipo = normalizar_modalidad(
                _first_non_empty(row, ("subtipo_hecho", "subtipo_delito", "subtipo"))
            ) or "NO ESPECIFICADO"
            records.append(
                CrimeRecord(
                    lat=lat,
                    lng=lng,
                    turno=normalize_turno(_first_non_empty(row, ("turno_hecho", "turno"))),
                    tipo=normalizar_modalidad(
                        _first_non_empty(row, ("tipo_hecho", "tipo_delito", "tipo"))
                    )
                    or "NO ESPECIFICADO",
                    subtipo=subtipo,
                    modalidad=modalidad,
                    peso_delito=(
                        int(row["peso_delito"])
                        if row.get("peso_delito") not in (None, "")
                        else obtener_peso_desde_campos(modalidad, subtipo)
                    ),
                    distrito=normalizar_modalidad(
                        _first_non_empty(row, ("distrito_hecho", "distrito"))
                    )
                    or "NO ESPECIFICADO",
                    fecha=_normalized_date(row),
                    dia_semana=_day_of_week(row),
                )
            )
    return records
