from __future__ import annotations

import csv
import html
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from app.flujo_entrenamiento_sidpol.gravedad import CrimeWeights


ROW_RE = re.compile(rb'<row r="(\d+)"[^>]*>.*?</row>', re.S)
CELL_RE = re.compile(rb'<c r="([A-Z]{1,3})\d+"[^>]*>(.*?)</c>', re.S)
TEXT_RE = re.compile(rb'<t(?: [^>]*)?>(.*?)</t>|<v>(.*?)</v>', re.S)
FIELDS = (
    "id_hecho", "fecha_hora_hecho", "fecha", "periodo", "hora",
    "latitud", "longitud", "turno", "tipo_delito", "subtipo_delito",
    "modalidad", "id_subtipo", "id_modalidad", "distrito",
    "peso_delito", "categoria_delito",
)
REQUIRED_HEADERS = {
    "id_doc_denuncia", "FECHA_HORA_HECHO", "DIST_HECHO", "xx", "yy",
    "TIPO", "ID_SUBTIPO", "SUB_TIPO", "ID_MODALIDAD", "MODALIDAD", "TURNO",
}


def _rows(stream):
    buffer = b""
    while chunk := stream.read(8 * 1024 * 1024):
        buffer += chunk
        last = 0
        for match in ROW_RE.finditer(buffer):
            last = match.end()
            yield match.group(1), match.group(0)
        buffer = buffer[last:]


def _cells(row: bytes) -> dict[str, str]:
    result = {}
    for match in CELL_RE.finditer(row):
        value = TEXT_RE.search(match.group(2))
        if value:
            result[match.group(1).decode("ascii")] = html.unescape(
                (value.group(1) or value.group(2) or b"").decode("utf-8", "replace")
            )
    return result


# Normaliza el turno almacenado en la fuente para mantener una representación consistente.
def _turno(value: str) -> str:
    name = unicodedata.normalize("NFKD", value.strip().lower())
    name = "".join(char for char in name if not unicodedata.combining(char))
    if name not in {"madrugada", "manana", "tarde", "noche"}:
        raise ValueError(f"Turno inesperado: {value!r}")
    return name


def _categoria(subtipo: str, modalidad: str) -> str:
    name = unicodedata.normalize("NFKD", f"{subtipo} {modalidad}".upper())
    name = "".join(char for char in name if not unicodedata.combining(char))
    for needle, label in (
        ("HOMICIDIO", "homicidio"),
        ("EXTORSION", "extorsion"),
        ("ROBO", "robo"),
        ("HURTO", "hurto"),
    ):
        if needle in name:
            return label
    return "otro"


# Lee el Excel por bloques, normaliza los registros y guarda la auditoría de cobertura geográfica.
def preparar_fuente(workbook: Path, destination: Path, audit_path: Path, weights_path: Path) -> dict:
    """Convierte las nueve hojas en un CSV espacial y audita también las filas sin GPS."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    monthly = defaultdict(Counter)
    subtype_counts = Counter()
    totals = Counter()
    weights = CrimeWeights(weights_path)
    with zipfile.ZipFile(workbook) as source, destination.open(
        "w", encoding="utf-8", newline=""
    ) as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for sheet_no, year in enumerate(range(2018, 2027), 1):
            with source.open(f"xl/worksheets/sheet{sheet_no}.xml") as stream:
                columns = None
                for row_no, row_xml in _rows(stream):
                    row = _cells(row_xml)
                    if row_no == b"1":
                        columns = {value: column for column, value in row.items()}
                        missing = REQUIRED_HEADERS - columns.keys()
                        if missing:
                            raise ValueError(f"Hoja {year}: columnas ausentes {sorted(missing)}")
                        continue
                    assert columns is not None
                    get = lambda name: row.get(columns[name], "").strip()
                    event = get("FECHA_HORA_HECHO")
                    period = event[:7].replace("/", "-")
                    if not re.fullmatch(r"\d{4}-\d{2}", period) or int(period[:4]) != year:
                        raise ValueError(f"Fecha de hecho inválida en hoja {year}: {event!r}")
                    totals["delitos"] += 1
                    monthly[period]["total"] += 1
                    subtype_counts[get("SUB_TIPO")] += 1
                    lat_raw, lon_raw = get("xx"), get("yy")
                    if not lat_raw or not lon_raw:
                        totals["sin_coordenadas"] += 1
                        continue
                    lat, lon = float(lat_raw), float(lon_raw)
                    if not (-13.5 <= lat <= -10.5 and -78 <= lon <= -76):
                        raise ValueError(f"Coordenadas fuera del rango limpio: {lat}, {lon}")
                    turno = _turno(get("TURNO"))
                    subtipo, modalidad = get("SUB_TIPO"), get("MODALIDAD")
                    writer.writerow({
                        "id_hecho": get("id_doc_denuncia"),
                        "fecha_hora_hecho": event,
                        "fecha": event[:10].replace("/", "-"),
                        "periodo": period,
                        "hora": int(event[11:13]),
                        "latitud": lat,
                        "longitud": lon,
                        "turno": turno,
                        "tipo_delito": get("TIPO"),
                        "subtipo_delito": subtipo,
                        "modalidad": modalidad,
                        "id_subtipo": get("ID_SUBTIPO"),
                        "id_modalidad": get("ID_MODALIDAD"),
                        "distrito": get("DIST_HECHO"),
                        "peso_delito": weights.get(get("ID_SUBTIPO"), get("ID_MODALIDAD"), subtipo, modalidad),
                        "categoria_delito": _categoria(subtipo, modalidad),
                    })
                    totals["geolocalizados"] += 1
                    monthly[period]["geolocalizados"] += 1
            print(f"Fuente {year}: {totals['delitos']:,} delitos acumulados", flush=True)
    audit = {
        "source": str(workbook),
        "weights_file": str(weights_path),
        "normalized": str(destination),
        "totals": dict(totals),
        "monthly": {
            period: {
                "total": counts["total"],
                "geolocalizados": counts["geolocalizados"],
                "cobertura": round(counts["geolocalizados"] / counts["total"], 6),
            }
            for period, counts in sorted(monthly.items())
        },
        "subtypes": dict(subtype_counts),
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit
