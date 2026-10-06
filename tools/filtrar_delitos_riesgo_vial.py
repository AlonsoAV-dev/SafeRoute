"""Filtra la base SIDPOL por delitos relevantes y normaliza coordenadas.

Lee y escribe el XLSX en flujo porque la base completa no cabe cómodamente en
memoria. El resultado conserva únicamente las nueve hojas anuales de datos.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/base-limpia/DELITOS 2018-2026 BASE LIMPIA.xlsx"
OUTDIR = ROOT / "outputs/delitos-filtrados"
DEST = OUTDIR / "DELITOS 2018-2026 FILTRADOS Y LIMPIOS.xlsx"
REPORT = OUTDIR / "auditoria_filtrado.json"

# Códigos de la tabla maestra SIDPOL del 23.04.2026. Un subtipo incluido
# comprende sus modalidades: cada denuncia se conserva una sola vez.
INCLUDE_SUBTYPES = {
    "10101",  # homicidio
    "10103",  # lesiones
    "10401",  # libertad personal: secuestro, coacción, acoso
    "10409",  # libertad sexual
    "10501",  # hurto y sus modalidades
    "10502",  # robo, agravado, tentativas y modalidades de vehículo
    "10508",  # extorsión
    "11201",  # peligro común
    "13002",  # faltas contra personas
    "13003",  # faltas contra patrimonio
    "50801",  # libertad sexual, tipificación especial
    "70101",  # robo agravado
    "70102",  # hurto agravado
    "70104",  # extorsión
    "70704",  # acoso sexual
    "71001",  # secuestro
    "71002",  # coacción
    "71101",  # homicidio
    "71102",  # tentativa de homicidio con lesiones
}
SPECIAL_MODALITIES = {
    "1140109",  # marcaje/reglaje
    "5040124",  # amenaza grave
    "5040125",  # coacción grave
}

ROW_RE = re.compile(rb'<row r="(\d+)"[^>]*>.*?</row>', re.S)
CELL_REF_RE = re.compile(rb'(<c r="[A-Z]{1,3})\d+(?=")')
DIM_RE = re.compile(rb'<dimension ref="A1:S\d+"/>')
FILTER_RE = re.compile(rb'<autoFilter ref="A1:S\d+"/>')
DECIMAL_LOST_RE = re.compile(r"-\d{4,17}")
UNUSABLE_TEXT = {"", "NA", "N/A", "NULL", "NONE", "NAN"}


def cell_value(row: bytes, col: bytes, old_num: bytes) -> str:
    start = row.find(b'<c r="' + col + old_num + b'"')
    if start < 0:
        return ""
    end = row.find(b"</c>", start)
    if end < 0:
        return ""
    fragment = row[start:end]
    for opening, closing in ((b"<t", b"</t>"), (b"<v>", b"</v>")):
        pos = fragment.find(opening)
        if pos >= 0:
            pos = fragment.find(b">", pos) + 1
            stop = fragment.find(closing, pos)
            if stop >= 0:
                return fragment[pos:stop].decode("utf-8", "replace")
    return ""


def replace_cell_value(row: bytes, col: bytes, old_num: bytes, new_value: str) -> bytes:
    start = row.find(b'<c r="' + col + old_num + b'"')
    if start < 0:
        return row
    end = row.find(b"</c>", start)
    if end < 0:
        return row
    end += len(b"</c>")
    fragment = row[start:end]
    t_start = fragment.find(b"<t")
    if t_start < 0:
        raise ValueError(f"Celda {col.decode()}{old_num.decode()} sin texto")
    value_start = fragment.find(b">", t_start) + 1
    value_end = fragment.find(b"</t>", value_start)
    if value_end < 0:
        raise ValueError("Celda sin etiqueta de cierre")
    fragment = fragment[:value_start] + new_value.encode("ascii") + fragment[value_end:]
    return row[:start] + fragment + row[end:]


def coordinate(raw: str) -> tuple[float | None, bool]:
    value = raw.strip().replace(",", ".")
    if value.upper() in UNUSABLE_TEXT:
        return None, False
    try:
        parsed = float(value)
    except ValueError:
        return None, False
    if DECIMAL_LOST_RE.fullmatch(value) and abs(parsed) > 180:
        digits = value[1:]
        return float("-" + digits[:2] + "." + digits[2:]), True
    return parsed, False


def coordinates(lat_raw: str, lon_raw: str) -> tuple[str, str, str]:
    lat, lat_fixed = coordinate(lat_raw)
    lon, lon_fixed = coordinate(lon_raw)
    if lat is None or lon is None:
        return "", "", "missing"
    # Caja amplia de Lima Metropolitana y Callao. Un encaje exacto en la red
    # vial se hará después y no se presume aquí.
    if not (-13.5 <= lat <= -10.5 and -78.0 <= lon <= -76.0):
        return "", "", "outside_lima_callao"
    kind = "decimal_recovered" if (lat_fixed or lon_fixed) else "valid"
    return f"{lat:.7f}", f"{lon:.7f}", kind


def valid_event_datetime(raw: str, year: int) -> bool:
    if raw.strip().upper() in UNUSABLE_TEXT:
        return False
    try:
        event = datetime.fromisoformat(raw.strip().replace("/", "-"))
    except ValueError:
        return False
    return event.year == year


def renumber(row: bytes, new_num: int) -> bytes:
    number = str(new_num).encode("ascii")
    row = re.sub(rb'^<row r="\d+"', b'<row r="' + number + b'"', row, count=1)
    return CELL_REF_RE.sub(lambda m: m.group(1) + number, row)


def sheet_parts(source, year: int, row_temp: Path, seen: set[bytes]) -> tuple[bytes, bytes, Counter]:
    """Transforma las filas de una hoja y devuelve cabecera, cierre y conteos."""
    buffer = b""
    prefix = None
    counts = Counter()
    new_num = 0
    with row_temp.open("wb") as dst:
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if chunk:
                buffer += chunk
            if prefix is None:
                first = buffer.find(b'<row r="')
                if first < 0:
                    if not chunk:
                        raise ValueError(f"Hoja {year} sin filas")
                    continue
                prefix, buffer = buffer[:first], buffer[first:]
            last = 0
            for match in ROW_RE.finditer(buffer):
                old_num = match.group(1)
                row = match.group(0)
                last = match.end()
                if old_num == b"1":
                    new_num = 1
                    dst.write(row)
                    continue
                counts["source_rows"] += 1
                subtype = cell_value(row, b"N", old_num).strip()
                modality = cell_value(row, b"P", old_num).strip()
                if subtype not in INCLUDE_SUBTYPES and modality not in SPECIAL_MODALITIES:
                    counts["excluded_other_typification"] += 1
                    continue
                counts["selected_crimes"] += 1
                record_id = cell_value(row, b"A", old_num).strip()
                event_date = cell_value(row, b"C", old_num).strip()
                if not record_id or not valid_event_datetime(event_date, year):
                    counts["excluded_invalid_id_or_event_date"] += 1
                    continue
                lat_raw = cell_value(row, b"J", old_num)
                lon_raw = cell_value(row, b"K", old_num)
                lat, lon, geo_status = coordinates(lat_raw, lon_raw)
                counts[f"geo_{geo_status}"] += 1
                reg_date = cell_value(row, b"B", old_num).strip()
                key = "|".join((record_id, reg_date, event_date, lat_raw, lon_raw, subtype, modality))
                digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
                if digest in seen:
                    counts["excluded_duplicate"] += 1
                    continue
                seen.add(digest)
                row = replace_cell_value(row, b"J", old_num, lat)
                row = replace_cell_value(row, b"K", old_num, lon)
                new_num += 1
                dst.write(renumber(row, new_num))
                counts["output_rows"] += 1
            buffer = buffer[last:]
            if not chunk:
                break
    if prefix is None or not buffer.startswith(b"</sheetData>"):
        raise ValueError(f"Estructura inesperada en hoja {year}")
    if counts["source_rows"] != counts["selected_crimes"] + counts["excluded_other_typification"]:
        raise AssertionError(f"Conteos de selección inconsistentes en {year}")
    if counts["selected_crimes"] != counts["output_rows"] + counts["excluded_invalid_id_or_event_date"] + counts["excluded_duplicate"]:
        raise AssertionError(f"Conteos de salida inconsistentes en {year}")
    counts["excel_last_row"] = new_num
    return prefix, buffer, counts


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    output_partial = OUTDIR / (DEST.name + ".partial")
    report_partial = OUTDIR / (REPORT.name + ".partial")
    seen: set[bytes] = set()
    per_year = {}
    with zipfile.ZipFile(SOURCE, "r") as src, zipfile.ZipFile(output_partial, "w") as dst:
        for info in src.infolist():
            year_match = re.fullmatch(r"xl/worksheets/sheet([1-9]).xml", info.filename)
            copied_info = copy.copy(info)
            copied_info.compress_type = zipfile.ZIP_DEFLATED
            copied_info._compresslevel = 4
            if year_match:
                year = 2017 + int(year_match.group(1))
                with tempfile.TemporaryDirectory(dir=OUTDIR) as temp_dir:
                    row_temp = Path(temp_dir) / "rows.xml"
                    with src.open(info) as stream:
                        prefix, suffix, counts = sheet_parts(stream, year, row_temp, seen)
                    last_row = counts["excel_last_row"]
                    prefix, n_dim = DIM_RE.subn(f'<dimension ref="A1:S{last_row}"/>'.encode(), prefix)
                    suffix, n_filter = FILTER_RE.subn(f'<autoFilter ref="A1:S{last_row}"/>'.encode(), suffix)
                    if n_dim != 1 or n_filter != 1:
                        raise ValueError(f"Dimensión o filtro ausente en hoja {year}")
                    with dst.open(copied_info, "w", force_zip64=True) as out:
                        out.write(prefix)
                        with row_temp.open("rb") as rows_file:
                            shutil.copyfileobj(rows_file, out, 4 * 1024 * 1024)
                        out.write(suffix)
                per_year[str(year)] = dict(counts)
                print(f"{year}: {counts['selected_crimes']:,} delitos, {counts['output_rows']:,} conservados, "
                      f"{counts['geo_missing'] + counts['geo_outside_lima_callao']:,} sin coordenadas válidas", flush=True)
            elif info.filename == "xl/workbook.xml":
                workbook = src.read(info)
                for year_str, counts in per_year.items():
                    pattern = (rf"('{year_str}'!\$A\$1:\$S\$)\d+").encode()
                    workbook, n = re.subn(pattern, lambda m: m.group(1) + str(counts["excel_last_row"]).encode(), workbook)
                    if n != 1:
                        raise ValueError(f"Nombre de filtro ausente en workbook para {year_str}")
                dst.writestr(copied_info, workbook)
            else:
                with src.open(info) as inp, dst.open(copied_info, "w", force_zip64=True) as out:
                    shutil.copyfileobj(inp, out, 4 * 1024 * 1024)

    total = Counter()
    for counts in per_year.values():
        total.update(counts)
    report = {
        "source": str(SOURCE),
        "output": str(DEST),
        "master_typification": str(ROOT / "SIDPOL_TABLAS_MAESTRAS_TIPIFICACCION_23.04.2026.xlsx"),
        "criteria": {
            "included_subtypes": sorted(INCLUDE_SUBTYPES),
            "included_modalities_outside_subtypes": sorted(SPECIAL_MODALITIES),
            "event_date_must_match_annual_sheet": True,
            "coordinates_kept_inside_lima_callao_envelope": {"latitude": [-13.5, -10.5], "longitude": [-78.0, -76.0]},
            "invalid_coordinates": "leave latitude and longitude blank; retain the crime record",
            "lost_decimal": "reinsert decimal after two latitude/longitude digits only when original is an unambiguous large negative integer and pair falls inside envelope",
            "turn": "preserve corrected source TURNO based on FECHA_HORA_HECHO",
        },
        "per_year": per_year,
        "totals": dict(total),
    }
    report_partial.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(output_partial, DEST)
    os.replace(report_partial, REPORT)
    print(f"TOTAL: {total['selected_crimes']:,} delitos seleccionados; {total['output_rows']:,} en archivo; "
          f"{total['geo_valid'] + total['geo_decimal_recovered']:,} con coordenadas utilizables", flush=True)
    print(DEST, flush=True)


if __name__ == "__main__":
    main()
