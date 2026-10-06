"""Genera una variante experimental con ubicaciones prestadas por distrito.

Estas coordenadas son sintéticas: sirven para estudiar sensibilidad, nunca
para afirmar dónde ocurrió un delito ni para evaluar exactitud espacial real.
El archivo original y la copia recuperada por denuncia permanecen intactos.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from app.flujo_entrenamiento_sidpol.fuente import ROW_RE, _cells, _rows  # noqa: E402


DIRECTORY = ROOT / "outputs" / "delitos-georreferenciados"
SOURCE = ROOT / "outputs" / "delitos-filtrados" / "DELITOS 2018-2026 FILTRADOS Y LIMPIOS.xlsx"
EXACT = DIRECTORY / "coordenadas_recuperadas_misma_denuncia.csv"
SIMULATED = DIRECTORY / "coordenadas_simuladas_anio_distrito.csv"
AUDIT = DIRECTORY / "auditoria_escenario_distrito.json"
OUTPUT = DIRECTORY / "DELITOS 2018-2026 ESCENARIO UBICACIONES SIMULADAS.xlsx"


def iter_data(book: zipfile.ZipFile):
    for number, year in enumerate(range(2018, 2027), 1):
        with book.open(f"xl/worksheets/sheet{number}.xml") as stream:
            columns = None
            for row_no, xml in _rows(stream):
                cells = _cells(xml)
                if row_no == b"1":
                    columns = {value: column for column, value in cells.items()}
                    required = {"id_doc_denuncia", "FECHA_HORA_HECHO", "UBIGEO_HECHO", "ID_TIPO",
                                "ID_SUBTIPO", "TURNO", "xx", "yy"}
                    if not required <= columns.keys():
                        raise ValueError(f"Faltan columnas en {year}: {required - columns.keys()}")
                    continue
                assert columns is not None
                get = lambda name: cells.get(columns[name], "").strip()
                event = get("FECHA_HORA_HECHO")
                yield (year, int(row_no), get("id_doc_denuncia"), get("UBIGEO_HECHO"),
                       event[5:7], get("TURNO"), get("ID_TIPO"), get("ID_SUBTIPO"),
                       get("xx"), get("yy"))


def keys(year, district, month, shift, crime_type, subtype):
    return (
        ("mes_subtipo_turno", year, district, month, subtype, shift),
        ("mes_subtipo", year, district, month, subtype),
        ("mes_tipo_turno", year, district, month, crime_type, shift),
        ("mes_tipo", year, district, month, crime_type),
        ("mes_turno", year, district, month, shift),
        ("mes", year, district, month),
        ("anio_distrito", year, district),
    )


def exact_rows() -> dict[int, dict[int, tuple[str, str]]]:
    result: dict[int, dict[int, tuple[str, str]]] = defaultdict(dict)
    with EXACT.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            year, number = int(row["hoja"]), int(row["fila_excel"])
            result[year][number] = row["xx_recuperada"], row["yy_recuperada"]
    return result


def fill_cell(row: bytes, column: str, number: int, value: str) -> bytes:
    pattern = re.compile(rb'<c r="' + column.encode() + str(number).encode() + rb'"[^>]*>.*?</c>', re.S)
    found = pattern.search(row)
    if found is None or found.group(0).count(b"<t></t>") != 1:
        raise ValueError(f"La celda {column}{number} no está vacía")
    cell = found.group(0).replace(b"<t></t>", b"<t>" + value.encode("ascii") + b"</t>", 1)
    return row[:found.start()] + cell + row[found.end():]


def status_cell(number: int, value: str) -> bytes:
    return (f'<c r="T{number}" t="inlineStr"><is><t>{value}</t></is></c>').encode("ascii")


def write_sheet(source, destination, year, exact, simulated, missing):
    buffer = b""
    updated_dimension = False
    counts = Counter()
    while chunk := source.read(8 * 1024 * 1024):
        buffer += chunk
        if not updated_dimension:
            pattern = re.compile(rb'<dimension ref="A1:S(\d+)"/>')
            if not pattern.search(buffer):
                raise ValueError(f"No se halló dimensión esperada en {year}")
            buffer = pattern.sub(rb'<dimension ref="A1:T\1"/>', buffer, count=1)
            updated_dimension = True
        cursor = 0
        for match in ROW_RE.finditer(buffer):
            destination.write(buffer[cursor:match.start()])
            number = int(match.group(1))
            row = match.group(0)
            if number == 1:
                status = "ORIGEN_UBI"
            elif number in exact:
                lat, lon = exact[number]
                row = fill_cell(fill_cell(row, "J", number, lat), "K", number, lon)
                status = "MISMA_DENUNCIA"
            elif number in simulated:
                lat, lon, _ = simulated[number]
                row = fill_cell(fill_cell(row, "J", number, lat), "K", number, lon)
                status = "SIMULADA"
            elif number in missing:
                status = "SIN_UBI"
            else:
                status = "OBSERVADA"
            row = row.replace(b"</row>", status_cell(number, status) + b"</row>", 1)
            destination.write(row)
            cursor = match.end()
            counts[status] += 1
        buffer = buffer[cursor:]
    destination.write(buffer)
    if counts["MISMA_DENUNCIA"] != len(exact) or counts["SIMULADA"] != len(simulated):
        raise ValueError(f"Conteos incompletos en {year}: {counts}")
    return counts


def run() -> None:
    if OUTPUT.exists() or SIMULATED.exists() or AUDIT.exists():
        raise FileExistsError("Ya existe un resultado del escenario; revíselo antes de ejecutar otra vez")
    exact = exact_rows()
    pools = defaultdict(list)
    missing = []
    missing_by_year = defaultdict(set)
    observed = Counter()
    with zipfile.ZipFile(SOURCE) as book:
        for year, number, report, district, month, shift, crime_type, subtype, lat, lon in iter_data(book):
            if not lat or not lon:
                missing.append((year, number, report, district, month, shift, crime_type, subtype))
                missing_by_year[year].add(number)
            else:
                try:
                    point = float(lat), float(lon)
                except ValueError:
                    continue
                if not (-13.5 <= point[0] <= -10.5 and -78 <= point[1] <= -76):
                    continue
                donor = (year, number, point[0], point[1])
                for key in keys(year, district, month, shift, crime_type, subtype):
                    pools[key].append(donor)
                observed[year] += 1
            if number % 100_000 == 0:
                print(f"Leyendo {year}: fila {number:,}", flush=True)

    simulated: dict[int, dict[int, tuple[str, str, str]]] = defaultdict(dict)
    levels = Counter()
    unresolved = Counter()
    with SIMULATED.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("hoja", "fila_excel", "id_doc_denuncia", "UBIGEO_HECHO", "mes_hecho",
                         "xx_simulada", "yy_simulada", "hoja_donante", "fila_donante", "nivel_coincidencia"))
        for year, number, report, district, month, shift, crime_type, subtype in missing:
            if number in exact.get(year, {}):
                continue
            if not district:
                unresolved[year] += 1
                continue
            for key in keys(year, district, month, shift, crime_type, subtype):
                candidates = pools.get(key)
                if candidates:
                    digest = hashlib.blake2b(f"{year}:{number}:{report}".encode(), digest_size=8).digest()
                    donor = candidates[int.from_bytes(digest, "big") % len(candidates)]
                    lat, lon = f"{donor[2]:.7f}", f"{donor[3]:.7f}"
                    writer.writerow((year, number, report, district, month, lat, lon,
                                     donor[0], donor[1], key[0]))
                    simulated[year][number] = lat, lon, key[0]
                    levels[key[0]] += 1
                    break
            else:
                unresolved[year] += 1

    print(f"Asignaciones simuladas: {sum(levels.values()):,}; sin donante: {sum(unresolved.values()):,}", flush=True)
    temporary = OUTPUT.with_suffix(".tmp.xlsx")
    if temporary.exists():
        raise FileExistsError(temporary)
    sheet_counts = {}
    try:
        with zipfile.ZipFile(SOURCE, "r") as original, zipfile.ZipFile(temporary, "w", allowZip64=True) as copy:
            copy.comment = original.comment
            for info in original.infolist():
                with original.open(info, "r") as read, copy.open(info, "w", force_zip64=True) as write:
                    match = re.fullmatch(r"xl/worksheets/sheet([1-9]).xml", info.filename)
                    if match:
                        year = 2017 + int(match.group(1))
                        sheet_counts[year] = dict(write_sheet(read, write, year, exact.get(year, {}),
                                                              simulated.get(year, {}), missing_by_year[year]))
                        print(f"Escrita hoja {year}", flush=True)
                    else:
                        shutil.copyfileobj(read, write, length=8 * 1024 * 1024)
        temporary.replace(OUTPUT)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    summary = {
        "source": str(SOURCE), "output": str(OUTPUT),
        "description": "Escenario de coordenadas simuladas: no son ubicaciones reales del hecho",
        "matching_order": ["mes+subtipo+turno", "mes+subtipo", "mes+tipo+turno", "mes+tipo",
                           "mes+turno", "mes", "año+distrito"],
        "same_complaint_recovered": sum(map(len, exact.values())),
        "simulated": sum(levels.values()), "without_donor": sum(unresolved.values()),
        "match_levels": dict(levels), "sheet_counts": sheet_counts,
        "warning": "El origen SIMULADA copia la posición de otro delito. No prueba la ubicación del hecho ni valida precisión por tramo.",
    }
    AUDIT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    run()
