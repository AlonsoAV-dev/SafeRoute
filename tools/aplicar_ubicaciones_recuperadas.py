"""Crea una copia del Excel con las coordenadas recuperadas por denuncia.

El libro tiene 1,7 millones de filas. La sustitución de celdas XML conserva
las hojas, columnas, valores y estilos originales sin cargarlo entero en RAM.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from app.flujo_entrenamiento_sidpol.fuente import ROW_RE  # noqa: E402


DIRECTORY = ROOT / "outputs" / "delitos-georreferenciados"
SOURCE = ROOT / "outputs" / "delitos-filtrados" / "DELITOS 2018-2026 FILTRADOS Y LIMPIOS.xlsx"
MAPPING = DIRECTORY / "coordenadas_recuperadas_misma_denuncia.csv"
AUDIT = DIRECTORY / "auditoria_recuperacion.json"
OUTPUT = DIRECTORY / "DELITOS 2018-2026 UBICACIONES RECUPERADAS.xlsx"


def load_mapping() -> dict[int, tuple[str, str]]:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    stat = SOURCE.stat()
    if stat.st_size != audit["source_size"] or stat.st_mtime_ns != audit["source_mtime_ns"]:
        raise ValueError("El Excel fuente cambió después de generar las asignaciones")
    mapping = {}
    with MAPPING.open(encoding="utf-8", newline="") as stream:
        for item in csv.DictReader(stream):
            if item["hoja"] != "2025":
                raise ValueError("Este libro requiere revisar una asignación fuera de 2025")
            number = int(item["fila_excel"])
            if number in mapping:
                raise ValueError(f"Fila repetida en asignaciones: {number}")
            mapping[number] = item["xx_recuperada"], item["yy_recuperada"]
    if len(mapping) != audit["totals"]["recuperados"]:
        raise ValueError("El número de asignaciones no coincide con la auditoría")
    return mapping


def fill_cell(row: bytes, column: str, number: int, value: str) -> bytes:
    pattern = re.compile(rb'<c r="' + column.encode() + str(number).encode() + rb'"[^>]*>.*?</c>', re.S)
    found = pattern.search(row)
    if found is None or found.group(0).count(b"<t></t>") != 1:
        raise ValueError(f"La celda {column}{number} no está vacía como se esperaba")
    new_cell = found.group(0).replace(b"<t></t>", b"<t>" + value.encode("ascii") + b"</t>", 1)
    return row[:found.start()] + new_cell + row[found.end():]


def fill_sheet(source, destination, mapping: dict[int, tuple[str, str]]) -> int:
    buffer = b""
    modified = 0
    while chunk := source.read(8 * 1024 * 1024):
        buffer += chunk
        cursor = 0
        for match in ROW_RE.finditer(buffer):
            destination.write(buffer[cursor:match.start()])
            number = int(match.group(1))
            row = match.group(0)
            if number in mapping:
                lat, lon = mapping[number]
                row = fill_cell(row, "J", number, lat)
                row = fill_cell(row, "K", number, lon)
                modified += 1
            destination.write(row)
            cursor = match.end()
        buffer = buffer[cursor:]
    destination.write(buffer)
    if modified != len(mapping):
        raise ValueError(f"Se completaron {modified} de {len(mapping)} filas previstas")
    return modified


def run() -> None:
    mapping = load_mapping()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    temporary = OUTPUT.with_suffix(".tmp.xlsx")
    if temporary.exists():
        raise FileExistsError(temporary)
    modified = 0
    try:
        with zipfile.ZipFile(SOURCE, "r") as original, zipfile.ZipFile(temporary, "w", allowZip64=True) as copy:
            copy.comment = original.comment
            for info in original.infolist():
                with original.open(info, "r") as read, copy.open(info, "w", force_zip64=True) as write:
                    if info.filename == "xl/worksheets/sheet8.xml":
                        modified = fill_sheet(read, write, mapping)
                        print(f"Hoja 2025: {modified:,} ubicaciones copiadas", flush=True)
                    else:
                        shutil.copyfileobj(read, write, length=8 * 1024 * 1024)
        temporary.replace(OUTPUT)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    run()
