"""Recupera coordenadas solo desde otra fila del mismo hecho denunciado.

La salida es un archivo de asignaciones por hoja y número de fila. No modifica
el Excel fuente ni asigna ubicaciones por distrito, delito o popularidad.
"""

from __future__ import annotations

import csv
from datetime import datetime
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from app.flujo_entrenamiento_sidpol.fuente import _cells, _rows  # noqa: E402


SOURCE = ROOT / "outputs" / "delitos-filtrados" / "DELITOS 2018-2026 FILTRADOS Y LIMPIOS.xlsx"
OUTPUT = ROOT / "outputs" / "delitos-georreferenciados" / "coordenadas_recuperadas_misma_denuncia.csv"
REPORT = OUTPUT.with_name("auditoria_recuperacion.json")
MAX_SPREAD_M = 20.0
MAX_HOURS = 6.0


def rows(workbook: zipfile.ZipFile):
    for sheet_number, year in enumerate(range(2018, 2027), 1):
        with workbook.open(f"xl/worksheets/sheet{sheet_number}.xml") as stream:
            names = None
            for number, xml in _rows(stream):
                values = _cells(xml)
                if number == b"1":
                    names = {value: column for column, value in values.items()}
                    required = {"id_doc_denuncia", "FECHA_HORA_HECHO", "UBIGEO_HECHO", "xx", "yy"}
                    if not required <= names.keys():
                        raise ValueError(f"Faltan columnas en hoja {year}: {required - names.keys()}")
                    continue
                assert names is not None
                get = lambda name: values.get(names[name], "").strip()
                key = (get("id_doc_denuncia"), get("UBIGEO_HECHO"))
                yield year, int(number), key, get("FECHA_HORA_HECHO"), get("xx"), get("yy")


def separation_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    latitude = math.radians((a[0] + b[0]) / 2)
    return 111_195 * math.hypot(a[0] - b[0], (a[1] - b[1]) * math.cos(latitude))


def run() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    source_stat = SOURCE.stat()
    missing: list[tuple[int, int, tuple[str, str], str]] = []
    missing_keys: set[tuple[str, str]] = set()
    by_year: dict[int, Counter] = defaultdict(Counter)
    with zipfile.ZipFile(SOURCE) as workbook:
        for year, number, key, event, lat, lon in rows(workbook):
            by_year[year]["total"] += 1
            if not lat or not lon:
                by_year[year]["missing"] += 1
                missing.append((year, number, key, event))
                if all(key):
                    missing_keys.add(key)
            if number % 100_000 == 0:
                print(f"Buscando faltantes {year}: fila {number:,}", flush=True)

        print(f"Faltantes: {len(missing):,}; claves: {len(missing_keys):,}", flush=True)
        donors: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for year, number, key, event, lat_raw, lon_raw in rows(workbook):
            if key not in missing_keys or not lat_raw or not lon_raw:
                continue
            try:
                point = float(lat_raw), float(lon_raw)
            except ValueError:
                continue
            if not (-13.5 <= point[0] <= -10.5 and -78 <= point[1] <= -76):
                continue
            donors[key].append({"point": point, "year": year, "row": number, "event": event})
            if number % 100_000 == 0:
                print(f"Buscando donantes {year}: fila {number:,}", flush=True)

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("hoja", "fila_excel", "id_doc_denuncia", "UBIGEO_HECHO", "FECHA_HORA_HECHO",
                         "xx_recuperada", "yy_recuperada", "hoja_donante", "fila_donante",
                         "donantes", "dispersion_m", "metodo"))
        for year, number, key, event in missing:
            candidates = donors.get(key, [])
            if not candidates:
                by_year[year]["sin_donante"] += 1
                continue
            event_time = datetime.strptime(event, "%Y/%m/%d %H:%M:%S")
            candidates = [(candidate, abs((datetime.strptime(candidate["event"], "%Y/%m/%d %H:%M:%S")
                                           - event_time).total_seconds()) / 3600)
                          for candidate in candidates]
            candidates = [(candidate, hours) for candidate, hours in candidates if hours <= MAX_HOURS]
            if not candidates:
                by_year[year]["donante_fuera_de_6h"] += 1
                continue
            donor, delta_hours = min(candidates, key=lambda item: item[1])
            spread = max(separation_m(donor["point"], candidate["point"]) for candidate, _ in candidates)
            if spread > MAX_SPREAD_M:
                by_year[year]["ambiguos"] += 1
                continue
            writer.writerow((year, number, *key, event, f"{donor['point'][0]:.7f}", f"{donor['point'][1]:.7f}",
                             donor["year"], donor["row"], len(candidates),
                             f"{spread:.2f}", "misma_denuncia_distrito_6h"))
            by_year[year]["recuperados"] += 1

    report = {
        "source": str(SOURCE), "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "assignment_file": str(OUTPUT), "max_donor_spread_m": MAX_SPREAD_M,
        "max_time_difference_hours": MAX_HOURS,
        "key": ["id_doc_denuncia", "UBIGEO_HECHO"],
        "scope": "Solo filas sin coordenadas con otro registro de la misma denuncia y distrito a menos de seis horas",
        "by_year": {str(year): dict(by_year[year]) for year in sorted(by_year)},
        "totals": {name: sum(counts[name] for counts in by_year.values())
                   for name in ("total", "missing", "recuperados", "ambiguos", "sin_donante", "donante_fuera_de_6h")},
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["totals"], ensure_ascii=False), flush=True)
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    run()
