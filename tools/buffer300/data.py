from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from comparar_factores_67 import NAMES, NEW, OLD, OUT as PREVIOUS, graph_data
from app.flujo_entrenamiento.riesgo import agregar_riesgo_base

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Backend/data/experimentos_sidpol/buffer_300_v1"
OUT = ROOT / "outputs/evaluacion-modelos-sidpol/optimizacion_buffer_300_v1"
PERIODS = tuple(str(p) for p in pd.period_range("2018-01", "2026-08", freq="M"))
THRESHOLD = 2.3442396250791653
SOURCE = NEW / "delitos_geolocalizados.csv"
CACHE = DATA / "panel_mensual_300m.npy"
META = DATA / "panel_mensual_300m.json"


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def labels(risk):
    return np.where(risk >= THRESHOLD, 2, np.where(risk > 0, 1, 0)).astype(np.int8)


def fingerprint():
    stat = SOURCE.stat()
    rules = ROOT / "Backend/app/services/pesos_delito.py"
    return {"source": str(SOURCE), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "weights_sha256": hashlib.sha256(rules.read_bytes()).hexdigest(),
            "radius_m": 300, "sigma_m": 100, "periods": list(PERIODS)}


def load_static():
    frame = pd.read_csv(OLD / "tramos_osm.csv", usecols=["tramo_id", "latitud", "longitud", "longitud_m"])
    static = frame[["latitud", "longitud", "longitud_m"]].to_numpy(dtype=np.float32)
    return static, np.maximum(static[:, 2] / 100.0, 1.0)


def audit_source(crimes):
    rows = []
    sim_path = ROOT / "outputs/delitos-georreferenciados/coordenadas_simuladas_anio_distrito.csv"
    sim = pd.read_csv(sim_path, dtype={"id_doc_denuncia": str, "hoja": str, "mes_hecho": str})
    sim_keys = set(zip(sim.id_doc_denuncia, sim.hoja + "-" + sim.mes_hecho.str.zfill(2)))
    for period, d in crimes.groupby("periodo", sort=True):
        simulated = sum((str(v), period) in sim_keys for v in d.id_hecho)
        rows.append({"periodo": period, "delitos": len(d),
                     "coordenadas_distintas": len(d[["latitud", "longitud"]].drop_duplicates()),
                     "coinciden_manifest_simuladas": simulated,
                     "peso_medio_reglas_ensayo": float(d.peso_delito.mean()),
                     "distritos": int(d.distrito.nunique()),
                     "max_fecha": str(d.fecha.max())})
    pd.DataFrame(rows).to_csv(OUT / "auditoria_fuente_mensual.csv", index=False)
    save_json(OUT / "auditoria_fuente.json", {
        "source": str(SOURCE), "records": len(crimes), "monthly": rows,
        "coordinate_origin": "La base conserva coordenadas observadas y copiadas; se identifican por id y periodo contra el manifiesto de simulación.",
        "interpretation": "La cobertura espacial cambia bruscamente al final de 2025. No se alteran ni eliminan registros para mejorar métricas."})


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    signature = fingerprint()
    if CACHE.exists() and META.exists():
        meta = json.loads(META.read_text(encoding="utf-8"))
        if meta["fingerprint"] != signature:
            raise ValueError("La fuente o las reglas cambiaron; use otra versión del experimento")
        print("Panel completo existente y compatible", flush=True)
        return
    columns = ["id_hecho", "fecha", "periodo", "latitud", "longitud", "turno", "subtipo_delito", "modalidad", "distrito"]
    crimes = pd.read_csv(SOURCE, usecols=columns, dtype={"id_hecho": str})
    crimes = agregar_riesgo_base(crimes)
    audit_source(crimes)
    tramos, lines, transformer = graph_data()
    n = len(tramos)
    del tramos
    xy = transformer.transform(crimes.longitud.to_numpy(), crimes.latitud.to_numpy())
    points = shapely.points(*xy)
    tree = STRtree(lines)
    weights = crimes.peso_delito.to_numpy(dtype=np.float32)
    grave = crimes.es_delito_grave.to_numpy(dtype=np.float32)
    categories = crimes.categoria_delito.to_numpy()
    turns = crimes.turno.to_numpy()
    period_ids = pd.Categorical(crimes.periodo, categories=PERIODS).codes
    old = np.load(PREVIOUS / "buffer_nuevo_2025_2026_300m.npy", mmap_mode="r")
    panel = np.lib.format.open_memmap(CACHE, mode="w+", dtype=np.float32,
                                     shape=(len(PERIODS), n, len(NAMES)))
    spatial = []
    for month, period in enumerate(PERIODS):
        indexes = np.flatnonzero(period_ids == month)
        if "2025-01" <= period <= "2026-05":
            panel[month] = old[month - PERIODS.index("2025-01")]
            pairs = int(panel[month, :, 0].sum())
        else:
            local, segments = tree.query(points[indexes], predicate="dwithin", distance=300.0)
            events = indexes[local]
            distances = shapely.distance(points[events], lines[segments])
            decay = np.exp(-0.5 * (distances / 100.0) ** 2)
            panel[month, :, 0] = np.bincount(segments, minlength=n)
            panel[month, :, 1] = np.bincount(segments, weights=weights[events] * decay, minlength=n)
            panel[month, :, 2] = np.bincount(segments, weights=grave[events], minlength=n)
            for col, category in enumerate(("hurtos", "robos", "extorsiones", "homicidios"), 3):
                panel[month, :, col] = np.bincount(segments, weights=categories[events] == category, minlength=n)
            for col, turn in enumerate(("manana", "tarde", "noche", "madrugada"), 7):
                panel[month, :, col] = np.bincount(segments, weights=turns[events] == turn, minlength=n)
            pairs = len(segments)
            del local, segments, events, distances, decay
        spatial.append({"periodo": period, "delitos": len(indexes), "pares": pairs,
                        "tramos_con_senal": int(np.count_nonzero(panel[month, :, 0]))})
        print(f"PANEL {period}: {len(indexes):,} delitos, {pairs:,} pares", flush=True)
        if month % 6 == 0:
            panel.flush()
            gc.collect()
    panel.flush()
    # La evaluación anterior debe tener exactamente las mismas etiquetas y variables base.
    start = PERIODS.index("2025-01")
    if not np.array_equal(panel[start:start + len(old)], old):
        raise AssertionError("Cambió el panel del ensayo original")
    save_json(META, {"fingerprint": signature, "shape": list(panel.shape), "monthly": spatial,
                     "baseline_equal": True, "threshold": THRESHOLD})
    pd.DataFrame(spatial).to_csv(OUT / "auditoria_buffer_mensual.csv", index=False)
    print("PREPARACION COMPLETA", flush=True)


if __name__ == "__main__":
    prepare()
