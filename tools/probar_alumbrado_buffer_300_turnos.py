"""Comparación controlada de alumbrado fijo por turno con buffer de 300 m.

Cada delito afecta a los tramos a 300 m o menos con el mismo decaimiento y peso
del experimento mensual. La etiqueta usa la gravedad de *ese turno*, y los dos
brazos difieren solo en cuatro atributos fijos de alumbrado.
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from comparar_factores_67 import graph_data
from app.flujo_entrenamiento.riesgo import agregar_riesgo_base
from buffer300.data import CACHE as MONTHLY, DATA as MONTHLY_DIR, PERIODS, SOURCE, THRESHOLD, save_json
from buffer300.modelos import evaluate, fit_model, matrix_metrics, predictions
from buffer300.features import Features
from buffer300.report import pct, render_matrix

DATA = ROOT / "Backend/data/experimentos_sidpol/buffer_300_turnos_luz_v1"
OUT = ROOT / "outputs/evaluacion-modelos-sidpol/alumbrado_buffer_300_turnos_v1"
PANEL = DATA / "panel_turnos_300m_2025_2026.npy"
META = DATA / "panel_turnos_300m_2025_2026.json"
LIGHT = MONTHLY_DIR / "alumbrado_estatico_v1.npy"
START = PERIODS.index("2025-01")
TURNS = ("madrugada", "manana", "tarde", "noche")
TURN_COLS = (10, 7, 8, 9)
VALIDATION = ("2025-11", "2025-12")
BENCHMARK = ("2026-01", "2026-02", "2026-03", "2026-04", "2026-05")
ADDITIONAL = ("2026-06", "2026-07", "2026-08")
SPEC = {"name": "contexto_3m_por_turno", "features": "contexto_turno", "balance": 1.25,
        "window": 3, "per_class_month_turn": 4_500, "seed": 42}


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not LIGHT.is_file():
        raise FileNotFoundError("Primero prepare el inventario fijo de alumbrado")
    if PANEL.exists() and META.exists():
        previous = json.loads(META.read_text(encoding="utf-8"))
        stat = SOURCE.stat()
        if previous["source_size"] != stat.st_size or previous["source_mtime_ns"] != stat.st_mtime_ns:
            raise ValueError("Cambió la fuente delictiva; cree otra versión del experimento")
        print("Panel de turnos 300 m ya preparado", flush=True)
        return
    fields = ["fecha", "periodo", "latitud", "longitud", "turno", "subtipo_delito", "modalidad"]
    chunks = []
    for part in pd.read_csv(SOURCE, usecols=fields, chunksize=125_000):
        chosen = part[part["periodo"].between("2025-01", "2026-08")]
        if len(chosen):
            chunks.append(chosen)
    crimes = agregar_riesgo_base(pd.concat(chunks, ignore_index=True))
    if not crimes["turno"].isin(TURNS).all():
        raise ValueError("Hay turnos desconocidos en el origen")
    tramos, lines, transformer = graph_data()
    n = len(tramos)
    del tramos
    points = shapely.points(*transformer.transform(crimes.longitud.to_numpy(), crimes.latitud.to_numpy()))
    tree = STRtree(lines)
    months = PERIODS[START:PERIODS.index("2026-08")+1]
    period_code = pd.Categorical(crimes.periodo, categories=months).codes
    turn_code = pd.Categorical(crimes.turno, categories=TURNS).codes
    weights = crimes.peso_delito.to_numpy(dtype=np.float32)
    existing = np.load(MONTHLY, mmap_mode="r")
    panel = np.lib.format.open_memmap(PANEL, mode="w+", dtype=np.float32,
                                     shape=(len(months), 4, n, 2))
    audit = []
    for j, period in enumerate(months):
        event_ids = np.flatnonzero(period_code == j)
        local, segment = tree.query(points[event_ids], predicate="dwithin", distance=300.0)
        events = event_ids[local]
        distance = shapely.distance(points[events], lines[segment])
        weighted = weights[events] * np.exp(-0.5*(distance/100.0)**2)
        for turn in range(4):
            panel[j, turn, :, 0] = existing[START+j, :, TURN_COLS[turn]]
            selected = turn_code[events] == turn
            panel[j, turn, :, 1] = np.bincount(segment[selected], weights=weighted[selected], minlength=n)
        count_delta = float(np.max(np.abs(panel[j, :, :, 0].sum(axis=0)-existing[START+j, :, 0])))
        weight_delta = float(np.max(np.abs(panel[j, :, :, 1].sum(axis=0)-existing[START+j, :, 1])))
        if count_delta != 0 or weight_delta > .001:
            raise AssertionError(f"El agregado por turnos no reproduce el buffer mensual {period}: {count_delta}, {weight_delta}")
        audit.append({"periodo": period, "delitos": len(event_ids), "pares_delito_tramo": len(segment),
                      "max_diferencia_gravedad": weight_delta})
        print(f"PANEL 300 m {period}: {len(event_ids):,} delitos, {len(segment):,} pares", flush=True)
        if j % 4 == 0:
            panel.flush()
    panel.flush()
    stat = SOURCE.stat()
    save_json(META, {"source": str(SOURCE), "source_size": stat.st_size,
              "source_mtime_ns": stat.st_mtime_ns, "months": months, "turns": TURNS,
              "shape": list(panel.shape), "radius_m": 300, "sigma_m": 100,
              "high_threshold": THRESHOLD, "monthly_buffer_reproduced": True,
              "monthly_audit": audit})
    print("PANEL DE TURNOS 300 m COMPLETO", flush=True)


class TurnFeatures:
    def __init__(self):
        self.monthly = Features()
        self.panel = np.load(PANEL, mmap_mode="r")
        self.factor = self.monthly.factor
        self.n = self.monthly.n
        self.light = np.load(LIGHT, mmap_mode="r")
        if self.panel.shape[2] != self.n or self.light.shape != (self.n, 4):
            raise AssertionError("Los tramos del panel mensual, por turno y de alumbrado no coinciden")

    @functools.lru_cache(maxsize=4)
    def monthly_base(self, month):
        return self.monthly.make(month, "contexto")

    def risk(self, month, turn):
        return self.panel[month-START, turn, :, 1] / self.factor

    def target(self, month, turn):
        risk = self.risk(month, turn)
        return np.where(risk >= THRESHOLD, 2, np.where(risk > 0, 1, 0)).astype(np.int8)

    @functools.lru_cache(maxsize=12)
    def make(self, month, turn):
        if month < START+3 or month > PERIODS.index("2026-08"):
            raise ValueError("El mes no tiene tres meses previos en el panel por turno")
        base = self.monthly_base(month)
        previous = [self.risk(month-lag, turn) for lag in (1, 2, 3)]
        last3 = np.stack(previous)
        count3 = self.panel[month-START-3:month-START, turn, :, 0]
        total3 = self.monthly.risk[month-3:month].mean(axis=0)
        neighbors = self.monthly._neighbors()
        extra = np.column_stack([
            np.full(self.n, float(turn == k), dtype=np.float32) for k in range(4)
        ] + previous + [
            last3.mean(axis=0), last3.std(axis=0), last3.max(axis=0),
            (last3 > 0).mean(axis=0), (last3 >= THRESHOLD).mean(axis=0),
            count3.mean(axis=0)/self.factor,
            last3.mean(axis=0)/(total3+.05),
            previous[0][neighbors[:, :8]].mean(axis=1),
            last3.mean(axis=0)[neighbors[:, :8]].mean(axis=1),
        ])
        return np.column_stack([base, extra]).astype(np.float32)


def training_data(f, month):
    months = range(month-3, month)
    rng = np.random.default_rng(SPEC["seed"])
    entries, class_counts = [], np.zeros(3, dtype=np.int64)
    for prior in months:
        for turn in range(4):
            y = f.target(prior, turn)
            class_counts += np.bincount(y, minlength=3)
            for label in range(3):
                available = np.flatnonzero(y == label)
                if not len(available):
                    continue
                keep = min(len(available), SPEC["per_class_month_turn"])
                selected = rng.choice(available, size=keep, replace=False)
                entries.append((prior, turn, selected, y[selected], np.full(keep, len(available)/keep, dtype=np.float32)))
    size = sum(len(entry[2]) for entry in entries)
    width = f.make(month-3, 0).shape[1]
    x = np.empty((size, width), dtype=np.float32)
    light = np.empty((size, 4), dtype=np.float32)
    labels = np.empty(size, dtype=np.int8)
    ratios = np.empty(size, dtype=np.float32)
    offset = 0
    for prior, turn, selected, selected_y, ratio in entries:
        end = offset+len(selected)
        x[offset:end] = f.make(prior, turn)[selected]
        light[offset:end] = f.light[selected]
        labels[offset:end] = selected_y
        ratios[offset:end] = ratio
        offset = end
    weight = ratios * (class_counts.sum()/(3*class_counts[labels]))**SPEC["balance"]
    weight = (weight/weight.mean()).astype(np.float32)
    return x, light, labels, weight, {"train_start": PERIODS[month-3],
                                    "train_end": PERIODS[month-1], "train_rows": size,
                                    "actual_class_counts": class_counts.tolist()}


def evaluate_month(f, month):
    x, light, y, weights, training = training_data(f, month)
    a = fit_model(SPEC, x, y, weights)
    b = fit_model(SPEC, np.column_stack([x, light]), y, weights)
    del x, light, y, weights
    gc.collect()
    matrices = {"sin_alumbrado": np.zeros((3, 3), dtype=np.int64),
                "con_alumbrado": np.zeros((3, 3), dtype=np.int64)}
    by_turn = {}
    for turn, label in enumerate(TURNS):
        features = f.make(month, turn)
        truth = f.target(month, turn)
        ma = np.asarray(evaluate(truth, predictions(a, features))["matriz_confusion"], dtype=np.int64)
        mb = np.asarray(evaluate(truth, predictions(b, np.column_stack([features, f.light])))["matriz_confusion"], dtype=np.int64)
        matrices["sin_alumbrado"] += ma
        matrices["con_alumbrado"] += mb
        by_turn[label] = {"sin_alumbrado": matrix_metrics(ma), "con_alumbrado": matrix_metrics(mb)}
        del features, truth
    return {"periodo": PERIODS[month], "training": training,
            "sin_alumbrado": matrix_metrics(matrices["sin_alumbrado"]),
            "con_alumbrado": matrix_metrics(matrices["con_alumbrado"]), "por_turno": by_turn}


def aggregate(records, periods, arm):
    return matrix_metrics(np.sum([np.asarray(records[p][arm]["matriz_confusion"]) for p in periods], axis=0))


def aggregate_turn(records, periods, turn, arm):
    return matrix_metrics(np.sum([np.asarray(records[p]["por_turno"][turn][arm]["matriz_confusion"]) for p in periods], axis=0))


def run():
    if not PANEL.exists():
        raise FileNotFoundError("Ejecute primero prepare")
    OUT.mkdir(parents=True, exist_ok=True)
    f = TurnFeatures()
    f.monthly.verify_causality()
    records = {}
    for period in VALIDATION+BENCHMARK+ADDITIONAL:
        path = OUT / f"resultado_{period}.json"
        if path.exists():
            records[period] = json.loads(path.read_text(encoding="utf-8"))
            continue
        row = evaluate_month(f, PERIODS.index(period))
        records[period] = row
        save_json(path, row)
        print(f"PRUEBA 300 m {period}: acc {row['sin_alumbrado']['accuracy']:.4f}→{row['con_alumbrado']['accuracy']:.4f}, "
              f"F1MA {row['sin_alumbrado']['f1_medio_alto']:.4f}→{row['con_alumbrado']['f1_medio_alto']:.4f}", flush=True)
    groups = {}
    for label, periods in (("validacion", VALIDATION), ("enero_mayo", BENCHMARK), ("junio_agosto", ADDITIONAL)):
        groups[label] = {"periodos": periods}
        for arm in ("sin_alumbrado", "con_alumbrado"):
            groups[label][arm] = aggregate(records, periods, arm)
            groups[label][f"{arm}_por_turno"] = {turn: aggregate_turn(records, periods, turn, arm) for turn in TURNS}
    save_json(OUT / "final_results.json", {"protocol": {"source": str(SOURCE),
              "spatial_method": "buffer 300 m; sigma 100 m; mismo peso y umbral que el experimento mensual",
              "unit": "tramo × mes × turno", "threshold_high": THRESHOLD,
              "static_lighting_2018_2026": True, "train_window_months": 3,
              "train_sampling": SPEC, "validation": VALIDATION,
              "test": BENCHMARK+ADDITIONAL, "test_used_to_select": False,
              "note": "2026 es retrospectivo; las ubicaciones completadas se tratan como las demás."},
              "groups": groups})
    print("EVALUACIÓN 300 m POR TURNOS COMPLETA", flush=True)


def report():
    data = json.loads((OUT / "final_results.json").read_text(encoding="utf-8"))
    fields = [("Accuracy", "accuracy", None), ("Precisión medio", "precision", 1),
              ("Detección medio", "recall", 1), ("F1 medio", "f1", 1),
              ("Precisión alto", "precision", 2), ("Detección alto", "recall", 2),
              ("F1 alto", "f1", 2), ("F1 medio+alto", "f1_medio_alto", None)]
    lines = ["# Alumbrado fijo por turno, buffer 300 m", "",
             "El ensayo mantiene el **buffer de 300 m** del modelo mensual, el decaimiento, la red, los pesos de delito "
             "y el umbral alto. La nueva etiqueta corresponde al riesgo de cada tramo, mes y turno. "
             "Los dos brazos difieren solo en cuatro variables estáticas de luminarias próximas.", "",
             "Los modelos se reentrenan cada mes con los tres meses anteriores. La validación usa noviembre–diciembre "
             "de 2025; la evaluación usa enero–agosto de 2026. Ningún mes objetivo entra en el ajuste que lo predice.", ""]
    for group, title in (("validacion", "Validación noviembre–diciembre de 2025"),
                         ("enero_mayo", "Evaluación enero–mayo de 2026"),
                         ("junio_agosto", "Evaluación junio–agosto de 2026")):
        a, b = data["groups"][group]["sin_alumbrado"], data["groups"][group]["con_alumbrado"]
        lines += [f"## {title}", "", "| Métrica | Sin alumbrado | Con alumbrado | Cambio (puntos) |",
                  "|---|---:|---:|---:|"]
        for title_field, key, index in fields:
            av = a[key] if index is None else a[key][index]
            bv = b[key] if index is None else b[key][index]
            lines.append(f"| {title_field} | {pct(av)} | {pct(bv)} | {(bv-av)*100:+.2f} |")
        lines += [""]
    for group, title in (("enero_mayo", "Enero–mayo de 2026 por turno"),
                         ("junio_agosto", "Junio–agosto de 2026 por turno")):
        lines += [f"## {title}", "", "| Turno | F1 medio+alto sin | F1 medio+alto con | Detección alto sin | Detección alto con | Precisión alto sin | Precisión alto con |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for turn in TURNS:
            a = data["groups"][group]["sin_alumbrado_por_turno"][turn]
            b = data["groups"][group]["con_alumbrado_por_turno"][turn]
            lines.append(f"| {turn} | {pct(a['f1_medio_alto'])} | {pct(b['f1_medio_alto'])} | "
                         f"{pct(a['recall'][2])} | {pct(b['recall'][2])} | {pct(a['precision'][2])} | {pct(b['precision'][2])} |")
        lines += [""]
    lines += ["## Conclusión", "",
              "Con buffer de 300 m, agregar alumbrado fijo no produjo una mejora equilibrada. En enero–mayo "
              "de 2026 el accuracy y el F1 medio+alto bajan; la detección de alto sube ligeramente "
              "mientras su precisión baja. El F1 medio+alto se reduce en cada uno de los cuatro turnos. "
              "Junio–agosto presenta el mismo patrón agregado. Se conserva la versión sin alumbrado "
              "como referencia de este objetivo por turno.", "",
              "El alumbrado se asume fijo para todo el historial. El archivo no informa si las luminarias funcionaban "
              "o cuánta luz daban. La evaluación de 2026 es retrospectiva y ya se consultó en otros ensayos. "
              "Estas métricas por turno no son comparables directamente con el 67,33 % mensual porque se divide "
              "cada tramo-mes en cuatro objetivos distintos.", ""]
    (OUT / "informe.md").write_text("\n".join(lines), encoding="utf-8")
    a = data["groups"]["enero_mayo"]["sin_alumbrado"]
    b = data["groups"]["enero_mayo"]["con_alumbrado"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8), facecolor="white")
    render_matrix(axes[0], a, "Sin alumbrado")
    render_matrix(axes[1], b, "Con alumbrado fijo")
    fig.suptitle("Buffer 300 m · riesgo por tramo × mes × turno", y=.98, fontsize=17, weight="bold")
    fig.text(.5, .91, "Enero–mayo de 2026 · porcentajes por clase real", ha="center", fontsize=12)
    fig.subplots_adjust(top=.78, bottom=.14, left=.06, right=.98, wspace=.25)
    fig.savefig(OUT / "matrices_sin_con_alumbrado_300m_turnos.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(OUT / "informe.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    {"prepare": prepare, "run": run, "report": report}[parser.parse_args().stage]()
