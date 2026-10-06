"""Propagación vial de delitos 2018–2026 a 200/300 m con corte temporal.

Uso: .venv/Scripts/python tools/probar_kernel_vial_2018_2026.py
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from probar_kernel_vial_200_300 import distances, kernels, SIGMA
from buffer300.report import render_matrix

DATA = ROOT / "Backend/data/experimentos_sidpol/escenario_distrito/procesados"
OLD = ROOT / "Backend/data/procesados/evaluacion_2026"
OUT = ROOT / "outputs/evaluacion-modelos-sidpol/kernel_vial_2018_2026"
MONTHS = tuple(f"{y}-{m:02d}" for y in range(2018, 2027) for m in range(1, 13))
FIRST = MONTHS.index("2018-04")
TRAIN_END = MONTHS.index("2025-10")
VALIDATION = (MONTHS.index("2025-11"), MONTHS.index("2025-12"))
TEST = tuple(range(MONTHS.index("2026-01"), MONTHS.index("2026-08") + 1))
HIGH = 3.0
SAMPLE_PER_CLASS_MONTH = 2000
SEED = 42
METRICS = ("count", "weight", "grave", "hurto", "robo", "extorsion", "homicidio")


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def label(risk):
    return np.where(risk >= HIGH, 2, np.where(risk > 0, 1, 0)).astype(np.int8)


def panel_features(panel, static, factor, month):
    history = np.asarray(panel[month - 3:month].sum(axis=0), dtype=np.float32)
    x = np.empty((len(factor), 16), dtype=np.float32)
    x[:, :3] = static
    x[:, 3:14] = history
    x[:, 14] = history[:, 0] / factor
    x[:, 15] = history[:, 1] / factor
    return x


def make_panel(radius, kernel_binary, kernel_gauss, source, n):
    path = OUT / f"panel_{radius}m.npy"
    complete = OUT / f"panel_{radius}m_complete.json"
    if complete.exists() and path.exists():
        return np.load(path, mmap_mode="r")
    panel = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                      shape=(MONTHS.index("2026-08") + 1, n, 11))
    for month in range(panel.shape[0]):
        row = month * 4
        for index, name in enumerate(METRICS):
            events = np.asarray(source[name][row:row + 4].sum(axis=0), dtype=np.float32).ravel()
            panel[month, :, index] = (kernel_gauss if name == "weight" else kernel_binary) @ events
        for turn, col in ((0, 10), (1, 7), (2, 8), (3, 9)):
            events = source["count"].getrow(row + turn).toarray().ravel().astype(np.float32)
            panel[month, :, col] = kernel_binary @ events
        if month % 6 == 5 or month == panel.shape[0] - 1:
            panel.flush()
            print(f"PANEL {radius} m: {MONTHS[month]}", flush=True)
    del panel
    save_json(complete, {"radius_m": radius, "last_month": "2026-08",
                         "n_segments": n, "features": list(METRICS) + ["manana", "tarde", "noche", "madrugada"]})
    return np.load(path, mmap_mode="r")


def train(panel, static, factor, original, radius):
    path = OUT / f"resultado_{radius}m.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    x_parts, y_parts = [], []
    train_distribution = np.zeros(3, dtype=np.int64)
    for month in range(FIRST, TRAIN_END + 1):
        y = label(panel[month, :, 1] / factor)
        train_distribution += np.bincount(y, minlength=3)
        chosen = np.concatenate([
            rng.choice(indices, size=min(SAMPLE_PER_CLASS_MONTH, len(indices)), replace=False)
            for cls in range(3)
            if len(indices := np.flatnonzero(y == cls))
        ])
        x_parts.append(panel_features(panel, static, factor, month)[chosen])
        y_parts.append(y[chosen])
        if month % 12 == 11:
            print(f"MUESTRAS {radius} m: {MONTHS[month]}", flush=True)
    x_train = np.concatenate(x_parts)
    y_train = np.concatenate(y_parts)
    del x_parts, y_parts
    model = clone(original)
    print(f"ENTRENANDO {radius} m: {len(y_train):,} muestras; clases {train_distribution.tolist()}", flush=True)
    model.fit(x_train, y_train)
    joblib.dump({"pipeline": model, "radius_m": radius, "threshold": HIGH},
                OUT / f"modelo_{radius}m.joblib", compress=3)
    del x_train, y_train
    gc.collect()

    def evaluate(months):
        truth, pred, prob = [], [], []
        for month in months:
            x = panel_features(panel, static, factor, month)
            y = label(panel[month, :, 1] / factor)
            p = model.predict_proba(x)
            truth.append(y)
            pred.append(np.argmax(p, axis=1).astype(np.int8))
            prob.append(p[:, 2].astype(np.float32))
            print(f"EVALUADO {radius} m: {MONTHS[month]}", flush=True)
        truth = np.concatenate(truth)
        pred = np.concatenate(pred)
        prob = np.concatenate(prob)
        matrix = confusion_matrix(truth, pred, labels=[0, 1, 2])
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, pred, labels=[0, 1, 2], zero_division=0)
        return {"matriz_confusion": matrix.tolist(), "accuracy": float(np.trace(matrix) / matrix.sum()),
                "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(),
                "support": support.tolist(),
                "average_precision_alto": float(average_precision_score(truth == 2, prob))}

    result = {"radius_m": radius, "source": str(DATA),
              "train": [MONTHS[FIRST], MONTHS[TRAIN_END]],
              "validation": [MONTHS[i] for i in VALIDATION],
              "test": [MONTHS[i] for i in TEST], "train_class_distribution": train_distribution.tolist(),
              "threshold_high": HIGH, "sample_per_class_month": SAMPLE_PER_CLASS_MONTH,
              "metrics_validation": evaluate(VALIDATION), "metrics_test": evaluate(TEST)}
    save_json(path, result)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    segments = pd.read_csv(OLD / "tramos_osm.csv", usecols=["tramo_id", "u", "v", "longitud_m", "latitud", "longitud"])
    ids = pd.read_csv(DATA / "tramos_osm.csv", usecols=["tramo_id"])
    if not segments["tramo_id"].equals(ids["tramo_id"]):
        raise ValueError("Los tramos no coinciden")
    static = segments[["latitud", "longitud", "longitud_m"]].to_numpy(dtype=np.float32)
    factor = np.maximum(static[:, 2] / 100, 1)
    source = {name: sparse.load_npz(DATA / f"matriz_{name}.npz").tocsr() for name in METRICS}
    n = len(segments)
    active = np.unique(source["count"].indices)
    print(f"FUENTE 2018–2026: {int(source['count'].sum()):,} delitos; {len(active):,} tramos", flush=True)
    row, column, distance, count = distances(active, segments)
    del segments, ids, active
    gc.collect()
    if n != count:
        raise ValueError("Inconsistencia de tramos")
    original = joblib.load(OLD / "modelo_entrenado_2025_xgboost.joblib")["pipeline"]
    results = {}
    for radius in (200, 300):
        binary, gaussian = kernels(row, column, distance, n, radius)
        panel = make_panel(radius, binary, gaussian, source, n)
        del binary, gaussian
        gc.collect()
        results[radius] = train(panel, static, factor, original, radius)
        del panel
        gc.collect()
    save_json(OUT / "comparacion.json", results)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="white")
    for ax, radius in zip(axes, (200, 300)):
        render_matrix(ax, results[radius]["metrics_test"], f"Kernel vial {radius} m")
    fig.suptitle("Delitos desde 2018 · Prueba: enero–agosto 2026 · tramo × mes", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, .94), w_pad=3)
    figure = OUT / "matrices_2018_2026.png"
    fig.savefig(figure, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"IMAGEN {figure}", flush=True)
    for radius, result in results.items():
        m = result["metrics_test"]
        print(f"RESUMEN {radius} m: acc={m['accuracy']:.4f}, recall_alto={m['recall'][2]:.4f}, "
              f"precision_alto={m['precision'][2]:.4f}", flush=True)


if __name__ == "__main__":
    main()
