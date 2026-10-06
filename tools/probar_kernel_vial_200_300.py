"""Ensayo controlado: el delito se propaga por calles conectadas a 200/300 m.

Uso: .venv/Scripts/python tools/probar_kernel_vial_200_300.py
"""

from __future__ import annotations

import gc
import heapq
import json
import sys
from array import array
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import comparar_factores_67 as comparison
from buffer300.report import render_matrix

OUT = ROOT / "outputs" / "evaluacion-modelos-sidpol" / "kernel_vial_200_300"
SOURCE = comparison.OUT / "snap_nuevo_2025_2026.npz"
TRAMOS = comparison.OLD / "tramos_osm.csv"
RADII = (200, 300)
SIGMA = 100.0


def distances(active: np.ndarray, tramos: pd.DataFrame):
    n = len(tramos)
    lengths = np.maximum(tramos["longitud_m"].to_numpy(dtype=np.float32), 1.0)
    ends = tramos[["u", "v"]].to_numpy(dtype=np.int64)
    _, inverse = np.unique(ends, return_inverse=True)
    ends = inverse.reshape(-1, 2)
    del inverse
    gc.collect()

    nodes = int(ends.max()) + 1
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(nodes)]
    incident: list[list[int]] = [[] for _ in range(nodes)]
    for edge, (a, b) in enumerate(ends):
        a, b = int(a), int(b)
        length = float(lengths[edge])
        adjacency[a].append((b, length))
        incident[a].append(edge)
        if a != b:
            adjacency[b].append((a, length))
            incident[b].append(edge)

    row, column, distance_values = array("i"), array("i"), array("f")
    limit = float(max(RADII))
    for step, source in enumerate(active):
        source = int(source)
        a, b = map(int, ends[source])
        half = float(lengths[source]) / 2
        reached = {source: 0.0}
        seen: dict[int, float] = {}
        queue: list[tuple[float, int]] = []
        if half < limit:
            seen[a] = half
            heapq.heappush(queue, (half, a))
            if b != a:
                seen[b] = half
                heapq.heappush(queue, (half, b))
        while queue:
            d, node = heapq.heappop(queue)
            if d != seen[node]:
                continue
            for target in incident[node]:
                candidate = d + float(lengths[target]) / 2
                if candidate < limit and candidate < reached.get(target, limit):
                    reached[target] = candidate
            for neighbor, length in adjacency[node]:
                candidate = d + length
                if candidate < limit and candidate < seen.get(neighbor, limit):
                    seen[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        for target, d in reached.items():
            row.append(target)
            column.append(source)
            distance_values.append(d)
        if (step + 1) % 5000 == 0:
            print(f"RED {step + 1:,}/{len(active):,} tramos fuente; {len(row):,} pares", flush=True)
    print(f"RED completa: {len(active):,} fuentes; {len(row):,} pares a 300 m", flush=True)
    return (np.frombuffer(row, dtype=np.int32),
            np.frombuffer(column, dtype=np.int32),
            np.frombuffer(distance_values, dtype=np.float32), n)


def kernels(row, column, distance, n, radius):
    keep = distance < radius
    target = row[keep]
    source = column[keep]
    d = distance[keep]
    binary = sparse.coo_matrix((np.ones(len(d), np.float32), (target, source)), shape=(n, n)).tocsr()
    gaussian = sparse.coo_matrix((np.exp(-.5 * (d / SIGMA) ** 2).astype(np.float32),
                                  (target, source)), shape=(n, n)).tocsr()
    print(f"KERNEL {radius} m: {binary.nnz:,} conexiones", flush=True)
    return binary, gaussian


def panel(snap, binary, gaussian, n):
    months = len(comparison.PERIODS)
    rows = months * 4
    result = np.empty((months, n, len(comparison.NAMES)), dtype=np.float32)
    for month, name in enumerate(comparison.PERIODS):
        for metric in range(7):
            source = np.asarray(snap[metric * rows + month * 4:metric * rows + month * 4 + 4]
                                .sum(axis=0), dtype=np.float32).ravel()
            result[month, :, metric] = (gaussian if metric == 1 else binary) @ source
        for turn, column in ((0, 10), (1, 7), (2, 8), (3, 9)):
            result[month, :, column] = binary @ snap[month * 4 + turn].toarray().ravel().astype(np.float32)
        print(f"PANEL {name}", flush=True)
    return result


def distributions(metric, factor):
    return {
        key: np.bincount(np.concatenate([comparison.label(metric[m, :, 1] / factor)
                                         for m in month_range]), minlength=3).tolist()
        for key, month_range in (("train", comparison.TRAIN), ("test", comparison.TEST))
    }


def accuracy(metrics):
    matrix = np.asarray(metrics["matriz_confusion"])
    return float(np.trace(matrix) / matrix.sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tramos = pd.read_csv(TRAMOS, usecols=["tramo_id", "u", "v", "longitud_m", "latitud", "longitud"])
    expected = pd.read_csv(comparison.NEW / "tramos_osm.csv", usecols=["tramo_id"])
    if not tramos["tramo_id"].equals(expected["tramo_id"]):
        raise ValueError("La red no conserva el orden de tramos de la matriz")
    static = tramos[["latitud", "longitud", "longitud_m"]].to_numpy(dtype=np.float32)
    factor = np.maximum(static[:, 2] / 100, 1)
    snap = sparse.load_npz(SOURCE).tocsr()
    if snap.shape != (7 * 4 * len(comparison.PERIODS), len(tramos)):
        raise ValueError(f"Dimensiones inesperadas para los delitos: {snap.shape}")
    active = np.unique(snap[:4 * len(comparison.PERIODS)].indices)
    print(f"DELITOS snap: {int(snap[:68].sum()):,}; tramos fuente {len(active):,}", flush=True)
    row, column, distance, n = distances(active, tramos)
    del tramos, expected, active
    gc.collect()

    original = joblib.load(comparison.OLD / "modelo_entrenado_2025_xgboost.joblib")["pipeline"]
    comparison.OUT = OUT
    results = {}
    for radius in RADII:
        binary, gaussian = kernels(row, column, distance, n, radius)
        metric = panel(snap, binary, gaussian, n)
        distribution = distributions(metric, factor)
        print(f"ETIQUETAS {radius} m: {distribution}", flush=True)
        del binary, gaussian
        gc.collect()
        result = comparison.train_arm(f"kernel_vial_{radius}m_mes", metric, static, factor, original)
        result["class_distribution"] = distribution
        result["radius_m"] = radius
        result["kernel"] = "conteos sin decaimiento; severidad gaussiana sigma 100 m; distancia por red entre puntos medios"
        result["metrics"]["accuracy"] = accuracy(result["metrics"])
        comparison.save_json(OUT / f"resultado_kernel_vial_{radius}m_mes.json", result)
        results[radius] = result
        del metric
        gc.collect()

    baseline_paths = {
        200: ROOT / "outputs" / "evaluacion-modelos-sidpol" / "comparacion_67" / "resultado_b_base_nueva_buffer_mes.json",
        300: ROOT / "outputs" / "evaluacion-modelos-sidpol" / "comparacion_67" / "resultado_b_base_nueva_buffer_300m_mes.json",
    }
    baselines = {r: json.loads(path.read_text(encoding="utf-8")) for r, path in baseline_paths.items()}
    for baseline in baselines.values():
        baseline["metrics"]["accuracy"] = accuracy(baseline["metrics"])
    comparison.save_json(OUT / "comparacion.json", {
        "train": "2025-04..2025-12", "test": "2026-01..2026-05",
        "unit": "tramo vial x mes", "source": str(SOURCE),
        "n_snapped_crimes": int(snap[:68].sum()),
        "snap_limit_m": 150, "kernel_sigma_m": SIGMA,
        "threshold": comparison.THRESHOLD,
        "baselines": baselines, "kernel_vial": results,
    })
    fig, axes = plt.subplots(2, 2, figsize=(17, 15), facecolor="white")
    for idx, radius in enumerate(RADII):
        render_matrix(axes[idx, 0], baselines[radius]["metrics"], f"Buffer euclidiano {radius} m")
        render_matrix(axes[idx, 1], results[radius]["metrics"], f"Kernel por red {radius} m")
    fig.suptitle("Mismos delitos y XGBoost · Entrena: abr–dic 2025 · Prueba: ene–may 2026",
                 fontsize=20, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, .965), h_pad=4, w_pad=3)
    image = OUT / "matrices_buffer_vs_kernel_vial.png"
    fig.savefig(image, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"IMAGEN {image}", flush=True)
    for radius in RADII:
        b, k = baselines[radius]["metrics"], results[radius]["metrics"]
        print(f"RESUMEN {radius} m: buffer acc={b['accuracy']:.4f} alto recall={b['recall'][2]:.4f} "
              f"red acc={k['accuracy']:.4f} alto recall={k['recall'][2]:.4f}", flush=True)


if __name__ == "__main__":
    main()
