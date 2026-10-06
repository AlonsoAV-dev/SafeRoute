"""Densidad de delitos sobre la red vial, calculada solo con historia anterior.

El evento conserva su único tramo asignado por snapping. Para NKDE se representa
en el punto medio de ese tramo y se propaga por distancia mínima sobre la red
física, sin atravesar calles que no estén conectadas. El kernel triangular se
normaliza por longitud vial alcanzada; el resultado es densidad por 100 m.
"""

from __future__ import annotations

import gc
import heapq
import json
from array import array
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from app.flujo_entrenamiento.red_vial import cargar_o_descargar_grafo, extraer_tramos
from app.flujo_entrenamiento_sidpol.optimizacion.variables import WindowFeatures
from app.flujo_entrenamiento_sidpol.segmentacion import MONTHS


BANDWIDTHS = (100, 250, 500)
NKDE_FIELDS = ("ultimo_mes_turno", "tres_meses_turno", "tres_meses_total", "seis_meses_total")


def _write_json(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


# NKDE: calcula la influencia de cada tramo fuente sobre calles conectadas de la red.
def build_kernels(graph_path: Path, tramos_path: Path, active: np.ndarray, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    paths = {h: output / f"kernel_{h}m.npz" for h in BANDWIDTHS}
    if all(path.exists() for path in paths.values()):
        return paths
    graph = cargar_o_descargar_grafo(graph_path)
    segments, geo = extraer_tramos(graph)
    del graph, geo
    expected = pd.read_csv(tramos_path, usecols=["tramo_id"])["tramo_id"]
    if not expected.equals(segments["tramo_id"]):
        raise ValueError("El orden de tramos del grafo no coincide con la matriz de delitos")
    n = len(segments)
    lengths = np.maximum(segments["longitud_m"].to_numpy(dtype=np.float32), 1.0)
    endpoints = segments[["u", "v"]].to_numpy(dtype=np.int64)
    nodes, inverse = np.unique(endpoints, return_inverse=True)
    endpoints = inverse.reshape(-1, 2)
    del nodes, inverse, segments, expected
    gc.collect()
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(int(endpoints.max()) + 1)]
    incident: list[list[int]] = [[] for _ in adjacency]
    for segment, (u, v) in enumerate(endpoints):
        u, v = int(u), int(v)
        length = float(lengths[segment])
        adjacency[u].append((v, length))
        incident[u].append(segment)
        if v != u:
            adjacency[v].append((u, length))
            incident[v].append(segment)

    # Mide distancias viales entre puntos medios; no usa distancia en línea recta.
    rows, columns, distances = array("i"), array("i"), array("f")
    limit = float(max(BANDWIDTHS))
    for step, source in enumerate(active):
        source = int(source)
        u, v = map(int, endpoints[source])
        half = float(lengths[source]) / 2.0
        reached = {source: 0.0}
        seen: dict[int, float] = {}
        heap: list[tuple[float, int]] = []
        if half < limit:
            seen[u] = half
            heapq.heappush(heap, (half, u))
            if v != u:
                seen[v] = half
                heapq.heappush(heap, (half, v))
        while heap:
            distance, node = heapq.heappop(heap)
            if distance != seen[node]:
                continue
            for target in incident[node]:
                candidate = distance + float(lengths[target]) / 2.0
                if candidate < limit and candidate < reached.get(target, limit):
                    reached[target] = candidate
            for neighbor, edge_length in adjacency[node]:
                candidate = distance + edge_length
                if candidate < limit and candidate < seen.get(neighbor, limit):
                    seen[neighbor] = candidate
                    heapq.heappush(heap, (candidate, neighbor))
        for target, distance in reached.items():
            rows.append(target)
            columns.append(source)
            distances.append(distance)
        if (step + 1) % 10_000 == 0:
            print(f"Vecindarios NKDE: {step + 1:,}/{len(active):,}; pares {len(rows):,}", flush=True)
    row = np.frombuffer(rows, dtype=np.int32)
    column = np.frombuffer(columns, dtype=np.int32)
    distance = np.frombuffer(distances, dtype=np.float32)
    print(f"Vecindarios NKDE completos: {len(active):,} fuentes; {len(row):,} pares", flush=True)
    for bandwidth, path in paths.items():
        if path.exists():
            continue
        selected = distance < bandwidth
        target = row[selected]
        source = column[selected]
        # Kernel triangular: la influencia disminuye hasta cero al alcanzar el ancho de banda.
        kernel = 1.0 - distance[selected] / bandwidth
        # Normaliza por longitud vial para expresar la densidad por cada 100 metros de red.
        normalization = np.bincount(source, weights=kernel * lengths[target], minlength=n)
        values = (100.0 * kernel / np.maximum(normalization[source], 1e-9)).astype(np.float32)
        matrix = sparse.coo_matrix((values, (target, source)), shape=(n, n)).tocsr()
        sparse.save_npz(path, matrix, compressed=True)
        print(f"Kernel {bandwidth} m: {matrix.nnz:,} conexiones", flush=True)
        del target, source, kernel, normalization, values, matrix
        gc.collect()
    _write_json(output / "kernel_metadata.json", {
        "graph": str(graph_path.resolve()), "tramos": str(tramos_path.resolve()),
        "bandwidth_m": list(BANDWIDTHS), "active_source_segments": int(len(active)),
        "source_target_pairs_500m": int(len(row)),
        "definition": "kernel triangular de distancia mínima entre puntos medios sobre red vial, normalizado por longitud vial; densidad por 100 m",
    })
    return paths


# HISTORIAL NKDE: aplica el kernel a los pesos delictivos registrados en cada periodo.
def project_history(kernels: dict[int, Path], weights: sparse.csr_matrix, output: Path):
    rows, segments = weights.shape
    for bandwidth, kernel_path in kernels.items():
        path = output / f"historia_{bandwidth}m.npy"
        marker = output / f"historia_{bandwidth}m_complete.json"
        if marker.exists() and path.exists():
            continue
        kernel = sparse.load_npz(kernel_path)
        projected = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(rows, segments))
        for index in range(rows):
            source = weights.getrow(index).toarray().ravel().astype(np.float32)
            projected[index] = kernel @ source
            if index % 48 == 47:
                print(f"Historia NKDE {bandwidth} m: {MONTHS[index // 4]}", flush=True)
        projected.flush()
        del projected, kernel
        _write_json(marker, {"rows": rows, "segments": segments, "bandwidth_m": bandwidth})
        gc.collect()


class NKDEWindowFeatures(WindowFeatures):
    """Añade densidades de meses anteriores sin cambiar la etiqueta objetivo."""

    def __init__(self, tramos, matrices, source_config, audit, windows, minimum_coverage,
                 high_threshold, history_dir: Path):
        super().__init__(tramos, matrices, source_config, audit, windows, minimum_coverage, high_threshold)
        base = self.groups["multi"].copy()
        self.groups["multi_base"] = base
        self.history = {h: np.load(history_dir / f"historia_{h}m.npy", mmap_mode="r") for h in BANDWIDTHS}
        nkde_start = len(self.names)
        for bandwidth in BANDWIDTHS:
            start = len(self.names)
            self.names += [f"nkde_{bandwidth}m_{field}" for field in NKDE_FIELDS]
            self.groups[f"multi_nkde_{bandwidth}m"] = base + list(range(start, len(self.names)))
        self.groups["multi_nkde_all"] = base + list(range(nkde_start, len(self.names)))

    def block(self, month, turn):
        baseline = super().block(month, turn)
        extra = []
        for bandwidth in BANDWIDTHS:
            history = self.history[bandwidth]
            for window, only_turn in ((1, True), (3, True), (3, False), (6, False)):
                months = self.months(month, window)
                rows = [m * 4 + t for m in months for t in ((turn,) if only_turn else range(4))]
                value = np.asarray(history[rows].sum(axis=0), dtype=np.float32) if rows else np.zeros(self.n_segments, dtype=np.float32)
                extra.append(value / max(len(months), 1))
        return np.column_stack((baseline, *extra)).astype(np.float32, copy=False)
