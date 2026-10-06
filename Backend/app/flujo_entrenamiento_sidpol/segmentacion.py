from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy import sparse
from shapely import STRtree

from app.flujo_entrenamiento.red_vial import cargar_o_descargar_grafo, extraer_tramos
from app.flujo_entrenamiento_sidpol.config import TURNOS, TrainingConfig


CATEGORIES = ("robo", "hurto", "extorsion", "homicidio")
MATRIX_NAMES = ("count", "weight", "grave", *CATEGORIES)
FIRST_YEAR = 2018
LAST_YEAR = 2026
MONTHS = tuple(f"{year}-{month:02d}" for year in range(FIRST_YEAR, LAST_YEAR + 1) for month in range(1, 13))
TURN_INDEX = {turn: index for index, turn in enumerate(TURNOS)}


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _month_indices(periods: pd.Series) -> np.ndarray:
    years = periods.str.slice(0, 4).astype(np.int16).to_numpy()
    months = periods.str.slice(5, 7).astype(np.int8).to_numpy()
    return (years - FIRST_YEAR) * 12 + months - 1


# Asocia los delitos a la red según el radio configurado y guarda las matrices de agregación.
def asociar_delitos(config: TrainingConfig, normalized_csv: Path) -> tuple[pd.DataFrame, dict]:
    """Asocia un evento al tramo OSM más cercano dentro del radio configurado.

    Las matrices escasas permiten cambiar modelos y variables sin repetir el
    costoso emparejamiento espacial, siempre que fuente, grafo y radio coincidan.
    """
    out = config.output
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "asignacion_espacial.json"
    tramos_path = out / "tramos_osm.csv"
    expected = {
        "source": _fingerprint(normalized_csv),
        "graph": _fingerprint(config.graph),
        "match_radius_m": config.match_radius_m,
        "turnos": list(TURNOS),
    }
    if manifest_path.exists() and tramos_path.exists() and all(
        (out / f"matriz_{name}.npz").exists() for name in MATRIX_NAMES
    ):
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(previous.get(key) == value for key, value in expected.items()):
            print("Asignación espacial reutilizada desde caché", flush=True)
            return pd.read_csv(tramos_path), previous

    graph = cargar_o_descargar_grafo(config.graph)
    tramos, tramos_geo = extraer_tramos(graph)
    del graph
    tramos = tramos[["tramo_id", "latitud", "longitud", "longitud_m"]].copy()
    tramos.to_csv(tramos_path, index=False)
    # Trabaja en metros (UTM 18S) para aplicar correctamente el umbral de snapping.
    metric_lines = tramos_geo.to_crs("EPSG:32718").geometry.to_numpy()
    tree = STRtree(metric_lines)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)
    del tramos_geo, metric_lines
    n_segments = len(tramos)
    n_rows = len(MONTHS) * len(TURNOS)
    row_chunks: list[np.ndarray] = []
    segment_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    category_chunks: list[np.ndarray] = []
    counts = Counter()
    monthly = Counter()
    for chunk in pd.read_csv(
        normalized_csv,
        usecols=["periodo", "latitud", "longitud", "turno", "peso_delito", "categoria_delito"],
        chunksize=25_000,
        dtype={"periodo": "string", "turno": "string", "categoria_delito": "string"},
    ):
        x, y = transformer.transform(
            chunk["longitud"].to_numpy(dtype=float),
            chunk["latitud"].to_numpy(dtype=float),
        )
        points = shapely.points(x, y)
        # SNAPPING: un delito se asigna al tramo más cercano dentro del radio configurado.
        # all_matches=False evita duplicarlo en calles empatadas; fuera del umbral no se asigna.
        pairs, _ = tree.query_nearest(
            points, max_distance=config.match_radius_m,
            all_matches=False, return_distance=True,
        )
        source_idx, segment_idx = pairs
        counts["geolocalizados"] += len(chunk)
        counts["asignados"] += len(source_idx)
        periods = chunk["periodo"].iloc[source_idx].reset_index(drop=True)
        turn_ids = chunk["turno"].iloc[source_idx].map(TURN_INDEX).to_numpy(dtype=np.int8)
        if np.any(pd.isna(turn_ids)):
            raise ValueError("La fuente contiene turnos fuera del catálogo")
        month_ids = _month_indices(periods)
        if np.any((month_ids < 0) | (month_ids >= len(MONTHS))):
            raise ValueError("Mes fuera del período 2018–2026")
        row_chunks.append((month_ids * len(TURNOS) + turn_ids).astype(np.int32))
        segment_chunks.append(segment_idx.astype(np.int32))
        weight_chunks.append(chunk["peso_delito"].iloc[source_idx].to_numpy(dtype=np.float32))
        category_chunks.append(
            chunk["categoria_delito"].iloc[source_idx].astype(str).to_numpy()
        )
        monthly.update(periods.tolist())
        if counts["geolocalizados"] % 250_000 < len(chunk):
            print(f"Asociados {counts['asignados']:,} / {counts['geolocalizados']:,}", flush=True)
    rows = np.concatenate(row_chunks)
    columns = np.concatenate(segment_chunks)
    weights = np.concatenate(weight_chunks)
    categories = np.concatenate(category_chunks)
    shape = (n_rows, n_segments)
    data = {
        "count": np.ones(len(rows), dtype=np.float32),
        "weight": weights,
        "grave": (weights >= 4).astype(np.float32),
    }
    for category in CATEGORIES:
        data[category] = (categories == category).astype(np.float32)
    for name, values in data.items():
        matrix = sparse.coo_matrix((values, (rows, columns)), shape=shape).tocsr()
        matrix.sum_duplicates()
        sparse.save_npz(out / f"matriz_{name}.npz", matrix, compressed=True)
    manifest = {
        **expected,
        "total_geolocalizados": counts["geolocalizados"],
        "total_asignados": counts["asignados"],
        "sin_tramo_cercano": counts["geolocalizados"] - counts["asignados"],
        "tramos": n_segments,
        "monthly_assigned": dict(sorted(monthly.items())),
        "assignment_method": "tramo OSM más cercano dentro del radio; un delito, un tramo",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return tramos, manifest


# Recupera las matrices dispersas necesarias para construir las variables históricas.
def cargar_matrices(output: Path) -> dict[str, sparse.csr_matrix]:
    return {name: sparse.load_npz(output / f"matriz_{name}.npz") for name in MATRIX_NAMES}
