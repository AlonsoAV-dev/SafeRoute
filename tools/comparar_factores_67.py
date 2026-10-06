"""Ablación del 67 % antiguo: fuente, asociación vial y unidad temporal.

Conserva el XGBoost y el protocolo del experimento enero–mayo 2026.
Ejecutar desde la raíz: .venv/Scripts/python tools/comparar_factores_67.py
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy import sparse
from shapely import STRtree
from sklearn.base import clone
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "Backend"
sys.path.insert(0, str(BACKEND))

from app.flujo_entrenamiento.modelos import _submuestrear_clase_baja
from app.flujo_entrenamiento.red_vial import cargar_o_descargar_grafo, extraer_tramos
from app.flujo_entrenamiento.riesgo import agregar_riesgo_base


OLD = BACKEND / "data" / "procesados" / "evaluacion_2026"
NEW = BACKEND / "data" / "experimentos_sidpol" / "escenario_distrito" / "procesados"
OUT = ROOT / "outputs" / "evaluacion-modelos-sidpol" / "comparacion_67"
PERIODS = tuple(f"{y}-{m:02d}" for y, stop in ((2025, 12), (2026, 5)) for m in range(1, stop + 1))
MONTH_ID = {period: i for i, period in enumerate(PERIODS)}
TRAIN = range(3, 12)  # abril–diciembre 2025
TEST = range(12, 17)  # enero–mayo 2026
TURNS = ("madrugada", "manana", "tarde", "noche")
TURN_ID = {name: index for index, name in enumerate(TURNS)}
NAMES = ("frecuencia_delitos", "suma_pesos", "delitos_graves", "hurtos", "robos",
         "extorsiones", "homicidios", "delitos_manana", "delitos_tarde",
         "delitos_noche", "delitos_madrugada")
THRESHOLD = float(json.loads((OLD / "resumen_evaluacion_2026.json").read_text(encoding="utf-8"))
                  ["metricas"][1]["umbral_alto"])


def save_json(path: Path, obj: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_crimes() -> pd.DataFrame:
    columns = ["fecha", "periodo", "latitud", "longitud", "turno", "subtipo_delito", "modalidad"]
    chunks = []
    source = NEW / "delitos_geolocalizados.csv"
    for chunk in pd.read_csv(source, usecols=columns, chunksize=100_000):
        chunks.append(chunk.loc[chunk["periodo"].between("2025-01", "2026-05")])
    crimes = pd.concat(chunks, ignore_index=True)
    # Las reglas textuales antiguas se aplican a ambas bases: así no cambia la escala de gravedad.
    crimes = agregar_riesgo_base(crimes)
    if len(crimes) != 284_683:
        raise ValueError(f"La fuente nueva cambió: {len(crimes):,} delitos en el periodo común")
    if not crimes["turno"].isin(TURNS).all():
        raise ValueError("Turnos desconocidos")
    return crimes


def graph_data():
    graph = cargar_o_descargar_grafo(BACKEND / "data" / "red_vial_lima.graphml")
    tramos, geo = extraer_tramos(graph)
    del graph
    old_ids = pd.read_csv(OLD / "tramos_osm.csv", usecols=["tramo_id"])["tramo_id"]
    if not old_ids.equals(tramos["tramo_id"]):
        raise ValueError("La red antigua y la actual no ordenan igual sus tramos")
    crs = geo.estimate_utm_crs()
    lines = geo.to_crs(crs).geometry.to_numpy()
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return tramos, lines, transformer


def prepare_spatial(crimes: pd.DataFrame, lines, transformer, n_segments: int):
    buffer_cache = OUT / "buffer_nuevo_2025_2026.npy"
    snap_cache = OUT / "snap_nuevo_2025_2026.npz"
    info_path = OUT / "datos_espaciales.json"
    if buffer_cache.exists() and snap_cache.exists() and info_path.exists():
        return np.load(buffer_cache, mmap_mode="r"), sparse.load_npz(snap_cache), json.loads(info_path.read_text(encoding="utf-8"))
    xy = transformer.transform(crimes["longitud"].to_numpy(), crimes["latitud"].to_numpy())
    points = shapely.points(*xy)
    tree = STRtree(lines)
    period_ids = crimes["periodo"].map(MONTH_ID).to_numpy(dtype=np.int8)
    turn_ids = crimes["turno"].map(TURN_ID).to_numpy(dtype=np.int8)
    weights = crimes["peso_delito"].to_numpy(dtype=np.float32)
    grave = crimes["es_delito_grave"].to_numpy(dtype=np.float32)
    category = crimes["categoria_delito"].to_numpy()
    buff = np.lib.format.open_memmap(buffer_cache, mode="w+", dtype=np.float32,
                                      shape=(len(PERIODS), n_segments, len(NAMES)))
    buff[:] = 0
    snap_rows, snap_cols, snap_weights, snap_grave, snap_categories = [], [], [], [], []
    monthly = {}
    for month, period in enumerate(PERIODS):
        event_global = np.flatnonzero(period_ids == month)
        month_points = points[event_global]
        pair = tree.query(month_points, predicate="dwithin", distance=200.0)
        event_local, segment = pair
        event_index = event_global[event_local]
        distance = shapely.distance(month_points[event_local], lines[segment])
        decay = np.exp(-0.5 * (distance / 100.0) ** 2)
        buff[month, :, 0] = np.bincount(segment, minlength=n_segments)
        buff[month, :, 1] = np.bincount(segment, weights=weights[event_index] * decay, minlength=n_segments)
        buff[month, :, 2] = np.bincount(segment, weights=grave[event_index], minlength=n_segments)
        for col, name in enumerate(("hurtos", "robos", "extorsiones", "homicidios"), 3):
            buff[month, :, col] = np.bincount(segment, weights=category[event_index] == name, minlength=n_segments)
        for col, shift in enumerate(("manana", "tarde", "noche", "madrugada"), 7):
            buff[month, :, col] = np.bincount(segment, weights=turn_ids[event_index] == TURN_ID[shift], minlength=n_segments)
        nearest, _ = tree.query_nearest(month_points, max_distance=150.0,
                                       all_matches=False, return_distance=True)
        local_snap, segment_snap = nearest
        matched = event_global[local_snap]
        snap_rows.append((month * 4 + turn_ids[matched]).astype(np.int32))
        snap_cols.append(segment_snap.astype(np.int32))
        snap_weights.append(weights[matched])
        snap_grave.append(grave[matched])
        snap_categories.append(category[matched])
        monthly[period] = {"crimes": int(len(event_global)), "buffer_pairs": int(len(segment)),
                           "buffer_segments_with_signal": int(np.count_nonzero(buff[month, :, 0])),
                           "snapped": int(len(matched))}
        print(f"ESPACIAL {period}: {len(event_global):,} delitos, {len(segment):,} pares buffer, {len(matched):,} snap", flush=True)
        del pair, distance, decay
        gc.collect()
    buff.flush()
    del buff
    rows = np.concatenate(snap_rows)
    cols = np.concatenate(snap_cols)
    values = np.concatenate(snap_weights)
    grave_values = np.concatenate(snap_grave)
    categories = np.concatenate(snap_categories)
    # Siete matrices apiladas: una por variable usada al reconstruir el panel.
    metrics = [np.ones(len(rows), dtype=np.float32), values, grave_values]
    metrics += [(categories == label).astype(np.float32) for label in ("hurtos", "robos", "extorsiones", "homicidios")]
    snap = sparse.vstack([sparse.coo_matrix((v, (rows, cols)), shape=(len(PERIODS) * 4, n_segments)).tocsr()
                          for v in metrics], format="csr")
    sparse.save_npz(snap_cache, snap, compressed=True)
    info = {"source": str(NEW / "delitos_geolocalizados.csv"), "records": int(len(crimes)),
            "weight_rules": "reglas textuales antiguas para todas las variantes",
            "buffer_radius_m": 200, "snapping_radius_m": 150, "monthly": monthly}
    save_json(info_path, info)
    return np.load(buffer_cache, mmap_mode="r"), snap, info


def prepare_snap_200(crimes: pd.DataFrame, lines, transformer, n_segments: int):
    path = OUT / "snap_nuevo_2025_2026_200m.npz"
    if path.exists():
        return sparse.load_npz(path)
    xy = transformer.transform(crimes["longitud"].to_numpy(), crimes["latitud"].to_numpy())
    points = shapely.points(*xy)
    tree = STRtree(lines)
    pair, _ = tree.query_nearest(points, max_distance=200.0,
                                 all_matches=False, return_distance=True)
    matched, segment = pair
    period_ids = crimes["periodo"].map(MONTH_ID).to_numpy(dtype=np.int8)
    turn_ids = crimes["turno"].map(TURN_ID).to_numpy(dtype=np.int8)
    rows = (period_ids[matched] * 4 + turn_ids[matched]).astype(np.int32)
    weights = crimes["peso_delito"].to_numpy(dtype=np.float32)[matched]
    grave = crimes["es_delito_grave"].to_numpy(dtype=np.float32)[matched]
    categories = crimes["categoria_delito"].to_numpy()[matched]
    values = [np.ones(len(rows), dtype=np.float32), weights, grave]
    values += [(categories == label).astype(np.float32)
               for label in ("hurtos", "robos", "extorsiones", "homicidios")]
    snap = sparse.vstack([sparse.coo_matrix((v, (rows, segment)),
                                             shape=(len(PERIODS) * 4, n_segments)).tocsr()
                          for v in values], format="csr")
    sparse.save_npz(path, snap, compressed=True)
    print(f"SNAP 200 m: {len(matched):,}/{len(crimes):,} delitos", flush=True)
    return snap


def snap_monthly(snap: sparse.csr_matrix, n_segments: int):
    metric = np.empty((len(PERIODS), n_segments, len(NAMES)), dtype=np.float32)
    for month in range(len(PERIODS)):
        for index in range(7):
            metric[month, :, index] = np.asarray(snap[index * len(PERIODS) * 4 + month * 4:
                                                      index * len(PERIODS) * 4 + month * 4 + 4].sum(axis=0)).ravel()
        for turn in range(4):
            col = {0: 10, 1: 7, 2: 8, 3: 9}[turn]
            metric[month, :, col] = snap[month * 4 + turn].toarray().ravel()
    return metric


def label(risk: np.ndarray) -> np.ndarray:
    return np.where(risk >= THRESHOLD, 2, np.where(risk > 0, 1, 0)).astype(np.int8)


def features(metric, static, factor, month, turn=None):
    history = np.asarray(metric[month - 3:month].sum(axis=0), dtype=np.float32)
    x = np.empty((len(factor), 16 + (2 if turn is not None else 0)), dtype=np.float32)
    x[:, :3] = static
    x[:, 3:14] = history
    x[:, 14] = history[:, 0] / factor
    x[:, 15] = history[:, 1] / factor
    if turn is not None:
        x[:, 16] = np.sin(np.pi * turn / 2)
        x[:, 17] = np.cos(np.pi * turn / 2)
    return x


def measures(truth, prediction, score):
    p, r, f1, counts = precision_recall_fscore_support(truth, prediction, labels=[0, 1, 2], zero_division=0)
    return {"support": counts.tolist(), "precision": p.tolist(), "recall": r.tolist(), "f1": f1.tolist(),
            "f1_medio_alto": float((f1[1] + f1[2]) / 2),
            "average_precision_alto": float(average_precision_score(truth == 2, score)),
            "matriz_confusion": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist()}


def train_arm(name, metric, static, factor, old_pipeline, snap=None, by_turn=False):
    result_path = OUT / f"resultado_{name}.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    n = len(factor)
    blocks = [(m, t) for m in TRAIN for t in (range(4) if by_turn else (None,))]
    train_labels = []
    for m, t in blocks:
        risk = (metric[m, :, 1] if t is None else snap[len(PERIODS) * 4 + m * 4 + t].toarray().ravel()) / factor
        train_labels.append(label(risk))
    y_all = np.concatenate(train_labels)
    selected = _submuestrear_clase_baja(y_all, 42)
    x_train = np.empty((len(selected), 16 + (2 if by_turn else 0)), dtype=np.float32)
    y_train = y_all[selected]
    for index, (m, t) in enumerate(blocks):
        chosen = selected[(selected >= index * n) & (selected < (index + 1) * n)]
        if len(chosen):
            # searchsorted keeps the same selected order without materializing all rows.
            first = np.searchsorted(selected, chosen[0])
            x_train[first:first + len(chosen)] = features(metric, static, factor, m, t)[chosen - index * n]
    pipeline = clone(old_pipeline)
    print(f"ENTRENANDO {name}: {len(y_all):,} unidades, {len(selected):,} muestras, "
          f"clases {np.bincount(y_all, minlength=3).tolist()}", flush=True)
    pipeline.fit(x_train, y_train, modelo__sample_weight=compute_sample_weight(class_weight="balanced", y=y_train))
    joblib.dump({"pipeline": pipeline, "threshold": THRESHOLD}, OUT / f"modelo_{name}.joblib", compress=3)
    del x_train, y_all, y_train
    gc.collect()
    truth_chunks, pred_chunks, score_chunks = [], [], []
    for m in TEST:
        for t in (range(4) if by_turn else (None,)):
            x = features(metric, static, factor, m, t)
            risk = (metric[m, :, 1] if t is None else snap[len(PERIODS) * 4 + m * 4 + t].toarray().ravel()) / factor
            truth = label(risk)
            pred = pipeline.predict(x).astype(np.int8)
            probability = pipeline.predict_proba(x)[:, 2].astype(np.float32)
            truth_chunks.append(truth)
            pred_chunks.append(pred)
            score_chunks.append(probability)
        print(f"EVALUADO {name}: {PERIODS[m]}", flush=True)
    truth = np.concatenate(truth_chunks)
    pred = np.concatenate(pred_chunks)
    score = np.concatenate(score_chunks)
    result = {"arm": name, "train_months": [PERIODS[m] for m in TRAIN],
              "test_months": [PERIODS[m] for m in TEST], "threshold": THRESHOLD,
              "n_features": 18 if by_turn else 16,
              "train_units": len(blocks) * n, "test_units": len(truth),
              "metrics": measures(truth, pred, score)}
    save_json(result_path, result)
    print(f"RESULTADO {name}: alto P={result['metrics']['precision'][2]:.4f} "
          f"R={result['metrics']['recall'][2]:.4f}, F1={result['metrics']['f1'][2]:.4f}", flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old_report = pd.read_csv(OLD / "classification_report_externo_xgboost.csv", index_col=0)
    old_matrix = pd.read_csv(OLD / "matriz_confusion_externa_xgboost.csv", index_col=0).to_numpy()
    old_result = {"arm": "a_base_antigua_buffer_mes", "source": str(OLD / "delitos_2025_2026_limpios.csv"),
                  "train_months": [PERIODS[m] for m in TRAIN], "test_months": [PERIODS[m] for m in TEST],
                  "threshold": THRESHOLD, "n_features": 16,
                  "train_units": 9 * 228294, "test_units": 5 * 228294,
                  "metrics": {"support": old_matrix.sum(axis=1).tolist(),
                              "precision": old_report.loc[["bajo", "medio", "alto"], "precision"].tolist(),
                              "recall": old_report.loc[["bajo", "medio", "alto"], "recall"].tolist(),
                              "f1": old_report.loc[["bajo", "medio", "alto"], "f1-score"].tolist(),
                              "matriz_confusion": old_matrix.tolist()}}
    old_result["metrics"]["f1_medio_alto"] = float(np.mean(old_result["metrics"]["f1"][1:]))
    old_result["metrics"]["average_precision_alto"] = float(pd.read_csv(OLD / "metricas_externas_xgboost.csv")
                                                              ["pr_auc_riesgo_alto"].iloc[0])
    save_json(OUT / "resultado_a_base_antigua_buffer_mes.json", old_result)
    tramos, lines, transformer = graph_data()
    static = tramos[["latitud", "longitud", "longitud_m"]].to_numpy(dtype=np.float32)
    factor = np.maximum(static[:, 2] / 100.0, 1.0)
    crimes = load_crimes()
    buff, snap, spatial = prepare_spatial(crimes, lines, transformer, len(tramos))
    snap_200 = prepare_snap_200(crimes, lines, transformer, len(tramos))
    snapped_200_count = int(snap_200[:len(PERIODS) * 4].sum())
    del crimes, lines, transformer, tramos
    gc.collect()
    original = joblib.load(OLD / "modelo_entrenado_2025_xgboost.joblib")["pipeline"]
    results = [old_result]
    results.append(train_arm("b_base_nueva_buffer_mes", buff, static, factor, original))
    del buff
    gc.collect()
    metric_200 = snap_monthly(snap_200, len(factor))
    results.append(train_arm("c_base_nueva_snap_200_mes", metric_200, static, factor, original))
    del metric_200, snap_200
    gc.collect()
    metric = snap_monthly(snap, len(factor))
    results.append(train_arm("c_base_nueva_snap_mes", metric, static, factor, original))
    results.append(train_arm("d_base_nueva_snap_turno", metric, static, factor, original,
                             snap=snap, by_turn=True))
    summary = {"status": "complete", "design": "escalera A→B→C→D→E; solo cambia fuente, luego asociación espacial, luego tolerancia, luego unidad temporal",
               "fixed": {"model": "XGBoost antiguo, 260 árboles y profundidad 6", "train": "2025-04..2025-12",
                         "test": "2026-01..2026-05", "history_months": 3, "threshold_high": THRESHOLD,
                         "weights": "reglas textuales antiguas", "buffer_m": 200, "snap_m": [200, 150]},
               "spatial": {**spatial, "snapped_200m": snapped_200_count}, "arms": results}
    save_json(OUT / "comparacion_controlada.json", summary)
    print(OUT / "comparacion_controlada.json", flush=True)


if __name__ == "__main__":
    main()
