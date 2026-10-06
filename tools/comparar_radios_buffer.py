"""Compara radios de buffer en el experimento B de comparacion_67.

Ejecutar desde la raíz del proyecto:
    .venv/Scripts/python tools/comparar_radios_buffer.py --radio 300
"""

from __future__ import annotations

import argparse
import gc
import json

import joblib
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from comparar_factores_67 import (
    NAMES, OLD, OUT, PERIODS, graph_data, load_crimes, save_json, train_arm,
)


BASELINE = OUT / "resultado_b_base_nueva_buffer_mes.json"


def prepare_buffer(crimes: pd.DataFrame, lines, transformer, n_segments: int, radius: int):
    cache = OUT / f"buffer_nuevo_2025_2026_{radius}m.npy"
    info_path = OUT / f"datos_espaciales_buffer_{radius}m.json"
    if cache.exists() and info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        cached = np.load(cache, mmap_mode="r")
        if (info.get("records") == len(crimes) and info.get("buffer_radius_m") == radius
                and cached.shape == (len(PERIODS), n_segments, len(NAMES))):
            return cached, info
        raise ValueError(f"La caché de {radius} m no corresponde a la fuente actual")

    xy = transformer.transform(crimes["longitud"].to_numpy(), crimes["latitud"].to_numpy())
    points = shapely.points(*xy)
    tree = STRtree(lines)
    period_ids = pd.Categorical(crimes["periodo"], categories=PERIODS).codes
    turn_ids = pd.Categorical(crimes["turno"], categories=("madrugada", "manana", "tarde", "noche")).codes
    weights = crimes["peso_delito"].to_numpy(dtype=np.float32)
    grave = crimes["es_delito_grave"].to_numpy(dtype=np.float32)
    category = crimes["categoria_delito"].to_numpy()
    buff = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float32,
                                     shape=(len(PERIODS), n_segments, len(NAMES)))
    buff[:] = 0
    monthly = {}
    try:
        for month, period in enumerate(PERIODS):
            event_global = np.flatnonzero(period_ids == month)
            month_points = points[event_global]
            local, segment = tree.query(month_points, predicate="dwithin", distance=float(radius))
            event_index = event_global[local]
            distance = shapely.distance(month_points[local], lines[segment])
            # Misma escala de decaimiento que el brazo B; solo cambia el radio.
            decay = np.exp(-0.5 * (distance / 100.0) ** 2)
            buff[month, :, 0] = np.bincount(segment, minlength=n_segments)
            buff[month, :, 1] = np.bincount(segment, weights=weights[event_index] * decay,
                                            minlength=n_segments)
            buff[month, :, 2] = np.bincount(segment, weights=grave[event_index],
                                            minlength=n_segments)
            for col, name in enumerate(("hurtos", "robos", "extorsiones", "homicidios"), 3):
                buff[month, :, col] = np.bincount(
                    segment, weights=category[event_index] == name, minlength=n_segments)
            for col, turn in enumerate(("manana", "tarde", "noche", "madrugada"), 7):
                code = {"madrugada": 0, "manana": 1, "tarde": 2, "noche": 3}[turn]
                buff[month, :, col] = np.bincount(
                    segment, weights=turn_ids[event_index] == code, minlength=n_segments)
            monthly[period] = {
                "crimes": int(len(event_global)),
                "buffer_pairs": int(len(segment)),
                "buffer_segments_with_signal": int(np.count_nonzero(buff[month, :, 0])),
            }
            print(f"BUFFER {radius} m {period}: {len(event_global):,} delitos, "
                  f"{len(segment):,} pares", flush=True)
            del local, segment, event_index, distance, decay
            gc.collect()
        buff.flush()
    finally:
        del buff
    info = {"source": str(OUT), "records": len(crimes), "buffer_radius_m": radius,
            "decay_sigma_m": 100.0, "monthly": monthly}
    save_json(info_path, info)
    return np.load(cache, mmap_mode="r"), info


def accuracy(result: dict) -> float:
    matrix = np.asarray(result["metrics"]["matriz_confusion"])
    return float(np.trace(matrix) / matrix.sum())


def main(radius: int):
    if radius == 200:
        raise ValueError("El ensayo de 200 m ya existe; elija un radio distinto")
    if not BASELINE.exists():
        raise FileNotFoundError(f"Falta el resultado de 200 m: {BASELINE}")
    tramos, lines, transformer = graph_data()
    static = tramos[["latitud", "longitud", "longitud_m"]].to_numpy(dtype=np.float32)
    factor = np.maximum(static[:, 2] / 100.0, 1.0)
    crimes = load_crimes()
    buff, spatial = prepare_buffer(crimes, lines, transformer, len(tramos), radius)
    del crimes, lines, transformer, tramos
    gc.collect()
    original = joblib.load(OLD / "modelo_entrenado_2025_xgboost.joblib")["pipeline"]
    trial_name = f"b_base_nueva_buffer_{radius}m_mes"
    trial = train_arm(trial_name, buff, static, factor, original)
    del buff
    available = {200: json.loads(BASELINE.read_text(encoding="utf-8")), radius: trial}
    first_100 = OUT / "resultado_b_base_nueva_buffer_100m_mes.json"
    if first_100.exists():
        available[100] = json.loads(first_100.read_text(encoding="utf-8"))
    rows = []
    for compared_radius, result in sorted(available.items()):
        metrics = result["metrics"]
        rows.append({"buffer_m": compared_radius, "test_units": result["test_units"],
                     "support_alto": metrics["support"][2],
                     "prevalencia_alto": metrics["support"][2] / result["test_units"],
                     "accuracy": accuracy(result),
                     "precision_medio": metrics["precision"][1],
                     "recall_medio": metrics["recall"][1],
                     "f1_medio": metrics["f1"][1],
                     "precision_alto": metrics["precision"][2],
                     "recall_alto": metrics["recall"][2],
                     "f1_alto": metrics["f1"][2],
                     "ap_alto": metrics["average_precision_alto"]})
    names = "_".join(str(item) for item in sorted(available))
    comparison = OUT / f"comparacion_buffer_{names}.json"
    report = OUT / f"comparacion_buffer_{names}.md"
    save_json(comparison, {"design": "Solo cambia el radio del buffer; "
                           "misma fuente, red, decaimiento, modelo, entrenamiento y prueba",
                           "sources": {str(r): str(OUT / f"resultado_{v['arm']}.json")
                                       for r, v in available.items()},
                           f"spatial_{radius}m": spatial, "rows": rows})
    lines_report = [f"# Comparación de buffer: {', '.join(str(r) + ' m' for r in sorted(available))}", "",
                    "Base nueva 2018–2026 restringida a enero 2025–mayo 2026. "
                    "Entrenamiento abril–diciembre de 2025 y prueba enero–mayo de 2026. "
                    "La unidad es tramo vial × mes. El umbral alto y el decaimiento (σ=100 m) "
                    "son iguales en todos los ensayos.", "",
                    "| Buffer | Altos reales | Prevalencia alto | Accuracy | Precisión medio | Recall medio | Precisión alto | Recall alto | F1 alto | AP alto |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines_report.append(
            f"| {row['buffer_m']} m | {row['support_alto']:,} | "
            f"{row['prevalencia_alto']:.2%} | {row['accuracy']:.2%} | "
            f"{row['precision_medio']:.2%} | {row['recall_medio']:.2%} | "
            f"{row['precision_alto']:.2%} | {row['recall_alto']:.2%} | "
            f"{row['f1_alto']:.2%} | {row['ap_alto']:.2%} |")
    lines_report += ["", "Los radios cambian qué tramos reciben etiqueta de riesgo alto. "
                     "Por ello, las métricas se calculan sobre la misma cantidad de unidades, "
                     "pero con verdades de referencia distintas.", ""]
    report.write_text("\n".join(lines_report), encoding="utf-8")
    print(report, flush=True)
    print(comparison, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--radio", type=int, required=True)
    main(parser.parse_args().radio)
