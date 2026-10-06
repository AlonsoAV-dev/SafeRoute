"""Ablación de alumbrado estático para el XGBoost mensual de buffer 300 m.

La única diferencia entre brazos son cuatro atributos del inventario de
luminarias. La geometría, etiquetas, meses, muestreo y entrenamiento se
mantienen iguales. El alumbrado se trata como fijo durante 2018–2026.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from buffer300.data import DATA, PERIODS, ROOT as DATA_ROOT, save_json, load_static
from buffer300.modelos import TrainingData, fit_model, predictions, evaluate, matrix_metrics, score
from buffer300.features import Features
from buffer300.report import pct, render_matrix

SOURCE = Path(r"C:\Users\Alonso\Downloads\equipo_ap_luminaria_1.csv")
OUT = ROOT / "outputs/evaluacion-modelos-sidpol/alumbrado_buffer_300_v1"
LIGHTS = DATA / "alumbrado_estatico_v1.npy"
AUDIT = OUT / "auditoria_alumbrado.json"
SELECTION = ROOT / "outputs/evaluacion-modelos-sidpol/optimizacion_buffer_300_mensual_v2/selection.json"
REFERENCE = ROOT / "outputs/evaluacion-modelos-sidpol/optimizacion_buffer_300_mensual_v2/final_results.json"
NAMES = ["luminarias_50m_log", "luminarias_100m_log", "distancia_luminaria_log_m", "luminaria_30m"]
VALIDATION = ("2025-11", "2025-12")
TEST = ("2026-01", "2026-02", "2026-03", "2026-04", "2026-05")
ADDITIONAL = ("2026-06", "2026-07", "2026-08")


def projected(xy):
    """Aproximación métrica local usada también para la vecindad del modelo."""
    result = np.asarray(xy, dtype=np.float64).copy()
    result[:, 0] *= 111_320 * np.cos(np.deg2rad(-12))
    result[:, 1] *= 111_320
    return result


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if LIGHTS.exists() and AUDIT.exists():
        saved = json.loads(AUDIT.read_text(encoding="utf-8"))
        stat = SOURCE.stat()
        if saved["source_size"] != stat.st_size or saved["source_mtime_ns"] != stat.st_mtime_ns:
            raise ValueError("Cambió el inventario de alumbrado; cree otra versión del experimento")
        print("Alumbrado estático ya preparado", flush=True)
        return
    static, _ = load_static()
    lat, lon = static[:, 0], static[:, 1]
    bounds = {"lon_min": float(lon.min()-.01), "lon_max": float(lon.max()+.01),
              "lat_min": float(lat.min()-.01), "lat_max": float(lat.max()+.01)}
    kept = []
    scanned = 0
    cuts = {}
    for chunk in pd.read_csv(SOURCE, usecols=["fecha_corte", "codemp", "codluminaria",
                                                 "coordenada_x", "coordenada_y"],
                             dtype={"codemp": "string", "codluminaria": "string"},
                             chunksize=250_000, low_memory=False, encoding_errors="replace"):
        scanned += len(chunk)
        for key, count in chunk["fecha_corte"].value_counts(dropna=False).items():
            cuts[str(key)] = cuts.get(str(key), 0) + int(count)
        x = pd.to_numeric(chunk["coordenada_x"], errors="coerce")
        y = pd.to_numeric(chunk["coordenada_y"], errors="coerce")
        local = x.between(bounds["lon_min"], bounds["lon_max"]) & y.between(bounds["lat_min"], bounds["lat_max"])
        if local.any():
            kept.append(chunk.loc[local, ["fecha_corte", "codemp", "codluminaria",
                                          "coordenada_x", "coordenada_y"]].copy())
        if scanned % 1_000_000 < 250_000:
            print(f"INVENTARIO {scanned:,} filas examinadas", flush=True)
    if not kept:
        raise ValueError("El archivo no contiene luminarias en la zona de la red vial")
    lights = pd.concat(kept, ignore_index=True)
    local_rows = len(lights)
    valid_id = lights["codemp"].notna() & lights["codluminaria"].notna()
    known = lights.loc[valid_id].sort_values("fecha_corte").drop_duplicates(
        ["codemp", "codluminaria"], keep="last")
    unknown = lights.loc[~valid_id].drop_duplicates(["coordenada_x", "coordenada_y"])
    lights = pd.concat([known, unknown], ignore_index=True)
    xy = lights[["coordenada_x", "coordenada_y"]].to_numpy(dtype=np.float64)
    tree = cKDTree(projected(xy))
    segment_xy = projected(static[:, [1, 0]])
    distance, _ = tree.query(segment_xy, k=1, workers=4)
    count50 = tree.query_ball_point(segment_xy, r=50, return_length=True, workers=4)
    count100 = tree.query_ball_point(segment_xy, r=100, return_length=True, workers=4)
    matrix = np.column_stack([np.log1p(count50), np.log1p(count100),
                              np.log1p(np.minimum(distance, 500)), distance <= 30]).astype(np.float32)
    if not np.isfinite(matrix).all() or len(matrix) != len(static):
        raise AssertionError("Variables de alumbrado inválidas")
    np.save(LIGHTS, matrix)
    stat = SOURCE.stat()
    save_json(AUDIT, {"source": str(SOURCE), "source_size": stat.st_size,
                      "source_mtime_ns": stat.st_mtime_ns, "rows_scanned": scanned,
                      "rows_in_network_bbox": local_rows, "unique_luminaires_in_bbox": len(lights),
                      "bbox": bounds, "fecha_corte_counts": cuts, "segment_count": len(static),
                      "segments_with_light_30m": int((distance <= 30).sum()),
                      "segments_with_light_100m": int((count100 > 0).sum()),
                      "nearest_distance_median_m": float(np.median(distance)),
                      "nearest_distance_p90_m": float(np.percentile(distance, 90)),
                      "variables": NAMES,
                      "assumption": "Inventario fijo para todos los meses 2018–2026; no se usa fecha de instalación ni estado operativo.",
                      "geometry": "Distancia euclidiana desde el punto central del tramo, aproximada en metros locales."})
    print(f"PREPARADO: {len(lights):,} luminarias locales; {(count100>0).sum():,} de {len(static):,} tramos con luminaria a 100 m", flush=True)


class LightingFeatures(Features):
    def __init__(self):
        super().__init__()
        self.light = np.load(LIGHTS)
        if self.light.shape != (self.n, len(NAMES)):
            raise AssertionError("El inventario de alumbrado no coincide con la red vial")

    def make(self, month, kind):
        if kind != "contexto_luz":
            return super().make(month, kind)
        base = super().make(month, "contexto")
        self.names[kind] = self.names["contexto"] + NAMES
        return np.column_stack([base, self.light]).astype(np.float32)


def specification():
    selected = json.loads(SELECTION.read_text(encoding="utf-8"))["selected"]
    if selected["name"] != "temporal_3m":
        raise ValueError("Cambió el modelo de referencia; revise el protocolo")
    return selected["spec"]


def month_result(features, training, spec, period):
    month = PERIODS.index(period)
    end = PERIODS[month-1]
    start = PERIODS[month-spec["window"]]
    training_spec = {**spec, "start": start}
    x, y, weight, details = training.get(training_spec, end)
    model = fit_model(training_spec, x, y, weight)
    truth = features.target([month])
    proba = predictions(model, features.make(month, spec["features"]))
    if not np.isfinite(proba).all() or not np.allclose(proba.sum(axis=1), 1, atol=1e-5):
        raise AssertionError(f"Probabilidades inválidas: {period}")
    metrics = evaluate(truth, proba)
    return {"periodo": period, "train": details, "metrics": metrics}, truth, proba


def aggregate(rows):
    matrices = [np.asarray(row["metrics"]["matriz_confusion"]) for row in rows]
    return matrix_metrics(np.sum(matrices, axis=0))


def run():
    if not LIGHTS.exists():
        raise ValueError("Ejecute primero prepare")
    OUT.mkdir(parents=True, exist_ok=True)
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    base_spec = specification()
    features = LightingFeatures()
    features.verify_causality()
    training = TrainingData(features)
    records = {}
    periods = VALIDATION + TEST + ADDITIONAL
    for period in periods:
        path = OUT / f"resultado_{period}.json"
        if path.exists():
            records[period] = json.loads(path.read_text(encoding="utf-8"))
            continue
        light_spec = {**base_spec, "features": "contexto_luz"}
        row, truth, proba = month_result(features, training, light_spec, period)
        reference_metrics = None
        if period in VALIDATION:
            validation = json.loads(SELECTION.read_text(encoding="utf-8"))["selected"]["months"]
            reference_metrics = next(item["metrics"] for item in validation if item["periodo"] == period)
        else:
            evaluation_name = "benchmark" if period in TEST else "additional_evaluation"
            reference_months = reference["evaluations"][evaluation_name]["monthly"]
            reference_metrics = next(item["metrics"] for item in reference_months if item["periodo"] == period)
        if row["metrics"]["support"] != reference_metrics["support"]:
            raise AssertionError(f"Las etiquetas cambiaron para {period}")
        row["reference_metrics"] = reference_metrics
        save_json(path, row)
        np.savez_compressed(OUT / f"probabilidades_{period}.npz", truth=truth, proba=proba)
        records[period] = row
        print(f"ALUMBRADO {period}: accuracy={row['metrics']['accuracy']:.4f} "
              f"vs {reference_metrics['accuracy']:.4f}, F1MA={row['metrics']['f1_medio_alto']:.4f} "
              f"vs {reference_metrics['f1_medio_alto']:.4f}", flush=True)
    grouped = {}
    for label, months in (("validation", VALIDATION), ("benchmark", TEST), ("additional_evaluation", ADDITIONAL)):
        rows = [records[m] for m in months]
        ours = aggregate(rows)
        baseline = matrix_metrics(np.sum([np.asarray(r["reference_metrics"]["matriz_confusion"]) for r in rows], axis=0))
        grouped[label] = {"periods": months, "alumbrado": ours, "sin_alumbrado": baseline,
                          "score_delta": score(ours)-score(baseline)}
    save_json(OUT / "final_results.json", {"model": base_spec["name"],
              "only_change": "Cuatro variables fijas de proximidad/cantidad de luminarias",
              "static_lighting_for_all_years": True, "groups": grouped,
              "selection_uses_2026": False,
              "warning": "La prueba de 2026 es retrospectiva; el inventario actual se asume estático en todo 2018–2026."})
    print("EVALUACION DE ALUMBRADO COMPLETA", flush=True)


def report():
    result = json.loads((OUT / "final_results.json").read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    fields = [("Accuracy", "accuracy", None), ("Precisión medio", "precision", 1),
              ("Detección medio", "recall", 1), ("F1 medio", "f1", 1),
              ("Precisión alto", "precision", 2), ("Detección alto", "recall", 2),
              ("F1 alto", "f1", 2), ("F1 medio+alto", "f1_medio_alto", None)]
    lines = ["# Alumbrado fijo en el modelo mensual con buffer 300 m", "",
             f"Inventario: {audit['rows_scanned']:,} filas; {audit['unique_luminaires_in_bbox']:,} luminarias "
             f"con coordenadas en el área de la red. {audit['segments_with_light_100m']:,} "
             f"de {audit['segment_count']:,} tramos tienen una luminaria a 100 m de su punto central.", "",
             "Se usa el mismo inventario en todos los meses de 2018–2026, según el supuesto fijado para este experimento. "
             "La fecha de instalación y el estado operativo no intervienen. La fuente es una fotografía del inventario y "
             "no demuestra que cada luminaria existiera o funcionara en años anteriores.", "",
             "Solo cambian cuatro atributos: cantidad a 50 y 100 m, distancia a la luminaria más próxima y presencia a 30 m. "
             "El resto del modelo es `temporal_3m`: misma fuente de delitos, etiquetas, entrenamiento de tres meses, "
             "buffer de 300 m, parámetros, muestreo y periodos.", ""]
    for group, title in (("validation", "Validación noviembre–diciembre de 2025"),
                         ("benchmark", "Evaluación enero–mayo de 2026"),
                         ("additional_evaluation", "Evaluación junio–agosto de 2026")):
        a = result["groups"][group]["sin_alumbrado"]
        b = result["groups"][group]["alumbrado"]
        lines += [f"## {title}", "", "| Métrica | Sin alumbrado | Con alumbrado fijo | Cambio (puntos) |",
                  "|---|---:|---:|---:|"]
        for label, key, i in fields:
            av, bv = (a[key], b[key]) if i is None else (a[key][i], b[key][i])
            lines.append(f"| {label} | {pct(av)} | {pct(bv)} | {(bv-av)*100:+.2f} |")
        lines += [""]
    lines += ["## Decisión", "",
              "Con el criterio de selección fijado antes de esta prueba (40 % accuracy y 60 % F1 medio+alto), "
              "la versión con alumbrado pierde en la validación de 2025. Se conserva el modelo sin alumbrado "
              "como referencia seleccionada. El aumento de accuracy de 0,13 puntos en enero–mayo de 2026 "
              "convive con una caída de 0,04 puntos en F1 medio+alto; es una diferencia pequeña y mixta.", "",
              "La prueba de 2026 ya había sido consultada en ensayos anteriores. Estas cifras son retrospectivas "
              "y no verifican cuánto alumbrado real había en cada mes. El objetivo sigue siendo riesgo por tramo y mes, "
              "no por turno.", ""]
    (OUT / "informe.md").write_text("\n".join(lines), encoding="utf-8")
    a = result["groups"]["benchmark"]["sin_alumbrado"]
    b = result["groups"]["benchmark"]["alumbrado"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8), facecolor="white")
    render_matrix(axes[0], a, "Sin alumbrado")
    render_matrix(axes[1], b, "Con alumbrado fijo")
    fig.suptitle("Misma prueba y mismas etiquetas · buffer 300 m", y=.98, fontsize=18, weight="bold")
    fig.text(.5, .91, "Enero–mayo de 2026 · porcentajes por clase real", ha="center", fontsize=12)
    fig.subplots_adjust(top=.78, bottom=.14, left=.06, right=.98, wspace=.25)
    fig.savefig(OUT / "matrices_sin_con_alumbrado.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(OUT / "informe.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    {"prepare": prepare, "run": run, "report": report}[parser.parse_args().stage]()
