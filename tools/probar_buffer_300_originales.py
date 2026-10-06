"""Repite el ensayo mensual de 300 m con las coordenadas del Excel limpio.

Excluye coordenadas recuperadas o simuladas. Conserva la especificación
temporal_3m y evalúa noviembre–diciembre 2025 y enero–mayo 2026, sin buscar
hiperparámetros nuevos. Guarda matrices e información reproducible, sin informe.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from buffer300.data import PERIODS, THRESHOLD, DATA as MIXED_DATA, save_json
from buffer300.features import Features
from buffer300.modelos import TrainingData, fit_model, predictions, evaluate, matrix_metrics, CONFIG
from buffer300.report import render_matrix
from comparar_factores_67 import graph_data
from app.flujo_entrenamiento.riesgo import agregar_riesgo_base

SOURCE_DIR = ROOT / "Backend/data/procesados_sidpol_v2"
SOURCE = SOURCE_DIR / "delitos_geolocalizados.csv"
SOURCE_AUDIT = SOURCE_DIR / "auditoria_fuente.json"
CLEAN_EXCEL = ROOT / "outputs/delitos-filtrados/DELITOS 2018-2026 FILTRADOS Y LIMPIOS.xlsx"
DATA = ROOT / "Backend/data/experimentos_sidpol/buffer_300_originales_v1"
OUT = ROOT / "outputs/evaluacion-modelos-sidpol/buffer_300_originales_v1"
PANEL = DATA / "panel_mensual_300m_originales.npy"
PANEL_META = DATA / "panel_mensual_300m_originales.json"
SELECTION = ROOT / "outputs/evaluacion-modelos-sidpol/optimizacion_buffer_300_mensual_v2/selection.json"
OLD_RESULTS = ROOT / "outputs/evaluacion-modelos-sidpol/alumbrado_buffer_300_v1/final_results.json"
OLD_LIGHT = ROOT / "outputs/evaluacion-modelos-sidpol/alumbrado_buffer_300_v1"
VALIDATION = ("2025-11", "2025-12")
TEST = tuple(f"2026-{m:02d}" for m in range(1, 6))
LIGHT_NAMES = ("luminarias_50m_log", "luminarias_100m_log", "distancia_luminaria_log_m", "luminaria_30m")


def fingerprint():
    stat = SOURCE.stat()
    return {"source": str(SOURCE), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "weights_sha256": hashlib.sha256((ROOT / "Backend/app/services/pesos_delito.py").read_bytes()).hexdigest(),
            "radius_m": 300, "sigma_m": 100, "periods": list(PERIODS)}


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if Path(audit["source"]).resolve() != CLEAN_EXCEL.resolve():
        raise ValueError("El CSV no procede del Excel limpio con coordenadas originales")
    signature = fingerprint()
    if PANEL.exists() and PANEL_META.exists():
        metadata = json.loads(PANEL_META.read_text(encoding="utf-8"))
        if metadata["fingerprint"] != signature:
            raise ValueError("La fuente original cambió; cree otra versión del ensayo")
        print("PANEL ORIGINAL existente y compatible", flush=True)
        return metadata

    crimes = pd.read_csv(SOURCE, usecols=["periodo", "latitud", "longitud", "turno", "subtipo_delito", "modalidad"])
    if len(crimes) != audit["totals"]["geolocalizados"]:
        raise ValueError("El número de filas no coincide con la auditoría del Excel limpio")
    if not np.isfinite(crimes[["latitud", "longitud"]].to_numpy()).all():
        raise ValueError("Coordenadas originales incompletas")
    if not crimes["periodo"].isin(PERIODS).all():
        raise ValueError("Periodos fuera de enero 2018–agosto 2026")
    crimes = agregar_riesgo_base(crimes)
    segments, lines, transformer = graph_data()
    n = len(segments)
    del segments
    points = shapely.points(*transformer.transform(crimes.longitud.to_numpy(), crimes.latitud.to_numpy()))
    tree = STRtree(lines)
    weights = crimes.peso_delito.to_numpy(dtype=np.float32)
    grave = crimes.es_delito_grave.to_numpy(dtype=np.float32)
    category = crimes.categoria_delito.to_numpy()
    turn = crimes.turno.to_numpy()
    month_ids = pd.Categorical(crimes.periodo, categories=PERIODS).codes
    panel = np.lib.format.open_memmap(PANEL, mode="w+", dtype=np.float32, shape=(len(PERIODS), n, 11))
    mixed = np.load(MIXED_DATA / "panel_mensual_300m.npy", mmap_mode="r")
    monthly = []
    for month, period in enumerate(PERIODS):
        indexes = np.flatnonzero(month_ids == month)
        expected = audit["monthly"][period]["geolocalizados"]
        if len(indexes) != expected:
            raise ValueError(f"Filas originales diferentes de la auditoría en {period}")
        local, targets = tree.query(points[indexes], predicate="dwithin", distance=300.0)
        events = indexes[local]
        distance = shapely.distance(points[events], lines[targets])
        decay = np.exp(-.5 * (distance / 100.0) ** 2)
        panel[month, :, 0] = np.bincount(targets, minlength=n)
        panel[month, :, 1] = np.bincount(targets, weights=weights[events] * decay, minlength=n)
        panel[month, :, 2] = np.bincount(targets, weights=grave[events], minlength=n)
        for column, name in enumerate(("hurtos", "robos", "extorsiones", "homicidios"), 3):
            panel[month, :, column] = np.bincount(targets, weights=category[events] == name, minlength=n)
        for column, name in enumerate(("manana", "tarde", "noche", "madrugada"), 7):
            panel[month, :, column] = np.bincount(targets, weights=turn[events] == name, minlength=n)
        # La fuente original debe ser un subconjunto espacial de la fuente completada.
        if np.any(panel[month, :, 0] > mixed[month, :, 0]):
            raise AssertionError(f"Hay conteos nuevos al quitar coordenadas asignadas: {period}")
        if np.any(panel[month, :, 1] > mixed[month, :, 1] + .01):
            raise AssertionError(f"Cambió la escala de severidad o el buffer: {period}")
        row = {"periodo": period, "delitos_originales": len(indexes),
               "delitos_totales_limpios": audit["monthly"][period]["total"],
               "cobertura_original": audit["monthly"][period]["cobertura"],
               "pares_buffer": int(len(targets)), "tramos_con_senal": int(np.count_nonzero(panel[month, :, 0]))}
        monthly.append(row)
        print(f"PANEL ORIGINAL {period}: {len(indexes):,} delitos; cobertura={row['cobertura_original']:.1%}", flush=True)
        del local, targets, events, distance, decay
        if month % 6 == 0:
            panel.flush()
            gc.collect()
    panel.flush()
    metadata = {"fingerprint": signature, "source_excel": str(CLEAN_EXCEL), "source_audit": str(SOURCE_AUDIT),
                "definition": "Solo filas con coordenadas presentes en el Excel filtrado y limpio; no recuperadas ni simuladas",
                "records": len(crimes), "excluded_missing_coordinates": audit["totals"]["sin_coordenadas"],
                "shape": list(panel.shape), "monthly": monthly,
                "checks": {"original_source_counts_reconciled": True, "spatial_subset_of_completed_data": True}}
    save_json(PANEL_META, metadata)
    del panel, mixed, crimes, points, lines, tree
    gc.collect()
    return metadata


class OriginalFeatures(Features):
    def __init__(self, lighting=False):
        super().__init__(panel_path=PANEL)
        self.lighting = lighting
        self.light = np.load(MIXED_DATA / "alumbrado_estatico_v1.npy") if lighting else None
        if lighting and self.light.shape != (self.n, len(LIGHT_NAMES)):
            raise ValueError("El inventario de luminarias no coincide con los tramos")

    def make(self, month, kind):
        base = super().make(month, kind)
        if self.lighting and kind == "contexto":
            self.names[kind] += list(LIGHT_NAMES)
            return np.column_stack([base, self.light]).astype(np.float32)
        return base


def prior_predictions(period, lighting):
    if lighting:
        path = OLD_LIGHT / f"probabilidades_{period}.npz"
        with np.load(path) as record:
            return record["proba"], record["truth"]
    path = MIXED_DATA / "mensual_v2" / f"prediccion_{period}.npz"
    with np.load(path) as record:
        return record["probability"], record["truth"]


def run_arm(lighting, spec):
    name = "con_alumbrado" if lighting else "sin_alumbrado"
    features = OriginalFeatures(lighting=lighting)
    features.verify_causality()
    training = TrainingData(features)
    rows = []
    for period in VALIDATION + TEST:
        output = OUT / f"resultado_{name}_{period}.json"
        prediction_path = DATA / f"prediccion_{name}_{period}.npz"
        if output.exists() and prediction_path.exists():
            rows.append(json.loads(output.read_text(encoding="utf-8")))
            continue
        month = PERIODS.index(period)
        fit_spec = {**spec, "start": PERIODS[month-spec["window"]]}
        x, y, weight, details = training.get(fit_spec, PERIODS[month-1])
        print(f"ENTRENANDO ORIGINALES {name} {period}: {len(y):,} muestras; {details['first_month']}..{details['last_month']}", flush=True)
        model = fit_model(fit_spec, x, y, weight)
        truth = features.target([month])
        probability = predictions(model, features.make(month, spec["features"]))
        if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1, atol=1e-5):
            raise AssertionError("Probabilidades inválidas")
        metrics = evaluate(truth, probability)
        row = {"periodo": period, "training": details, "metrics": metrics}
        if period in TEST:
            old_probability, old_truth = prior_predictions(period, lighting)
            if np.any(truth > old_truth):
                raise AssertionError("Quitar delitos elevó una etiqueta de riesgo")
            row["previous_model_on_original_targets"] = evaluate(truth, old_probability)
            row["changed_targets_vs_previous"] = int(np.count_nonzero(truth != old_truth))
            del old_probability, old_truth
        save_json(output, row)
        np.savez_compressed(prediction_path, truth=truth, probability=probability)
        joblib.dump({"model": model, "spec": fit_spec, "training": details,
                     "high_threshold": THRESHOLD, "panel": str(PANEL), "source_origin": "original"},
                    DATA / f"modelo_{name}_{period}.joblib", compress=3)
        rows.append(row)
        print(f"RESULTADO ORIGINAL {name} {period}: acc={metrics['accuracy']:.4f}; alto P={metrics['precision'][2]:.4f}, R={metrics['recall'][2]:.4f}", flush=True)
        del model, probability, truth
        gc.collect()
    del features, training
    gc.collect()

    groups = {}
    for group, periods in (("validation", VALIDATION), ("benchmark", TEST)):
        members = [row for row in rows if row["periodo"] in periods]
        metric = matrix_metrics(sum(np.asarray(row["metrics"]["matriz_confusion"]) for row in members))
        labels, probabilities = [], []
        for period in periods:
            with np.load(DATA / f"prediccion_{name}_{period}.npz") as record:
                labels.append(record["truth"])
                probabilities.append(record["probability"][:, 2])
        metric["average_precision_alto"] = float(average_precision_score(np.concatenate(labels) == 2, np.concatenate(probabilities)))
        groups[group] = {"periods": list(periods), "metrics": metric, "monthly": members}
        if group == "benchmark":
            groups[group]["previous_model_on_original_targets"] = matrix_metrics(sum(
                np.asarray(row["previous_model_on_original_targets"]["matriz_confusion"]) for row in members))
            groups[group]["changed_targets_vs_previous"] = sum(row["changed_targets_vs_previous"] for row in members)
    return groups


def main():
    metadata = prepare()
    selected = json.loads(SELECTION.read_text(encoding="utf-8"))["selected"]
    if selected["name"] != "temporal_3m":
        raise ValueError("La especificación de referencia cambió")
    protocol = {"source": str(SOURCE), "source_excel": str(CLEAN_EXCEL), "radius_m": 300,
                "sigma_m": 100, "high_threshold": THRESHOLD, "candidate": selected["spec"],
                "model_parameters": "Mismos parámetros de temporal_3m y muestreo del ensayo anterior",
                "training_config": CONFIG, "validation": list(VALIDATION), "test": list(TEST),
                "training": "Reentrenar cada mes con los tres meses previos; las variables históricas usan hasta doce meses anteriores",
                "new_tuning": False, "coordinates": "originales presentes en el Excel limpio",
                "evaluation": "retrospectiva; 2026 ya fue examinado en ensayos previos"}
    save_json(OUT / "protocolo.json", protocol)
    results = {name: run_arm(light, selected["spec"])
               for name, light in (("sin_alumbrado", False), ("con_alumbrado", True))}
    previous = json.loads(OLD_RESULTS.read_text(encoding="utf-8"))["groups"]["benchmark"]
    save_json(OUT / "final_results.json", {"protocol": protocol, "source_metadata": metadata,
              "results": results, "previous_on_completed_targets": previous})

    fig, axes = plt.subplots(1, 3, figsize=(22, 8), facecolor="white")
    render_matrix(axes[0], results["sin_alumbrado"]["benchmark"]["previous_model_on_original_targets"],
                  "Modelo anterior\nevaluado con originales")
    render_matrix(axes[1], results["sin_alumbrado"]["benchmark"]["metrics"], "Entrenado con originales\nsin alumbrado")
    render_matrix(axes[2], results["con_alumbrado"]["benchmark"]["metrics"], "Entrenado con originales\ncon alumbrado fijo")
    fig.suptitle("Buffer 300 m · Misma prueba y etiquetas con ubicaciones originales", fontsize=21, weight="bold")
    fig.text(.5, .925, "Enero–mayo de 2026 · tramo × mes · porcentajes por clase real", ha="center", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .89), w_pad=3)
    image = OUT / "matrices_solo_originales_300m.png"
    fig.savefig(image, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"IMAGEN {image}", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="white")
    for ax, name, title in zip(axes, ("sin_alumbrado", "con_alumbrado"),
                               ("Sin alumbrado", "Con alumbrado fijo")):
        render_matrix(ax, results[name]["benchmark"]["metrics"], title)
    fig.suptitle("Buffer 300 m · Entrenamiento y evaluación con ubicaciones originales", fontsize=18, weight="bold")
    fig.text(.5, .925, "Enero–mayo de 2026 · tramo × mes · porcentajes por clase real", ha="center", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .89), w_pad=3)
    original_image = OUT / "matrices_entrenamiento_solo_originales_300m.png"
    fig.savefig(original_image, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"IMAGEN ORIGINALES {original_image}", flush=True)
    for name, arm in results.items():
        metric = arm["benchmark"]["metrics"]
        print(f"FINAL {name}: accuracy={metric['accuracy']:.6f}; recall_medio={metric['recall'][1]:.6f}; "
              f"recall_alto={metric['recall'][2]:.6f}; precision_alto={metric['precision'][2]:.6f}", flush=True)


if __name__ == "__main__":
    main()
