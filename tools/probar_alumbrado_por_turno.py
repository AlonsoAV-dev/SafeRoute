"""Ablación de alumbrado fijo sobre el modelo XGBoost por tramo, mes y turno.

Reutiliza el muestreo y las matrices temporales del experimento completo
2018–2026. Entrena hasta diciembre de 2025; calibra en enero–febrero de 2026,
elige la regla en marzo–abril y evalúa mayo–agosto de 2026.
"""

from __future__ import annotations

import argparse
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
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "Backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT / "tools"))

from app.flujo_entrenamiento_sidpol.config import TrainingConfig
from app.flujo_entrenamiento_sidpol.optimizacion.evaluacion import (
    ProbabilityCalibration, metrics, predict_levels, tune_joint_thresholds,
)
from app.flujo_entrenamiento_sidpol.optimizacion.modelos import create_model
from app.flujo_entrenamiento_sidpol.optimizacion.variables import WindowFeatures
from app.flujo_entrenamiento_sidpol.segmentacion import cargar_matrices
from buffer300.data import save_json
from buffer300.report import render_matrix

BASE = BACKEND / "data/experimentos_sidpol/entrenamiento_completo_2018_2026"
CACHE = BASE / "cache"
OUTPUT = ROOT / "outputs/evaluacion-modelos-sidpol/alumbrado_turnos_v1"
MODEL_DIR = BACKEND / "data/experimentos_sidpol/alumbrado_turnos_v1"
LIGHT = BACKEND / "data/experimentos_sidpol/buffer_300_v1/alumbrado_estatico_v1.npy"
ORIGINAL = BASE / "evaluacion_temporal.json"
CONFIG = BACKEND / "config_experimento_sidpol_completo_2018_2026.json"
MODEL = MODEL_DIR / "modelo_evaluado_xgb_multi_luz.joblib"
INDEX = MODEL_DIR / "indice_alumbrado_muestra.npy"
TURNS = ("madrugada", "mañana", "tarde", "noche")


def setup():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not LIGHT.is_file():
        raise FileNotFoundError("Primero prepare el alumbrado fijo con probar_alumbrado_buffer_300.py")
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))
    spec = next(item for item in settings["candidates"] if item["name"] == "xgb_multi_recent")
    cfg = TrainingConfig.load((BACKEND / settings["source_config"]).resolve())
    tramos = pd.read_csv(cfg.output / "tramos_osm.csv")
    audit = json.loads((cfg.output / "auditoria_fuente.json").read_text(encoding="utf-8"))
    builder = WindowFeatures(tramos, cargar_matrices(cfg.output), cfg, audit,
                             settings["windows"], settings["minimum_history_coverage"], settings["high_threshold"])
    light = np.load(LIGHT, mmap_mode="r")
    if light.shape != (builder.n_segments, 4):
        raise AssertionError("La red de tramos no coincide con el inventario de alumbrado")
    base_ids = pd.read_csv(BACKEND / "data/procesados/evaluacion_2026/tramos_osm.csv", usecols=["tramo_id"])["tramo_id"]
    if not base_ids.equals(tramos["tramo_id"]):
        raise AssertionError("El orden de tramos cambió entre experimentos")
    return settings, spec, builder, tramos, light


def training_indices(tramos, light):
    if INDEX.exists():
        index = np.load(INDEX, mmap_mode="r")
        if len(index) == len(np.load(CACHE / "train_2018_2025_x.npy", mmap_mode="r")):
            return index
        raise AssertionError("El índice de la muestra no coincide con el entrenamiento")
    source = np.load(CACHE / "train_2018_2025_x.npy", mmap_mode="r")
    coord = tramos[["longitud", "latitud"]].to_numpy(dtype=np.float32)
    meter = np.array([111_320 * np.cos(np.deg2rad(-12)), 111_320], dtype=np.float64)
    tree = cKDTree(coord * meter)
    index = np.lib.format.open_memmap(INDEX, mode="w+", dtype=np.int32, shape=(len(source),))
    for left in range(0, len(source), 250_000):
        right = min(left + 250_000, len(source))
        queried = source[left:right, :2][:, [1, 0]].astype(np.float64) * meter
        distance, matched = tree.query(queried, workers=4)
        if distance.max() > .05:
            raise AssertionError(f"No se pudo alinear la muestra con los tramos: {distance.max():.3f} m")
        index[left:right] = matched
    index.flush()
    del source
    print(f"Muestra enlazada con {len(index):,} tramos-mes-turno; distancia máxima < 5 cm", flush=True)
    return np.load(INDEX, mmap_mode="r")


def train():
    if MODEL.exists():
        print(f"Modelo ya entrenado: {MODEL}", flush=True)
        return
    settings, spec, builder, tramos, light = setup()
    index = training_indices(tramos, light)
    source = np.load(CACHE / "train_2018_2025_x.npy", mmap_mode="r")
    metadata = np.load(CACHE / "train_2018_2025_metadata.npz")
    columns = builder.groups["multi"]
    n = len(source)
    x = np.empty((n, len(columns) + light.shape[1]), dtype=np.float32)
    for left in range(0, n, 200_000):
        right = min(left + 200_000, n)
        x[left:right, :-4] = source[left:right, columns]
        x[left:right, -4:] = light[index[left:right]]
    weight = metadata["weights"] ** spec["weight_power"]
    age = max(metadata["months"]) - metadata["months"]
    weight *= np.exp2(-age / spec["half_life_months"])
    weight /= weight.mean()
    y = metadata["y"]
    model = create_model(spec, settings["threads"], settings["random_state"])
    print(f"ENTRENANDO {spec['name']} con alumbrado: {n:,} filas × {x.shape[1]} variables", flush=True)
    model.fit(x, y, sample_weight=weight)
    joblib.dump(model, MODEL, compress=3)
    source_config = TrainingConfig.load((BACKEND / settings["source_config"]).resolve())
    save_json(OUTPUT / "protocolo.json", {"source": str(source_config.output),
              "base_model": str(BASE / "modelo_evaluado_xgb_multi_recent.joblib"),
              "light_source": str(LIGHT), "model": str(MODEL), "spec": spec,
              "training": "2018-01 a 2025-12", "calibration": ["2026-01", "2026-02"],
              "selection": ["2026-03", "2026-04"], "test": ["2026-05", "2026-06", "2026-07", "2026-08"],
              "static_light": True, "light_features": ["log luces 50 m", "log luces 100 m", "log distancia mínima", "luz a 30 m"],
              "target": "riesgo por tramo × mes × turno; snapping 150 m; alto si gravedad normalizada >=3"})
    print(MODEL, flush=True)
    metadata.close()
    del x, y, weight, model, source, index
    gc.collect()


def stage_probability(model, stage, columns, light, augmented):
    x = np.load(CACHE / f"{stage}_x.npy", mmap_mode="r")
    nseg = len(light)
    probability = np.empty((len(x), 3), dtype=np.float32)
    for left in range(0, len(x), nseg):
        right = min(left+nseg, len(x))
        part = np.ascontiguousarray(x[left:right, columns])
        if augmented:
            part = np.column_stack([part, light[np.arange(left, right) % nseg]]).astype(np.float32)
        probability[left:right] = model.predict_proba(part)
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1, atol=1e-5):
        raise AssertionError(f"Probabilidades inválidas para {stage}")
    return probability


def matrix_statistics(matrix):
    m = np.asarray(matrix, dtype=np.int64)
    support = m.sum(axis=1)
    predicted = m.sum(axis=0)
    diagonal = m.diagonal()
    precision = np.divide(diagonal, predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2*diagonal, support+predicted, out=np.zeros(3), where=(support+predicted) > 0)
    return {"accuracy": float(diagonal.sum()/m.sum()), "precision": precision.tolist(),
            "recall": recall.tolist(), "f1": f1.tolist(), "f1_medio_alto": float(f1[1:].mean()),
            "support": support.tolist(), "matriz_confusion": m.tolist()}


def by_turn(truth, probability, cutoffs, nseg):
    pred = predict_levels(probability, cutoffs)
    result = {}
    for turn, label in enumerate(TURNS):
        matrix = np.zeros((3, 3), dtype=np.int64)
        for offset in range(turn*nseg, len(truth), 4*nseg):
            y = truth[offset:offset+nseg].astype(np.int64)
            p = pred[offset:offset+nseg].astype(np.int64)
            matrix += np.bincount(y*3+p, minlength=9).reshape(3, 3)
        result[label] = matrix_statistics(matrix)
    return result


def evaluate():
    if (OUTPUT / "final_results.json").exists():
        print("Evaluación por turno ya completada", flush=True)
        return
    if not MODEL.exists():
        raise FileNotFoundError("Ejecute train antes de evaluate")
    settings, spec, builder, _, light = setup()
    model = joblib.load(MODEL)
    columns = builder.groups["multi"]
    truth_cal = np.load(CACHE / "calibration_y.npy", mmap_mode="r")
    truth_sel = np.load(CACHE / "selection_y.npy", mmap_mode="r")
    truth_test = np.load(CACHE / "retrospective_y.npy", mmap_mode="r")
    p_cal = stage_probability(model, "calibration", columns, light, True)
    p_sel = stage_probability(model, "selection", columns, light, True)
    calibrator = ProbabilityCalibration().fit(truth_cal, p_cal, settings["random_state"])
    joblib.dump(calibrator, MODEL_DIR / "calibrador_evaluado_xgb_multi_luz.joblib", compress=3)
    alternatives = {}
    for variant in ("raw", "calibrated"):
        pc = p_cal if variant == "raw" else calibrator.predict_proba(p_cal)
        ps = p_sel if variant == "raw" else calibrator.predict_proba(p_sel)
        cutoffs = tune_joint_thresholds(truth_cal, pc)
        selection = metrics(truth_sel, ps, cutoffs)
        alternatives[variant] = {"cutoffs": cutoffs, "selection": selection}
        print(f"VALIDACIÓN {variant}: F1 medio+alto={selection['f1_medio_alto']:.4f}", flush=True)
        del pc, ps
        gc.collect()
    chosen = max(alternatives, key=lambda name: alternatives[name]["selection"]["f1_medio_alto"])
    cutoffs = alternatives[chosen]["cutoffs"]
    del p_cal, p_sel
    gc.collect()
    p_test = stage_probability(model, "retrospective", columns, light, True)
    if chosen == "calibrated":
        p_test = calibrator.predict_proba(p_test)
    augmented = metrics(truth_test, p_test, cutoffs)
    augmented_turns = by_turn(truth_test, p_test, cutoffs, builder.n_segments)
    print(f"PRUEBA con alumbrado: F1 medio+alto={augmented['f1_medio_alto']:.4f}", flush=True)
    baseline_result = json.loads(ORIGINAL.read_text(encoding="utf-8"))["candidates"]["xgb_multi_recent"]
    base_model = joblib.load(BASE / "modelo_evaluado_xgb_multi_recent.joblib")
    base_p = stage_probability(base_model, "retrospective", columns, light, False)
    base_cutoffs = baseline_result["variants"][baseline_result["selected_variant"]]["cutoffs"]
    if baseline_result["selected_variant"] == "calibrated":
        base_calibrator = joblib.load(BASE / "calibrador_evaluado_xgb_multi_recent.joblib")
        base_p = base_calibrator.predict_proba(base_p)
    baseline = metrics(truth_test, base_p, base_cutoffs)
    if baseline["matriz_confusion"] != baseline_result["test"]["matriz_confusion"]:
        raise AssertionError("No se reprodujo la matriz original sin alumbrado")
    baseline_turns = by_turn(truth_test, base_p, base_cutoffs, builder.n_segments)
    save_json(OUTPUT / "final_results.json", {"reference": "xgb_multi_recent; mismo muestreo y periodos",
              "selected_variant_with_light": chosen, "alternatives_with_light": alternatives,
              "baseline_selected_variant": baseline_result["selected_variant"],
              "baseline_cutoffs": base_cutoffs, "test": {"without_light": baseline,
              "with_light": augmented, "by_turn_without_light": baseline_turns,
              "by_turn_with_light": augmented_turns},
              "static_lighting_for_all_years": True,
              "test_used_for_selection": False})
    print("EVALUACIÓN POR TURNO COMPLETA", flush=True)


def report():
    result = json.loads((OUTPUT / "final_results.json").read_text(encoding="utf-8"))
    base = result["test"]["without_light"]
    light = result["test"]["with_light"]
    historical = json.loads(ORIGINAL.read_text(encoding="utf-8"))["candidates"]["xgb_multi_recent"]
    baseline_validation = historical["variants"][historical["selected_variant"]]["selection"]["f1_medio_alto"]
    light_validation = result["alternatives_with_light"][result["selected_variant_with_light"]]["selection"]["f1_medio_alto"]
    def row(name, a, b):
        return f"| {name} | {a*100:.2f} % | {b*100:.2f} % | {(b-a)*100:+.2f} |"
    lines = ["# Alumbrado fijo en la predicción por turno", "",
             "Modelo: XGBoost de ventanas (`xgb_multi_recent`) sobre delitos asignados por snapping a un tramo "
             "dentro de 150 m. La unidad es tramo × mes × turno. Ambos brazos usan la misma base 2018–2026, "
             "las mismas etiquetas y la misma muestra. Se entrenó hasta diciembre de 2025, se calibró en "
             "enero–febrero de 2026, se seleccionó la regla en marzo–abril y se evaluó mayo–agosto de 2026.", "",
             "El alumbrado se mantiene fijo para todos los años. Se añaden cantidad de luminarias próximas "
             "y distancia al punto central del tramo; no se conoce si estaban encendidas o funcionando.", "",
             f"En la selección de marzo–abril de 2026, el F1 medio+alto fue {baseline_validation*100:.2f} % "
             f"sin alumbrado y {light_validation*100:.2f} % con alumbrado.", "",
             "## Prueba total de mayo–agosto de 2026", "",
             "| Métrica | Sin alumbrado | Con alumbrado fijo | Cambio (puntos) |", "|---|---:|---:|---:|"]
    lines += [row("Accuracy", base["accuracy"], light["accuracy"]),
              row("F1 medio+alto", base["f1_medio_alto"], light["f1_medio_alto"])]
    for key, index in (("medio", 1), ("alto", 2)):
        a, b = base["clases"][key], light["clases"][key]
        lines += [row(f"Precisión {key}", a["precision"], b["precision"]),
                  row(f"Detección {key}", a["recall"], b["recall"]),
                  row(f"F1 {key}", a["f1"], b["f1"]),
                  row(f"AP {key}", a["average_precision"], b["average_precision"])]
    lines += ["", "## Resultado por turno", "",
              "| Turno | F1 medio+alto sin | F1 medio+alto con | Detección alto sin | Detección alto con | Precisión alto sin | Precisión alto con |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label in TURNS:
        a = result["test"]["by_turn_without_light"][label]
        b = result["test"]["by_turn_with_light"][label]
        lines.append(f"| {label} | {a['f1_medio_alto']*100:.2f} % | {b['f1_medio_alto']*100:.2f} % | "
                     f"{a['recall'][2]*100:.2f} % | {b['recall'][2]*100:.2f} % | "
                     f"{a['precision'][2]*100:.2f} % | {b['precision'][2]*100:.2f} % |")
    lines += ["", "## Interpretación", "",
              "El F1 medio+alto mejora ligeramente, pero baja la detección de alto en los cuatro turnos "
              "y la AP de medio y alto también disminuye. Esta evidencia no justifica reemplazar automáticamente "
              "el modelo por turnos sin alumbrado si se busca equilibrar precisión y detección de riesgo alto.", "",
              "La clase baja domina casi todas las unidades; el accuracy aislado puede verse alto "
              "aunque se detecten pocos casos medios y altos. Las pruebas de 2026 son retrospectivas y "
              "ya se habían examinado en otros ensayos. La comparación por turno usa snapping y un umbral "
              "diferente al modelo mensual con buffer, por lo que sus cifras no se comparan directamente.", ""]
    (OUTPUT / "informe.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8), facecolor="white")
    render_matrix(axes[0], matrix_statistics(base["matriz_confusion"]), "Sin alumbrado")
    render_matrix(axes[1], matrix_statistics(light["matriz_confusion"]), "Con alumbrado fijo")
    fig.suptitle("Predicción por tramo × mes × turno", y=.98, fontsize=18, weight="bold")
    fig.text(.5, .91, "Prueba mayo–agosto de 2026 · porcentajes por clase real", ha="center", fontsize=12)
    fig.subplots_adjust(top=.78, bottom=.14, left=.06, right=.98, wspace=.25)
    fig.savefig(OUTPUT / "matrices_sin_con_alumbrado_turnos.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(OUTPUT / "informe.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "evaluate", "report"))
    {"train": train, "evaluate": evaluate, "report": report}[parser.parse_args().stage]()
