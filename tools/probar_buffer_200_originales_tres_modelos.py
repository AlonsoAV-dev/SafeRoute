"""Buffer 200 m sin alumbrado: XGBoost, LSTM y Random Forest por turno.

Ensayo independiente con coordenadas originales e historial desde 2018.
Ejecutar: .venv/Scripts/python tools/probar_buffer_200_originales_tres_modelos.py
"""
from __future__ import annotations

import functools
import gc
import json
import shutil
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
from sklearn.ensemble import RandomForestClassifier
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from comparar_factores_67 import graph_data
from app.flujo_entrenamiento.riesgo import agregar_riesgo_base
from buffer300.data import PERIODS, THRESHOLD, save_json
from buffer300.features import Features
from buffer300.modelos import fit_model, predictions, evaluate, matrix_metrics
from buffer300.report import render_matrix
from probar_buffer_300_originales import (
    SOURCE, SOURCE_AUDIT, PANEL as PREVIOUS_MONTHLY, PANEL_META as PREVIOUS_META, fingerprint,
)
import probar_alumbrado_turnos_originales_xgb_lstm as previous

DATA = ROOT / "Backend/data/experimentos_sidpol/buffer_200_originales_tres_modelos_v1"
OUT = ROOT / "outputs/evaluacion-modelos-sidpol/buffer_200_originales_tres_modelos_v1"
MONTHLY_PANEL = DATA / "panel_mensual_200m_originales.npy"
TURN_PANEL = DATA / "panel_turnos_200m_originales.npy"
META = DATA / "paneles_200m_originales.json"
TURNS, TURN_COLUMNS = previous.TURNS, previous.TURN_COLUMNS
VALIDATION, TEST = previous.VALIDATION, previous.TEST
SPEC = {**previous.SPEC, "name": "contexto_historia_original_buffer200_por_turno"}
RF_PARAMETERS = {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 20,
                 "min_samples_split": 40, "max_features": "sqrt", "bootstrap": True,
                 "max_samples": .8, "class_weight": None, "random_state": 42, "n_jobs": 8}
VARIANT = "sin_alumbrado"


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    original_signature = fingerprint()
    original_meta = json.loads(PREVIOUS_META.read_text(encoding="utf-8"))
    if original_meta["fingerprint"] != original_signature:
        raise ValueError("El panel de referencia no coincide con la fuente original")
    signature = {**original_signature, "radius_m": 200, "turn_from_event_hour": True}
    if META.exists() and MONTHLY_PANEL.exists() and TURN_PANEL.exists():
        meta = json.loads(META.read_text(encoding="utf-8"))
        if meta["fingerprint"] != signature:
            raise ValueError("Cambió la fuente; cree otra versión del experimento")
        print("PANELES ORIGINALES DE 200 m YA PREPARADOS", flush=True)
        return meta
    columns = ["periodo", "hora", "latitud", "longitud", "turno", "subtipo_delito", "modalidad"]
    crimes = pd.read_csv(SOURCE, usecols=columns)
    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if len(crimes) != audit["totals"]["geolocalizados"]:
        raise ValueError("La fuente no coincide con la auditoría de coordenadas originales")
    hour = pd.to_numeric(crimes.hora, errors="raise").to_numpy()
    if not np.isfinite(crimes[["latitud", "longitud"]].to_numpy()).all() or not ((hour >= 0) & (hour < 24)).all():
        raise ValueError("Coordenadas u horas inválidas")
    turn_codes = (hour // 6).astype(np.int8)
    corrections = np.asarray(TURNS)[turn_codes] != crimes.turno.to_numpy()
    corrected_by_month = crimes.loc[corrections, "periodo"].value_counts().sort_index().to_dict()
    crimes = agregar_riesgo_base(crimes)
    tramos, lines, transformer = graph_data()
    n = len(tramos)
    del tramos
    required = len(PERIODS)*n*(11+4*2)*4+400_000_000
    if shutil.disk_usage(DATA).free < required:
        raise OSError("Espacio insuficiente para los paneles y resultados del experimento")
    points = shapely.points(*transformer.transform(crimes.longitud.to_numpy(), crimes.latitud.to_numpy()))
    tree = STRtree(lines)
    month_codes = pd.Categorical(crimes.periodo, categories=PERIODS).codes
    if np.any(month_codes < 0):
        raise ValueError("Hay meses fuera de enero 2018–agosto 2026")
    weights = crimes.peso_delito.to_numpy(dtype=np.float32)
    grave = crimes.es_delito_grave.to_numpy(dtype=np.float32)
    categories = crimes.categoria_delito.to_numpy()
    reference = np.load(PREVIOUS_MONTHLY, mmap_mode="r")
    if n != reference.shape[1]:
        raise AssertionError("El orden de segmentos no coincide con la red de referencia")
    monthly = np.lib.format.open_memmap(MONTHLY_PANEL, mode="w+", dtype=np.float32,
                                       shape=(len(PERIODS), n, 11))
    turns = np.lib.format.open_memmap(TURN_PANEL, mode="w+", dtype=np.float32,
                                     shape=(len(PERIODS), 4, n, 2))
    rows = []
    for month, period in enumerate(PERIODS):
        indexes = np.flatnonzero(month_codes == month)
        if len(indexes) != audit["monthly"][period]["geolocalizados"]:
            raise ValueError(f"La fuente no se reconcilia con la auditoría: {period}")
        local, segments = tree.query(points[indexes], predicate="dwithin", distance=200.)
        events = indexes[local]
        distance = shapely.distance(points[events], lines[segments])
        weighted = weights[events]*np.exp(-.5*(distance/100.)**2)
        monthly[month, :, 0] = np.bincount(segments, minlength=n)
        monthly[month, :, 1] = np.bincount(segments, weights=weighted, minlength=n)
        monthly[month, :, 2] = np.bincount(segments, weights=grave[events], minlength=n)
        for column, category in enumerate(("hurtos", "robos", "extorsiones", "homicidios"), 3):
            monthly[month, :, column] = np.bincount(segments, weights=categories[events] == category, minlength=n)
        for turn, column in enumerate(TURN_COLUMNS):
            chosen = turn_codes[events] == turn
            counts = np.bincount(segments[chosen], minlength=n)
            monthly[month, :, column] = counts
            turns[month, turn, :, 0] = counts
            turns[month, turn, :, 1] = np.bincount(segments[chosen], weights=weighted[chosen], minlength=n)
        if not np.array_equal(turns[month, :, :, 0].sum(axis=0), monthly[month, :, 0]):
            raise AssertionError(f"Los conteos por turno no reproducen el mes {period}")
        if np.max(np.abs(turns[month, :, :, 1].sum(axis=0)-monthly[month, :, 1])) > .001:
            raise AssertionError(f"La gravedad por turno no reproduce el mes {period}")
        if np.any(monthly[month, :, 0] > reference[month, :, 0]) or np.any(monthly[month, :, 1] > reference[month, :, 1]+.01):
            raise AssertionError(f"El buffer de 200 m no es subconjunto del de 300 m: {period}")
        rows.append({"periodo": period, "delitos_originales": len(indexes), "pares_delito_tramo": len(segments),
                     "tramos_con_senal": int(np.count_nonzero(monthly[month, :, 0]))})
        if month % 6 == 5 or month == len(PERIODS)-1:
            monthly.flush()
            turns.flush()
            print(f"PANEL 200 m {period}: {len(indexes):,} delitos; controles correctos", flush=True)
    monthly.flush()
    turns.flush()
    meta = {"fingerprint": signature, "source_excel": original_meta["source_excel"],
            "only_original_coordinates": True, "original_crimes": len(crimes),
            "excluded_missing_coordinates": audit["totals"]["sin_coordenadas"],
            "monthly_shape": list(monthly.shape), "turn_shape": list(turns.shape), "monthly": rows,
            "turn_rule": "Hora del hecho: madrugada 00–05, mañana 06–11, tarde 12–17, noche 18–23",
            "turns_corrected_from_event_hour": int(corrections.sum()), "turns_corrected_by_month": corrected_by_month,
            "checks": {"original_counts_reconciled": True, "turns_reproduce_monthly": True,
                       "buffer200_is_subset_of_buffer300": True}}
    save_json(META, meta)
    del monthly, turns, reference, crimes, points, tree, lines
    gc.collect()
    return meta


class Buffer200Data(previous.TurnData):
    def __init__(self):
        self.monthly = Features(panel_path=MONTHLY_PANEL)
        self.panel = np.load(TURN_PANEL, mmap_mode="r")
        self.factor, self.static, self.n = self.monthly.factor, self.monthly.static, self.monthly.n
        self.neighbors = self.monthly._neighbors()
        self.feature_cache = {}

    def lighting(self, *args):
        raise AssertionError("Este experimento excluye el alumbrado")

    @functools.lru_cache(maxsize=16)
    def temporal_monthly(self, month):
        raw = np.maximum(self.monthly.panel[month]/self.factor[:, None], 0)
        risk = raw[:, 1]
        return np.column_stack([np.log1p(raw), np.log1p(risk[self.neighbors[:, :8]].mean(axis=1)),
                                np.log1p(risk[self.neighbors].mean(axis=1)), np.ones(self.n)]).astype(np.float16)

    def sequence(self, month, turn, ids):
        result = np.zeros((len(ids), previous.WINDOW, previous.CHANNELS), dtype=np.float32)
        for prior in range(max(0, month-previous.WINDOW), month):
            position = previous.WINDOW-month+prior
            result[:, position, :14] = self.temporal_monthly(prior)[ids]
            result[:, position, 14:] = self.temporal_turn(prior, turn)[ids]
        return result

    def training_xgb(self, records, variant):
        if variant != VARIANT:
            raise ValueError("Solo se permite la variante sin alumbrado")
        key = (records[0][0], records[-1][0], sum(len(r[2]) for r in records))
        if key not in self.feature_cache:
            self.feature_cache[key] = super().training_xgb(records, variant)
        return self.feature_cache[key]


def fit_tabular(family, x, y, weights):
    if family == "xgboost":
        return fit_model(SPEC, x, y, weights)
    model = RandomForestClassifier(**RF_PARAMETERS)
    model.fit(x, y, sample_weight=weights)
    return model


def eval_periods(model, data, family, periods, scales=None, save=False):
    monthly, turns = [], {turn: np.zeros((3, 3), dtype=np.int64) for turn in TURNS}
    for period in periods:
        month = PERIODS.index(period)
        by_turn = {}
        for turn, name in enumerate(TURNS):
            truth = data.target(month, turn)
            if family == "lstm":
                probability = previous.lstm_predict(model, data, month, turn, VARIANT, scales)
            else:
                probability = predictions(model, data.xgb(month, turn, VARIANT, np.arange(data.n)))
            metrics = evaluate(truth, probability)
            by_turn[name] = metrics
            turns[name] += np.asarray(metrics["matriz_confusion"], dtype=np.int64)
            if save:
                np.savez_compressed(DATA / f"predicciones_{family}_{period}_{name}.npz", truth=truth, probability=probability)
        total = matrix_metrics(sum(np.asarray(m["matriz_confusion"]) for m in by_turn.values()))
        row = {"periodo": period, "metrics": total, "por_turno": by_turn}
        monthly.append(row)
        if save:
            save_json(OUT / f"resultado_{family}_{period}.json", row)
        print(f"EVAL 200 m {family} {period}: accuracy={total['accuracy']:.4f}; alto={total['recall'][2]:.4f}", flush=True)
    return {"monthly": monthly, "metrics": matrix_metrics(sum(turns.values())),
            "por_turno": {turn: matrix_metrics(matrix) for turn, matrix in turns.items()},
            "noche_madrugada": matrix_metrics(turns["noche"]+turns["madrugada"])}


def run(data, family):
    path = OUT / f"resultado_{family}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    records, y, weights, selection = data.selection("2025-10")
    previous.seed()
    print(f"ENTRENANDO 200 m {family}: {selection}", flush=True)
    epochs, history = None, None
    if family == "lstm":
        model, scales, history, chosen = previous.train_lstm(data, records, y, weights, VARIANT, previous.MAX_EPOCHS, True)
        epochs, validation = chosen["epoch"], chosen["validation"]
        del model, scales
    else:
        x = data.training_xgb(records, VARIANT)
        model = fit_tabular(family, x, y, weights)
        validation = eval_periods(model, data, family, VALIDATION)
        del model, x
    del records, y, weights
    gc.collect()
    records, y, weights, final_selection = data.selection("2025-12")
    previous.seed()
    print(f"AJUSTE FINAL 200 m {family}: {final_selection['sampled_rows']:,} muestras; epocas={epochs}", flush=True)
    if family == "lstm":
        model, scales, _, _ = previous.train_lstm(data, records, y, weights, VARIANT, epochs, False)
        torch.save({"state_dict": model.state_dict(), "scales": scales, "training": final_selection,
                    "epochs": epochs, "buffer_m": 200, "lighting": False,
                    "architecture": {"window": previous.WINDOW, "channels": previous.CHANNELS,
                    "hidden": previous.HIDDEN, "context_features": 9}}, DATA / "modelo_lstm.pt")
    else:
        x = data.training_xgb(records, VARIANT)
        model = fit_tabular(family, x, y, weights)
        scales = None
        joblib.dump({"model": model, "training": final_selection, "buffer_m": 200,
                     "lighting": False, "source": str(TURN_PANEL)}, DATA / f"modelo_{family}.joblib", compress=3)
        del x
    test = eval_periods(model, data, family, TEST, scales, save=True)
    result = {"family": family, "buffer_m": 200, "lighting": False, "training_selection": selection,
              "training_final": final_selection, "validation": validation, "test": test,
              "epochs_selected_by_2025_validation": epochs, "validation_history": history,
              "test_used_for_selection": False}
    save_json(path, result)
    del model, records, y, weights
    gc.collect()
    return result


def main():
    previous.seed()
    source = prepare()
    protocol = {"script": str(Path(__file__).resolve()), "source": str(SOURCE), "only_original_coordinates": True,
        "unit": "tramo × mes × turno", "buffer_m": 200, "sigma_m": 100, "high_threshold": THRESHOLD,
        "lighting": False, "nkde": False, "snapping": False, "history_starts": "2018-01",
        "selection_train_targets": ["2018-04", "2025-10"], "validation": list(VALIDATION),
        "final_train_targets": ["2018-04", "2025-12"], "test": list(TEST), "models_fixed_during_test": True,
        "sampling_per_class_month_turn": previous.SAMPLE, "seed": 42, "class_balance_power": SPEC["balance"],
        "turn_rule": source["turn_rule"], "xgboost": SPEC, "random_forest": RF_PARAMETERS,
        "lstm": {"window": previous.WINDOW, "channels": previous.CHANNELS, "hidden": previous.HIDDEN,
                 "context_features": 9, "max_epochs": previous.MAX_EPOCHS},
        "test_status": "Evaluación retrospectiva; estos meses ya fueron examinados",
        "comparison_to_300m": "Las etiquetas se reconstruyen a 200 m; los porcentajes de 200 y 300 m no usan idénticas etiquetas",
        "test_used_for_selection": False}
    save_json(OUT / "protocolo.json", protocol)
    data = Buffer200Data()
    data.monthly.verify_causality()
    results = {family: run(data, family) for family in ("xgboost", "lstm", "random_forest")}
    if len({tuple(r["test"]["metrics"]["support"]) for r in results.values()}) != 1:
        raise AssertionError("Las etiquetas de prueba difieren entre modelos")
    if len({r["training_final"]["sample_sha256"] for r in results.values()}) != 1:
        raise AssertionError("Las muestras de entrenamiento difieren entre modelos")
    save_json(OUT / "final_results.json", {"protocol": protocol, "source_audit": source, "results": results})
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), facecolor="white")
    for ax, family, title in zip(axes, ("xgboost", "lstm", "random_forest"), ("XGBoost", "LSTM", "Random Forest")):
        render_matrix(ax, results[family]["test"]["metrics"], title)
    fig.suptitle("Buffer 200 m · Sin alumbrado · Delitos con coordenadas originales", fontsize=21, weight="bold")
    fig.text(.5, .91, "Tramo × mes × turno · Historial desde 2018 · Prueba enero–mayo de 2026", ha="center", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .82), w_pad=3)
    path = OUT / "matrices_buffer_200_tres_modelos.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"IMAGEN {path}", flush=True)
    for family, result in results.items():
        m = result["test"]["metrics"]
        print(f"FINAL {family}: accuracy={m['accuracy']:.6f}; recall_medio={m['recall'][1]:.6f}; "
              f"recall_alto={m['recall'][2]:.6f}; precision_alto={m['precision'][2]:.6f}", flush=True)


if __name__ == "__main__":
    main()
