"""Ensayo LSTM de riesgo tramo×mes usando secuencias desde 2018.

La selección utiliza noviembre–diciembre de 2025. El modelo final se ajusta
hasta diciembre de 2025 y se evalúa en enero–agosto de 2026. Los datos del mes
objetivo nunca se incluyen en sus variables de entrada.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from buffer300.data import CACHE, DATA, META, PERIODS, THRESHOLD, load_static, save_json
from buffer300.modelos import matrix_metrics
from buffer300.report import render_matrix

OUT = ROOT / "outputs/evaluacion-modelos-sidpol/lstm_2018_300_v1"
MODEL_DIR = DATA / "lstm_v1"
TEMP = MODEL_DIR / "secuencias_mensuales_float16.npy"
TEMP_META = MODEL_DIR / "secuencias_mensuales_meta.json"
REFERENCE = ROOT / "outputs/evaluacion-modelos-sidpol/optimizacion_buffer_300_mensual_v2"
WINDOWS = (12, 24)
VAL_MONTHS = ("2025-11", "2025-12")
TEST_MONTHS = tuple(f"2026-{m:02d}" for m in range(1, 9))
SEED = 42
SAMPLE_PER_CLASS_MONTH = 1200
BATCH = 2048
MAX_EPOCHS = 4
HIDDEN = 32
FEATURES = 13


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, torch.get_num_threads()))


def labels_for_month(panel, factor, month):
    score = panel[month, :, 1] / factor
    return np.where(score >= THRESHOLD, 2, np.where(score > 0, 1, 0)).astype(np.int8)


def prepare_temporal():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    source = json.loads(META.read_text(encoding="utf-8"))["fingerprint"]
    if TEMP.exists() and TEMP_META.exists():
        meta = json.loads(TEMP_META.read_text(encoding="utf-8"))
        if meta["source_fingerprint"] != source or tuple(meta["shape"]) != (len(PERIODS), 228294, FEATURES):
            raise ValueError("Cambió el panel; use una nueva versión del experimento LSTM")
        print("VARIABLES_TEMPORALES_EXISTENTES", flush=True)
        return
    panel = np.load(CACHE, mmap_mode="r")
    static, factor = load_static()
    neighbors = np.load(DATA / "vecinos_32.npy", mmap_mode="r")
    if neighbors.shape != (len(factor), 32):
        raise ValueError("Red de vecinos incompatible")
    temporal = np.lib.format.open_memmap(TEMP, mode="w+", dtype=np.float16,
                                        shape=(len(PERIODS), len(factor), FEATURES))
    for month, period in enumerate(PERIODS):
        raw = np.maximum(panel[month] / factor[:, None], 0)
        risk = raw[:, 1]
        temporal[month, :, :11] = np.log1p(raw).astype(np.float16)
        temporal[month, :, 11] = np.log1p(risk[neighbors[:, :8]].mean(axis=1)).astype(np.float16)
        temporal[month, :, 12] = np.log1p(risk[neighbors].mean(axis=1)).astype(np.float16)
        if month % 12 == 0:
            temporal.flush()
            print(f"VARIABLES {period}", flush=True)
    temporal.flush()
    save_json(TEMP_META, {"source_fingerprint": source, "shape": list(temporal.shape),
              "dtype": "float16", "last_period": PERIODS[-1],
              "input_channels": "11 variables del panel normalizadas por longitud y log1p; "
                                "riesgo vecino medio 8 y 32, log1p"})
    print("VARIABLES_TEMPORALES_COMPLETAS", flush=True)


class SequenceData:
    def __init__(self):
        self.panel = np.load(CACHE, mmap_mode="r")
        self.temporal = np.load(TEMP, mmap_mode="r")
        static, self.factor = load_static()
        raw = np.column_stack([static[:, 0], static[:, 1], np.log1p(static[:, 2])]).astype(np.float32)
        self.static_mean = raw.mean(axis=0)
        self.static_std = np.maximum(raw.std(axis=0), 1e-6)
        self.static = ((raw-self.static_mean)/self.static_std).astype(np.float32)
        self.n = len(static)

    def sample(self, window, last_period):
        """Toma ejemplos de cada año y clase sin consultar meses posteriores."""
        final = PERIODS.index(last_period)
        months = range(window, final+1)
        records, counts = [], np.zeros(3, dtype=np.int64)
        for month in months:
            y = labels_for_month(self.panel, self.factor, month)
            count = np.bincount(y, minlength=3)
            counts += count
            rng = np.random.default_rng(SEED+month)
            for klass in range(3):
                candidates = np.flatnonzero(y == klass)
                keep = min(len(candidates), SAMPLE_PER_CLASS_MONTH)
                selected = rng.choice(candidates, size=keep, replace=False)
                records.append((month, selected, klass, len(candidates)/keep))
        total = sum(len(item[1]) for item in records)
        x = np.empty((total, window, FEATURES), dtype=np.float32)
        static_x = np.empty((total, 3), dtype=np.float32)
        y = np.empty(total, dtype=np.int64)
        weight = np.empty(total, dtype=np.float32)
        offset = 0
        for month, segments, klass, ratio in records:
            right = offset+len(segments)
            x[offset:right] = self.temporal[month-window:month][:, segments, :].transpose(1, 0, 2)
            static_x[offset:right] = self.static[segments]
            y[offset:right] = klass
            weight[offset:right] = ratio*(counts.sum()/(3*counts[klass]))**1.25
            offset = right
        weight /= weight.mean()
        return x, static_x, y, weight, {"first_target": PERIODS[window],
                "last_target": last_period, "sampled_rows": total,
                "population_rows": len(months)*self.n, "population_class_counts": counts.tolist()}

    def predict(self, model, month, window, temporal_mean, temporal_std, bias):
        probabilities = np.empty((self.n, 3), dtype=np.float32)
        model.eval()
        with torch.inference_mode():
            for left in range(0, self.n, 8192):
                right = min(left+8192, self.n)
                ids = np.arange(left, right)
                seq = self.temporal[month-window:month][:, ids, :].transpose(1, 0, 2).astype(np.float32)
                seq = (seq-temporal_mean)/temporal_std
                logits = model(torch.from_numpy(seq), torch.from_numpy(self.static[left:right]))
                probabilities[left:right] = torch.softmax(logits, dim=1).numpy()
        pred = (probabilities*np.asarray(bias, dtype=np.float32)).argmax(axis=1).astype(np.int8)
        return self.truth(month), probabilities, pred

    def truth(self, month):
        return labels_for_month(self.panel, self.factor, month)


class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence = nn.LSTM(FEATURES, HIDDEN, batch_first=True)
        self.head = nn.Sequential(nn.Linear(HIDDEN+3, 32), nn.ReLU(),
                                  nn.Dropout(.1), nn.Linear(32, 3))

    def forward(self, sequence, static):
        _, (hidden, _) = self.sequence(sequence)
        return self.head(torch.cat([hidden[-1], static], dim=1))


def metrics_from_predictions(y, pred):
    matrix = np.bincount(y.astype(np.int64)*3+pred.astype(np.int64), minlength=9).reshape(3, 3)
    return matrix_metrics(matrix)


def decision_search(y, probability):
    reference = json.loads((REFERENCE / "selection.json").read_text(encoding="utf-8"))["selected"]["metrics"]
    grid = (.65, .80, 1., 1.20, 1.50, 1.80)
    rows = []
    for medium in grid:
        for high in grid:
            bias = [1., medium, high]
            pred = (probability*np.asarray(bias, dtype=np.float32)).argmax(axis=1)
            m = metrics_from_predictions(y, pred)
            eligible = m["precision"][2] >= .9*reference["precision"][2] and m["recall"][2] >= .9*reference["recall"][2]
            rows.append({"bias": bias, "metrics": m, "eligible": bool(eligible),
                         "score": .4*m["accuracy"]+.6*m["f1_medio_alto"]})
    selected = max([row for row in rows if row["eligible"]] or rows, key=lambda row: row["score"])
    return selected, rows


def train_once(data, window, last_period, epochs, validate):
    x, static_x, y, weights, details = data.sample(window, last_period)
    temporal_mean = x.mean(axis=(0, 1))
    temporal_std = np.maximum(x.std(axis=(0, 1)), .05)
    x -= temporal_mean
    x /= temporal_std
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(static_x),
                            torch.from_numpy(y), torch.from_numpy(weights))
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)
    model = LSTMClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    best, best_state, history = None, None, []
    for epoch in range(1, epochs+1):
        started = time.perf_counter()
        model.train()
        losses = []
        for seq, sta, target, weight in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(seq, sta)
            loss = (F.cross_entropy(logits, target, reduction="none")*weight).sum()/weight.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        record = {"epoch": epoch, "training_loss": float(np.mean(losses)),
                  "seconds": round(time.perf_counter()-started, 1)}
        if validate:
            ys, ps = [], []
            for period in VAL_MONTHS:
                truth, probability, _ = data.predict(model, PERIODS.index(period), window,
                                                     temporal_mean, temporal_std, [1, 1, 1])
                ys.append(truth)
                ps.append(probability)
            val_y, val_p = np.concatenate(ys), np.concatenate(ps)
            decision, _ = decision_search(val_y, val_p)
            record["validation"] = decision
            score = decision["score"]
            if best is None or score > best["score"]:
                best = {"epoch": epoch, **decision}
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(record)
        suffix = (f" val_acc={record['validation']['metrics']['accuracy']:.4f} "
                  f"val_alto={record['validation']['metrics']['recall'][2]:.4f} "
                  f"score={record['validation']['score']:.4f}" if validate else "")
        print(f"VENTANA {window} EPOCA {epoch}/{epochs}: loss={record['training_loss']:.4f} "
              f"seg={record['seconds']}{suffix}", flush=True)
    if validate:
        model.load_state_dict(best_state)
    return model, temporal_mean, temporal_std, details, history, best


def validate_models():
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not TEMP.exists():
        prepare_temporal()
    set_seed()
    data = SequenceData()
    results = []
    for window in WINDOWS:
        path = OUT / f"validacion_{window}m.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        set_seed()
        print(f"INICIO_VALIDACION ventana={window} historia_desde={PERIODS[window-12]}", flush=True)
        model, mean, std, details, history, best = train_once(data, window, "2025-10", MAX_EPOCHS, True)
        result = {"window": window, "training": details, "best": best, "history": history,
                  "selection_periods": VAL_MONTHS, "seed": SEED}
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                    "window": window, "best": best, "training": details},
                   MODEL_DIR / f"validacion_{window}m.pt")
        save_json(path, result)
        results.append(result)
        del model
        gc.collect()
    selected = max(results, key=lambda r: r["best"]["score"])
    save_json(OUT / "selection.json", {"candidates": results, "selected": selected,
              "validation": VAL_MONTHS, "test_used_for_selection": False,
              "protocol": "LSTM con ventanas de 12 o 24 meses, entrenada con filas objetivo anteriores a 2025-11"})
    print(f"SELECCION LSTM ventana={selected['window']} epocas={selected['best']['epoch']} "
          f"bias={selected['best']['bias']} score={selected['best']['score']:.4f}", flush=True)


def fit_and_evaluate():
    selected = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))["selected"]
    window, epochs = selected["window"], selected["best"]["epoch"]
    bias = selected["best"]["bias"]
    set_seed()
    data = SequenceData()
    print(f"AJUSTE_FINAL ventana={window} epocas={epochs} historial_hasta=2025-12", flush=True)
    model, mean, std, details, history, _ = train_once(data, window, "2025-12", epochs, False)
    package = {"state_dict": model.state_dict(), "mean": mean, "std": std,
               "window": window, "bias": bias, "training": details,
               "architecture": {"hidden": HIDDEN, "channels": FEATURES, "layers": 1},
               "last_train_target": "2025-12", "first_test": TEST_MONTHS[0]}
    torch.save(package, MODEL_DIR / "modelo_final.pt")
    monthly = []
    for period in TEST_MONTHS:
        truth, probability, pred = data.predict(model, PERIODS.index(period), window, mean, std, bias)
        result = {"periodo": period, "metrics": metrics_from_predictions(truth, pred)}
        save_json(OUT / f"resultado_{period}.json", result)
        np.savez_compressed(OUT / f"predicciones_{period}.npz", truth=truth, probability=probability, predicted=pred)
        monthly.append(result)
        m = result["metrics"]
        print(f"EVALUACION {period}: acc={m['accuracy']:.4f} F1MA={m['f1_medio_alto']:.4f} "
              f"P/R alto={m['precision'][2]:.4f}/{m['recall'][2]:.4f}", flush=True)
    groups = {}
    for label, rows in (("enero_mayo", monthly[:5]), ("junio_agosto", monthly[5:]),
                        ("enero_agosto", monthly)):
        matrix = sum((np.asarray(item["metrics"]["matriz_confusion"]) for item in rows),
                     np.zeros((3, 3), dtype=np.int64))
        groups[label] = {"periods": [item["periodo"] for item in rows],
                         "metrics": matrix_metrics(matrix)}
    save_json(OUT / "final_results.json", {"selected_window": window, "selected_epochs": epochs,
              "bias": bias, "training": details, "monthly": monthly, "groups": groups,
              "test_used_for_selection": False,
              "note": "2026 fue consultado en ensayos anteriores; evaluación retrospectiva"})
    print("EVALUACION_LSTM_COMPLETA", flush=True)


def rolling_evaluate():
    """Reentrena desde 2018 hasta el mes anterior a cada predicción de 2026."""
    selected = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))["selected"]
    window, epochs, bias = selected["window"], selected["best"]["epoch"], selected["best"]["bias"]
    data = SequenceData()
    monthly = []
    for period in TEST_MONTHS:
        output = OUT / f"rolling_resultado_{period}.json"
        if output.exists():
            monthly.append(json.loads(output.read_text(encoding="utf-8")))
            print(f"EXISTENTE_ROLLING {period}", flush=True)
            continue
        month = PERIODS.index(period)
        set_seed()
        print(f"REENTRENAMIENTO_LSTM {period}: historia hasta {PERIODS[month-1]}", flush=True)
        model, mean, std, details, history, _ = train_once(data, window, PERIODS[month-1], epochs, False)
        truth, probability, pred = data.predict(model, month, window, mean, std, bias)
        result = {"periodo": period, "training": details, "metrics": metrics_from_predictions(truth, pred)}
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                    "window": window, "bias": bias, "training": details,
                    "architecture": {"hidden": HIDDEN, "channels": FEATURES, "layers": 1},
                    "target_period": period}, MODEL_DIR / f"modelo_para_{period}.pt")
        save_json(output, result)
        np.savez_compressed(OUT / f"rolling_predicciones_{period}.npz",
                            truth=truth, probability=probability, predicted=pred)
        monthly.append(result)
        m = result["metrics"]
        print(f"ROLLING {period}: acc={m['accuracy']:.4f} F1MA={m['f1_medio_alto']:.4f} "
              f"P/R alto={m['precision'][2]:.4f}/{m['recall'][2]:.4f}", flush=True)
        del model, probability
        gc.collect()
    groups = {}
    for label, rows in (("enero_mayo", monthly[:5]), ("junio_agosto", monthly[5:]),
                        ("enero_agosto", monthly)):
        matrix = sum((np.asarray(item["metrics"]["matriz_confusion"]) for item in rows),
                     np.zeros((3, 3), dtype=np.int64))
        groups[label] = {"periods": [item["periodo"] for item in rows],
                         "metrics": matrix_metrics(matrix)}
    save_json(OUT / "rolling_final_results.json", {"selected_window": window,
              "selected_epochs": epochs, "bias": bias, "monthly": monthly, "groups": groups,
              "monthly_refit": True, "test_used_for_selection": False,
              "note": "2026 fue consultado en ensayos anteriores; evaluación retrospectiva"})
    print("EVALUACION_ROLLING_COMPLETA", flush=True)


def report(rolling=False):
    filename = "rolling_final_results.json" if rolling else "final_results.json"
    result = json.loads((OUT / filename).read_text(encoding="utf-8"))
    baseline = json.loads((REFERENCE / "final_results.json").read_text(encoding="utf-8"))
    table = []
    for label, reference_label in (("enero_mayo", "benchmark"),
                                   ("junio_agosto", "additional_evaluation")):
        a = baseline["evaluations"][reference_label]["metrics"]
        b = result["groups"][label]["metrics"]
        table.append((label, a, b))
    matrices = [np.asarray(baseline["evaluations"][name]["metrics"]["matriz_confusion"])
                for name in ("benchmark", "additional_evaluation")]
    baseline_total = matrix_metrics(sum(matrices))
    table.append(("enero_agosto", baseline_total, result["groups"]["enero_agosto"]["metrics"]))
    suffix = "_rolling" if rolling else ""
    save_json(OUT / f"comparacion_xgboost{suffix}.json", {name: {"xgboost_3m": a, "lstm": b}
                                                            for name, a, b in table})
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8), facecolor="white")
    render_matrix(axes[0], baseline_total, "XGBoost · 3 meses")
    render_matrix(axes[1], result["groups"]["enero_agosto"]["metrics"],
                  f"LSTM mensual · {result['selected_window']} meses de secuencia" if rolling else
                  f"LSTM fija · {result['selected_window']} meses de secuencia")
    fig.suptitle("Riesgo por tramo y mes · buffer 300 m", y=.98, fontsize=18, weight="bold")
    fig.text(.5, .91, "Enero–agosto de 2026 · porcentajes por clase real", ha="center", fontsize=12)
    fig.subplots_adjust(top=.78, bottom=.14, left=.06, right=.98, wspace=.25)
    destination = OUT / f"matriz_confusion_lstm_vs_xgboost{suffix}.png"
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)
    print(destination, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "validate", "evaluate", "report", "rolling", "rolling_report"))
    stage = parser.parse_args().stage
    {"prepare": prepare_temporal, "validate": validate_models,
     "evaluate": fit_and_evaluate, "report": report,
     "rolling": rolling_evaluate, "rolling_report": lambda: report(True)}[stage]()
