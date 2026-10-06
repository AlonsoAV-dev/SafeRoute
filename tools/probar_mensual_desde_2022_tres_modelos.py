"""Mensual, 300 m, originales: train 2022–2024, val 2025, test 2026.

Se conserva el entrenamiento hasta diciembre 2024 durante validación y prueba.
Ejecutar: .venv/Scripts/python tools/probar_mensual_desde_2022_tres_modelos.py
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
from sklearn.ensemble import RandomForestClassifier
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from buffer300.data import PERIODS, THRESHOLD, save_json
from buffer300.modelos import fit_model, predictions, evaluate, matrix_metrics, score, CONFIG
from buffer300.report import render_matrix
from probar_buffer_300_originales import PANEL, PANEL_META, SOURCE, fingerprint
from probar_alumbrado_fecha_xgb_lstm import DATA as TEMPORAL_DATA, seed, normalize_training
from modelos_mensuales import (
    MonthlyData, MonthlyLSTM, lstm_predict, RF_PARAMETERS, WINDOW, CHANNELS, HIDDEN,
)

DATA = ROOT / "Backend/data/experimentos_sidpol/buffer_300_mensual_desde_2022_v1"
OUT = ROOT / "outputs/evaluacion-modelos-sidpol/buffer_300_mensual_desde_2022_v1"
START = PERIODS.index("2022-01")
MONTHS = PERIODS[START:]
TRAIN_FIRST, TRAIN_LAST = MONTHS.index("2022-04"), MONTHS.index("2024-12")
VALIDATION = tuple(f"2025-{month:02d}" for month in range(1, 13))
TEST = tuple(f"2026-{month:02d}" for month in range(1, 6))
FAMILIES = ("xgboost", "lstm", "random_forest")
MAX_EPOCHS = 4
SPEC = {"name": "contexto_desde2022_hasta2024", "features": "contexto", "balance": 1.25}


class Data2022(MonthlyData):
    def __init__(self):
        super().__init__()
        self.panel = self.panel[START:]
        self.risk = self.risk[START:]
        self.temporal = self.temporal[START:]

    def selection(self):
        months = list(range(TRAIN_FIRST, TRAIN_LAST+1))
        y_all = self.target(months)
        counts = np.bincount(y_all, minlength=3)
        rng = np.random.default_rng(CONFIG["seed"])
        indexes, ratios = [], []
        limit = min(CONFIG["sampling"]["per_class_month"],
                    CONFIG["sampling"]["maximum_training_rows"]//(3*len(months)))
        for local, month in enumerate(months):
            labels = y_all[local*self.n:(local+1)*self.n]
            for cls in range(3):
                candidates = np.flatnonzero(labels == cls)
                keep = min(limit, len(candidates))
                if keep:
                    indexes.append(rng.choice(candidates, keep, replace=False)+local*self.n)
                    ratios.append(np.full(keep, len(candidates)/keep, dtype=np.float32))
        indexes = np.concatenate(indexes)
        order = np.argsort(indexes)
        indexes, ratios = indexes[order], np.concatenate(ratios)[order]
        y = y_all[indexes].astype(np.int64)
        weights = ratios*(counts.sum()/(3*counts[y]))**SPEC["balance"]
        weights = (weights/weights.mean()).astype(np.float32)
        entries = []
        for local, month in enumerate(months):
            left, right = np.searchsorted(indexes, [local*self.n, (local+1)*self.n])
            entries.append((month, indexes[left:right]-local*self.n))
        digest = hashlib.sha256(indexes.astype(np.int32).tobytes()).hexdigest()
        details = {"first_target": MONTHS[TRAIN_FIRST], "last_target": MONTHS[TRAIN_LAST],
                   "history_starts": MONTHS[0], "train_units": len(y_all), "train_samples": len(y),
                   "population_counts": counts.tolist(), "sample_counts": np.bincount(y, minlength=3).tolist(),
                   "sampling_per_class_month": limit, "sample_sha256": digest,
                   "class_balance_power": SPEC["balance"]}
        assert max(month for month, _ in entries) < MONTHS.index(VALIDATION[0]) < MONTHS.index(TEST[0])
        return entries, y, weights, details

    def verify_causality(self):
        month = MONTHS.index("2025-08")
        original = self.panel, self.risk, self.temporal
        class PastOnly:
            def __init__(self, values):
                self.values = values
            def __getitem__(self, key):
                first = key[0] if isinstance(key, tuple) else key
                if isinstance(first, slice):
                    assert first.stop is not None and first.stop <= month
                else:
                    assert first < month
                return self.values[key]
        try:
            self.panel, self.risk, self.temporal = (PastOnly(values) for values in original)
            for kind in ("legacy", "temporal", "contexto"):
                assert np.isfinite(self.make(month, kind)).all()
            assert np.isfinite(self.sequence(month, np.arange(10))).all()
        finally:
            self.panel, self.risk, self.temporal = original


def evaluate_periods(model, data, family, periods, training, scales=None, save=False):
    rows = []
    for period in periods:
        month = MONTHS.index(period)
        if family == "lstm":
            probability = lstm_predict(model, data, month, scales)
        else:
            probability = predictions(model, data.make(month, "contexto"))
        truth = data.target([month])
        if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1, atol=1e-5):
            raise AssertionError("Probabilidades inválidas")
        row = {"periodo": period, "family": family, "training": training, "metrics": evaluate(truth, probability)}
        rows.append(row)
        if save:
            np.savez_compressed(DATA / f"predicciones_{family}_{period}.npz", truth=truth, probability=probability)
            save_json(OUT / f"resultado_{family}_{period}.json", row)
        if save or period in (periods[0], periods[-1]):
            m = row["metrics"]
            print(f"EVAL DESDE2022 {family} {period}: accuracy={m['accuracy']:.4f}; alto={m['recall'][2]:.4f}", flush=True)
    return {"monthly": rows, "metrics": matrix_metrics(sum(np.asarray(row["metrics"]["matriz_confusion"]) for row in rows))}


def train_lstm(data, entries, y, weights, training):
    seed()
    print("PREPARANDO SECUENCIAS LSTM DESDE ENERO 2022", flush=True)
    seq, context = data.training_lstm(entries)
    scales = normalize_training(seq, context)
    loader = DataLoader(TensorDataset(torch.from_numpy(seq), torch.from_numpy(context),
                        torch.from_numpy(y), torch.from_numpy(weights)), batch_size=2048, shuffle=True, num_workers=0)
    model = MonthlyLSTM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    history, best = [], None
    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        losses = []
        for x, ctx, target, weight in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = (F.cross_entropy(model(x, ctx), target, reduction="none")*weight).sum()/weight.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate_periods(model, data, "lstm", VALIDATION, training, scales)
        current = score(validation["metrics"])
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "validation": validation, "score": current}
        history.append(row)
        if best is None or current > best["score"]:
            best = {"epoch": epoch, "score": current, "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "validation": validation}
        m = validation["metrics"]
        print(f"LSTM DESDE2022 epoca={epoch}/{MAX_EPOCHS}; val2025 accuracy={m['accuracy']:.4f}; "
              f"alto={m['recall'][2]:.4f}; score={current:.4f}", flush=True)
        save_json(OUT / "historial_validacion_lstm.json", history)
    model.load_state_dict(best["state"])
    return model, scales, history, {"epoch": best["epoch"], "score": best["score"], "validation": best["validation"]}


def run(data, family, entries, y, weights, training, x=None):
    path = OUT / f"resultado_{family}.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["training"]["sample_sha256"] != training["sample_sha256"]:
            raise ValueError("Cambió la muestra del experimento guardado")
        return result
    seed()
    print(f"ENTRENANDO DESDE2022 {family}: {training}", flush=True)
    scales, history, chosen = None, None, None
    if family == "lstm":
        model, scales, history, chosen = train_lstm(data, entries, y, weights, training)
        torch.save({"state_dict": model.state_dict(), "scales": scales, "training": training,
                    "epoch": chosen["epoch"], "architecture": {"window": WINDOW, "channels": CHANNELS,
                    "context_features": 5, "hidden": HIDDEN}, "buffer_m": 300, "lighting": False},
                   DATA / "modelo_lstm.pt")
    else:
        if family == "xgboost":
            model = fit_model(SPEC, x, y, weights)
        else:
            model = RandomForestClassifier(**RF_PARAMETERS)
            model.fit(x, y, sample_weight=weights)
        joblib.dump({"model": model, "training": training, "buffer_m": 300, "lighting": False,
                     "source_panel": str(PANEL)}, DATA / f"modelo_{family}.joblib", compress=3)
    validation = evaluate_periods(model, data, family, VALIDATION, training, scales, save=True)
    test = evaluate_periods(model, data, family, TEST, training, scales, save=True)
    result = {"family": family, "training": training, "validation": validation, "test": test,
              "validation_history": history, "lstm_epoch_selection": chosen,
              "model_training_ends": "2024-12", "refit_on_validation": False, "test_used_for_selection": False}
    save_json(path, result)
    del model, scales
    gc.collect()
    return result


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    signature = fingerprint()
    source = json.loads(PANEL_META.read_text(encoding="utf-8"))
    temporal = json.loads((TEMPORAL_DATA / "secuencias_complete.json").read_text(encoding="utf-8"))
    if source["fingerprint"] != signature or temporal["fingerprint"] != signature:
        raise ValueError("La fuente, el panel y las secuencias no coinciden")
    protocol = {"script": str(Path(__file__).resolve()), "source": str(SOURCE), "source_panel": str(PANEL),
        "only_original_coordinates": True, "unit": "tramo × mes", "buffer_m": 300, "sigma_m": 100,
        "high_threshold": THRESHOLD, "lighting": False, "turn_conditioned_prediction": False,
        "history_starts": "2022-01", "training_targets": ["2022-04", "2024-12"],
        "early_history": "Enero–marzo 2022 sirve de historia; LSTM usa padding hasta completar 12 meses",
        "validation": list(VALIDATION), "test": list(TEST), "feature_history_months": WINDOW,
        "forecast_horizon_months": 1, "models_fixed_during_validation_and_test": True,
        "refit_on_validation": False, "weight_updates_use_2025_or_2026_targets": False,
        "lstm_epoch_selection": "Maximizar 0.4 accuracy + 0.6 F1 medio/alto en todo 2025",
        "xgboost": SPEC, "random_forest": RF_PARAMETERS,
        "lstm": {"window": WINDOW, "channels": CHANNELS, "context_features": 5, "hidden": HIDDEN, "max_epochs": MAX_EPOCHS},
        "sampling": CONFIG["sampling"], "seed": 42, "class_balance_power": SPEC["balance"],
        "test_used_for_selection": False, "test_status": "Evaluación retrospectiva; 2026 ya fue examinado",
        "features": "Cada mes usa hechos observados hasta el mes anterior, incluidos los meses previos de validación/prueba"}
    save_json(OUT / "protocolo.json", protocol)
    data = Data2022()
    data.verify_causality()
    entries, y, weights, training = data.selection()
    save_json(OUT / "muestra_entrenamiento.json", training)
    print(f"MUESTRA COMUN DESDE2022: {training}", flush=True)
    results = {}
    if any(not (OUT / f"resultado_{family}.json").exists() for family in ("xgboost", "random_forest")):
        x = data.training_tabular(entries)
    else:
        x = None
    for family in ("xgboost", "random_forest"):
        results[family] = run(data, family, entries, y, weights, training, x)
    del x
    gc.collect()
    results["lstm"] = run(data, "lstm", entries, y, weights, training)
    if len({tuple(r["test"]["metrics"]["support"]) for r in results.values()}) != 1:
        raise AssertionError("Las etiquetas de prueba difieren entre modelos")
    if len({r["training"]["sample_sha256"] for r in results.values()}) != 1:
        raise AssertionError("Las muestras de entrenamiento difieren entre modelos")
    save_json(OUT / "final_results.json", {"protocol": protocol, "training": training,
              "source_metadata": source, "results": results})
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), facecolor="white")
    for ax, family, title in zip(axes, FAMILIES, ("XGBoost", "LSTM", "Random Forest")):
        render_matrix(ax, results[family]["test"]["metrics"], title)
    fig.suptitle("Mensual · Buffer 300 m · Sin alumbrado · Coordenadas originales", fontsize=21, weight="bold")
    fig.text(.5, .91, "Train 2022–2024 · Validación 2025 · Prueba enero–mayo 2026 · Modelos fijos", ha="center", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .82), w_pad=3)
    path = OUT / "matrices_mensuales_desde_2022.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"IMAGEN {path}", flush=True)
    for family in FAMILIES:
        m = results[family]["test"]["metrics"]
        print(f"FINAL {family}: accuracy={m['accuracy']:.6f}; recall_medio={m['recall'][1]:.6f}; "
              f"recall_alto={m['recall'][2]:.6f}; precision_alto={m['precision'][2]:.6f}", flush=True)


if __name__ == "__main__":
    main()
