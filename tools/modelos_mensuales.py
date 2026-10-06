"""Ensayo mensual de 300 m: XGBoost, LSTM y Random Forest sin alumbrado.

Replica la ventana móvil de tres meses del ensayo mensual de referencia.
Ejecutar: .venv/Scripts/python tools/modelos_mensuales.py
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
from buffer300.features import Features
from buffer300.modelos import fit_model, predictions, evaluate, matrix_metrics, score, CONFIG
from buffer300.report import render_matrix
from probar_buffer_300_originales import PANEL, PANEL_META, SOURCE, fingerprint
from probar_alumbrado_fecha_xgb_lstm import (
    TEMPORAL, DATA as TEMPORAL_DATA, seed, normalize_training,
)

DATA = ROOT / "Backend/data/experimentos_sidpol/buffer_300_mensual_tres_modelos_v1"
OUT = ROOT / "outputs/evaluacion-modelos-sidpol/buffer_300_mensual_tres_modelos_v1"
VALIDATION = ("2025-11", "2025-12")
TEST = tuple(f"2026-{month:02d}" for month in range(1, 6))
SPEC = {"name": "temporal_3m", "features": "contexto", "balance": 1.25, "window": 3}
# RANDOM FOREST: combina 200 árboles para clasificar el riesgo del segmento.
# La profundidad y el tamaño mínimo de las hojas limitan el sobreajuste.
# Esta configuración y la arquitectura LSTM también se reutilizan en el ensayo con NKDE.
RF_PARAMETERS = {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 20,
                 "min_samples_split": 40, "max_features": "sqrt", "bootstrap": True,
                 "max_samples": .8, "class_weight": None, "random_state": 42, "n_jobs": 8}
# LSTM: secuencias de 12 meses, 14 variables por mes y 32 unidades ocultas.
WINDOW, CHANNELS, HIDDEN, MAX_EPOCHS = 12, 14, 32, 4
FAMILIES = ("xgboost", "lstm", "random_forest")


class MonthlyData(Features):
    def __init__(self):
        super().__init__(panel_path=PANEL)
        self.temporal = np.load(TEMPORAL, mmap_mode="r")

    def selection(self, month):
        months = range(month-SPEC["window"], month)
        y_all = self.target(list(months))
        counts = np.bincount(y_all, minlength=3)
        rng = np.random.default_rng(CONFIG["seed"])
        indexes, ratios = [], []
        limit = min(CONFIG["sampling"]["per_class_month"],
                    CONFIG["sampling"]["maximum_training_rows"]//(3*len(months)))
        for local, prior in enumerate(months):
            y_month = y_all[local*self.n:(local+1)*self.n]
            for cls in range(3):
                candidates = np.flatnonzero(y_month == cls)
                keep = min(limit, len(candidates))
                if keep:
                    chosen = rng.choice(candidates, keep, replace=False)
                    indexes.append(chosen+local*self.n)
                    ratios.append(np.full(keep, len(candidates)/keep, dtype=np.float32))
        indexes = np.concatenate(indexes)
        order = np.argsort(indexes)
        indexes, ratios = indexes[order], np.concatenate(ratios)[order]
        y = y_all[indexes].astype(np.int64)
        weights = ratios*(counts.sum()/(3*counts[y]))**SPEC["balance"]
        weights = (weights/weights.mean()).astype(np.float32)
        entries = []
        for local, prior in enumerate(months):
            left, right = np.searchsorted(indexes, [local*self.n, (local+1)*self.n])
            entries.append((prior, indexes[left:right]-local*self.n))
        details = {"first_month": PERIODS[month-SPEC["window"]], "last_month": PERIODS[month-1],
                   "train_units": len(y_all), "train_samples": len(y), "class_counts": counts.tolist(),
                   "sample_sha256": hashlib.sha256(indexes.astype(np.int32).tobytes()).hexdigest(),
                   "maximum_samples_per_class_month": limit}
        return entries, y, weights, details

    def training_tabular(self, entries):
        return np.concatenate([self.make(month, "contexto")[ids] for month, ids in entries])

    # Prepara los 12 meses anteriores de cada segmento, sin incluir el mes que se predice.
    def sequence(self, month, ids):
        start = max(0, month-WINDOW)
        result = np.zeros((len(ids), WINDOW, CHANNELS), dtype=np.float32)
        known = np.asarray(self.temporal[start:month][:, ids], dtype=np.float32).transpose(1, 0, 2)
        result[:, WINDOW-known.shape[1]:] = known
        return result

    # Añade ubicación, logaritmo de longitud vial y seno/coseno del mes: cinco variables.
    def context(self, month, ids):
        static = self.static[ids]
        season = np.broadcast_to([np.sin(2*np.pi*(month % 12)/12), np.cos(2*np.pi*(month % 12)/12)], (len(ids), 2))
        return np.column_stack([static[:, :2], np.log1p(static[:, 2]), season]).astype(np.float32)

    def training_lstm(self, entries):
        n = sum(len(ids) for _, ids in entries)
        seq = np.empty((n, WINDOW, CHANNELS), dtype=np.float32)
        context = np.empty((n, 5), dtype=np.float32)
        left = 0
        for month, ids in entries:
            right = left+len(ids)
            seq[left:right], context[left:right] = self.sequence(month, ids), self.context(month, ids)
            left = right
        return seq, context


# LSTM: aprende patrones temporales del historial para estimar el riesgo futuro.
class MonthlyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Resume la secuencia mensual en un vector de 32 componentes.
        self.sequence = nn.LSTM(CHANNELS, HIDDEN, batch_first=True)
        # Combina el historial con el contexto y produce tres puntajes: Bajo, Medio y Alto.
        self.head = nn.Sequential(nn.Linear(HIDDEN+5, 32), nn.ReLU(), nn.Dropout(.1), nn.Linear(32, 3))

    def forward(self, seq, context):
        _, (hidden, _) = self.sequence(seq)
        return self.head(torch.cat([hidden[-1], context], dim=1))


# PREDICCIÓN: normaliza con el entrenamiento y convierte los puntajes a probabilidades.
def lstm_predict(model, data, month, scales):
    probability = np.empty((data.n, 3), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for left in range(0, data.n, 8192):
            ids = np.arange(left, min(left+8192, data.n))
            seq = (data.sequence(month, ids)-scales["mean"])/scales["std"]
            context = (data.context(month, ids)-scales["context_mean"])/scales["context_std"]
            probability[ids] = torch.softmax(model(torch.from_numpy(seq), torch.from_numpy(context)), dim=1).numpy()
    return probability


# ENTRENAMIENTO LSTM: aprende con secuencias anteriores y etiquetas de riesgo conocidas.
def train_lstm(data, entries, y, weights, month, epochs, validate):
    seed()
    seq, context = data.training_lstm(entries)
    scales = normalize_training(seq, context)
    loader = DataLoader(TensorDataset(torch.from_numpy(seq), torch.from_numpy(context),
                        torch.from_numpy(y), torch.from_numpy(weights)), batch_size=2048, shuffle=True, num_workers=0)
    model = MonthlyLSTM()
    # AdamW actualiza los pesos de la red con una tasa de aprendizaje de 0,001.
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        for x, ctx, target, weight in loader:
            optimizer.zero_grad(set_to_none=True)
            # Calcula el error de clasificación y pondera las muestras de entrenamiento.
            loss = (F.cross_entropy(model(x, ctx), target, reduction="none")*weight).sum()/weight.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        if validate:
            row["metrics"] = evaluate(data.target([month]), lstm_predict(model, data, month, scales))
        history.append(row)
        print(f"LSTM MENSUAL {PERIODS[month]} epoca={epoch}/{epochs}, loss={row['loss']:.4f}", flush=True)
    return model, scales, history


def save_evaluation(data, month, family, probability, training):
    truth = data.target([month])
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1, atol=1e-5):
        raise AssertionError("Probabilidades inválidas")
    row = {"periodo": PERIODS[month], "family": family, "training": training, "metrics": evaluate(truth, probability)}
    save_json(OUT / f"resultado_{family}_{PERIODS[month]}.json", row)
    np.savez_compressed(DATA / f"predicciones_{family}_{PERIODS[month]}.npz", truth=truth, probability=probability)
    m = row["metrics"]
    print(f"EVAL MENSUAL {family} {PERIODS[month]}: accuracy={m['accuracy']:.4f}; alto={m['recall'][2]:.4f}", flush=True)
    return row


# RF Y XGBOOST: reciben las mismas filas tabulares y se evalúan en el mismo mes.
def tabular_month(data, month, entries, y, weights, training):
    missing = [family for family in ("xgboost", "random_forest")
               if not (OUT / f"resultado_{family}_{PERIODS[month]}.json").exists()]
    if missing:
        x = data.training_tabular(entries)
        current = data.make(month, "contexto")
        for family in missing:
            seed()
            print(f"ENTRENANDO MENSUAL {family} {PERIODS[month]}: {training['first_month']}–{training['last_month']}; {len(y):,} muestras", flush=True)
            if family == "xgboost":
                # XGBoost ajusta árboles sucesivos para reducir el error de clasificación.
                model = fit_model(SPEC, x, y, weights)
            else:
                # Random Forest ajusta el conjunto de árboles con los parámetros definidos arriba.
                model = RandomForestClassifier(**RF_PARAMETERS)
                model.fit(x, y, sample_weight=weights)
            # Ambos modelos entregan P(Bajo), P(Medio) y P(Alto) para cada segmento.
            probability = predictions(model, current)
            joblib.dump({"model": model, "training": training, "buffer_m": 300, "lighting": False,
                         "unit": "tramo x mes", "source_panel": str(PANEL)},
                        DATA / f"modelo_{family}_{PERIODS[month]}.joblib", compress=3)
            save_evaluation(data, month, family, probability, training)
            del model, probability
            gc.collect()
        del x, current
    return {family: json.loads((OUT / f"resultado_{family}_{PERIODS[month]}.json").read_text(encoding="utf-8"))
            for family in ("xgboost", "random_forest")}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(PANEL_META.read_text(encoding="utf-8"))
    signature = fingerprint()
    if source["fingerprint"] != signature:
        raise ValueError("Cambió la fuente original del panel mensual")
    temporal_meta = json.loads((TEMPORAL_DATA / "secuencias_complete.json").read_text(encoding="utf-8"))
    if temporal_meta["fingerprint"] != signature:
        raise ValueError("Las secuencias no coinciden con el panel original")
    protocol = {"script": str(Path(__file__).resolve()), "source": str(SOURCE), "source_panel": str(PANEL),
        "only_original_coordinates": True, "buffer_m": 300, "sigma_m": 100, "high_threshold": THRESHOLD,
        "unit": "tramo × mes", "lighting": False, "turn_conditioned_prediction": False,
        "training_window_months": 3, "feature_history_months": 12, "forecast_horizon_months": 1,
        "training": "Cada mes se reentrenan los tres modelos con las etiquetas de los tres meses anteriores",
        "validation": list(VALIDATION), "test": list(TEST), "panel_starts": "2018-01",
        "earliest_history_actually_used": "2024-08", "all_2018_2026_targets_used_for_training": False,
        "labels_fixed_across_models": True, "same_samples_across_models": True,
        "sampling": CONFIG["sampling"], "seed": 42, "class_balance_power": SPEC["balance"],
        "xgboost": SPEC, "random_forest": RF_PARAMETERS,
        "lstm": {"window": WINDOW, "channels": CHANNELS, "context_features": 5, "hidden": HIDDEN,
                 "max_validation_epochs": MAX_EPOCHS, "epoch_selection": "Score agregado de noviembre y diciembre de 2025"},
        "test_used_for_selection": False, "test_status": "Evaluación retrospectiva; 2026 ya fue examinado"}
    save_json(OUT / "protocolo.json", protocol)
    data = MonthlyData()
    data.verify_causality()
    validations = {family: [] for family in FAMILIES}
    for period in VALIDATION:
        month = PERIODS.index(period)
        entries, y, weights, training = data.selection(month)
        for family, row in tabular_month(data, month, entries, y, weights, training).items():
            validations[family].append(row)
        path = OUT / f"validacion_lstm_{period}.json"
        if not path.exists():
            model, scales, history = train_lstm(data, entries, y, weights, month, MAX_EPOCHS, True)
            save_json(path, {"periodo": period, "training": training, "history": history})
            del model, scales
        validations["lstm"].append(json.loads(path.read_text(encoding="utf-8")))
        del entries, y, weights
        gc.collect()
    choices = []
    for epoch in range(1, MAX_EPOCHS+1):
        matrix = sum(np.asarray(row["history"][epoch-1]["metrics"]["matriz_confusion"]) for row in validations["lstm"])
        metrics = matrix_metrics(matrix)
        choices.append({"epoch": epoch, "metrics": metrics, "score": score(metrics)})
    selected = max(choices, key=lambda row: row["score"])
    epochs = selected["epoch"]
    save_json(OUT / "seleccion_epocas_lstm.json", {"selected": selected, "all": choices, "test_used": False})
    print(f"EPOCAS LSTM ELEGIDAS CON VALIDACION 2025: {epochs}", flush=True)
    results = {family: {"test": {"monthly": []}} for family in FAMILIES}
    for period in TEST:
        month = PERIODS.index(period)
        entries, y, weights, training = data.selection(month)
        for family, row in tabular_month(data, month, entries, y, weights, training).items():
            results[family]["test"]["monthly"].append(row)
        path = OUT / f"resultado_lstm_{period}.json"
        if not path.exists():
            model, scales, history = train_lstm(data, entries, y, weights, month, epochs, False)
            torch.save({"state_dict": model.state_dict(), "scales": scales, "training": training,
                        "buffer_m": 300, "lighting": False, "unit": "tramo x mes", "epochs": epochs,
                        "architecture": {"window": WINDOW, "channels": CHANNELS, "context_features": 5, "hidden": HIDDEN}},
                       DATA / f"modelo_lstm_{period}.pt")
            save_evaluation(data, month, "lstm", lstm_predict(model, data, month, scales), training)
            del model, scales
        results["lstm"]["test"]["monthly"].append(json.loads(path.read_text(encoding="utf-8")))
        del entries, y, weights
        gc.collect()
    for family, result in results.items():
        rows = result["test"]["monthly"]
        result["test"]["metrics"] = matrix_metrics(sum(np.asarray(row["metrics"]["matriz_confusion"]) for row in rows))
        if family == "lstm":
            result["validation"] = {"metrics": selected["metrics"], "epochs": epochs, "history": validations[family]}
        else:
            result["validation"] = {"metrics": matrix_metrics(sum(np.asarray(row["metrics"]["matriz_confusion"]) for row in validations[family])),
                                    "monthly": validations[family]}
    for i, period in enumerate(TEST):
        rows = [results[family]["test"]["monthly"][i] for family in FAMILIES]
        if len({tuple(row["metrics"]["support"]) for row in rows}) != 1:
            raise AssertionError(f"Las etiquetas de prueba difieren entre modelos en {period}")
        if len({row["training"]["sample_sha256"] for row in rows}) != 1:
            raise AssertionError(f"Las muestras difieren entre modelos en {period}")
    save_json(OUT / "final_results.json", {"protocol": protocol, "source_metadata": source,
              "lstm_epoch_selection": selected, "results": results})
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), facecolor="white")
    for ax, family, title in zip(axes, FAMILIES, ("XGBoost", "LSTM", "Random Forest")):
        render_matrix(ax, results[family]["test"]["metrics"], title)
    fig.suptitle("Predicción mensual · Buffer 300 m · Sin alumbrado · Coordenadas originales", fontsize=21, weight="bold")
    fig.text(.5, .91, "Tramo × mes · Entrenamiento móvil de 3 meses · Prueba enero–mayo de 2026", ha="center", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .82), w_pad=3)
    image = OUT / "matrices_mensuales_tres_modelos_300m.png"
    fig.savefig(image, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"IMAGEN {image}", flush=True)
    for family, result in results.items():
        m = result["test"]["metrics"]
        print(f"FINAL {family}: accuracy={m['accuracy']:.6f}; recall_medio={m['recall'][1]:.6f}; "
              f"recall_alto={m['recall'][2]:.6f}; precision_alto={m['precision'][2]:.6f}", flush=True)


if __name__ == "__main__":
    main()
