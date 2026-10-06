"""Ensayo aislado por turnos: XGBoost/LSTM, con/sin alumbrado.

Solo coordenadas originales; buffer 300 m; historia desde 2018.
Ejecutar con .venv/Scripts/python tools/probar_alumbrado_turnos_originales_xgb_lstm.py
"""
from __future__ import annotations

import functools
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from comparar_factores_67 import graph_data
from app.flujo_entrenamiento.riesgo import agregar_riesgo_base
from buffer300.data import PERIODS, THRESHOLD, save_json
from buffer300.features import Features
from buffer300.modelos import fit_model, predictions, evaluate, matrix_metrics, score
from buffer300.report import render_matrix
from probar_buffer_300_originales import (
    PANEL as MONTHLY_PANEL, PANEL_META as MONTHLY_META, SOURCE, SOURCE_AUDIT, fingerprint,
)
from probar_alumbrado_fecha_xgb_lstm import (
    LIGHT_HISTORY, TEMPORAL, LIGHT_NAMES, prepare as prepare_lighting, seed,
)

DATA = ROOT / "Backend/data/experimentos_sidpol/alumbrado_turnos_originales_v1"
OUT = ROOT / "outputs/evaluacion-modelos-sidpol/alumbrado_turnos_originales_v1"
TURN_PANEL = DATA / "gravedad_turnos_300m_originales.npy"
TURN_META = DATA / "gravedad_turnos_300m_originales.json"
TURNS = ("madrugada", "manana", "tarde", "noche")
TURN_COLUMNS = (10, 7, 8, 9)
VALIDATION = ("2025-11", "2025-12")
TEST = tuple(f"2026-{month:02d}" for month in range(1, 6))
FIRST, SAMPLE, WINDOW, CHANNELS, HIDDEN, MAX_EPOCHS = 3, 300, 12, 18, 32, 4
SPEC = {"name": "contexto_historia_original_por_turno", "features": "contexto_turno", "balance": 1.25}


def prepare_turns():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    signature = fingerprint()
    original_meta = json.loads(MONTHLY_META.read_text(encoding="utf-8"))
    if original_meta["fingerprint"] != signature:
        raise ValueError("El panel mensual no coincide con la fuente original actual")
    if TURN_META.exists() and TURN_PANEL.exists():
        meta = json.loads(TURN_META.read_text(encoding="utf-8"))
        if meta["fingerprint"] != signature:
            raise ValueError("Cambió la fuente; use otra versión del experimento")
        print("PANEL ORIGINAL POR TURNOS YA PREPARADO", flush=True)
        return meta
    fields = ["periodo", "hora", "latitud", "longitud", "turno", "subtipo_delito", "modalidad"]
    crimes = pd.read_csv(SOURCE, usecols=fields)
    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if len(crimes) != audit["totals"]["geolocalizados"]:
        raise ValueError("El origen no coincide con la auditoría de coordenadas originales")
    hour = pd.to_numeric(crimes.hora, errors="raise").to_numpy()
    if not ((hour >= 0) & (hour < 24)).all():
        raise ValueError("Horas fuera de 0–23")
    turn_code = (hour // 6).astype(np.int8)
    corrected = np.asarray(TURNS)[turn_code] != crimes.turno.to_numpy()
    corrected_by_month = crimes.loc[corrected, "periodo"].value_counts().sort_index().to_dict()
    print(f"TURNOS RECALCULADOS SEGUN HORA DEL HECHO: {int(corrected.sum()):,} diferencias", flush=True)
    crimes = agregar_riesgo_base(crimes)
    tramos, lines, transformer = graph_data()
    n = len(tramos)
    del tramos
    monthly = np.load(MONTHLY_PANEL, mmap_mode="r")
    if n != monthly.shape[1]:
        raise AssertionError("La red vial y el panel original no coinciden")
    points = shapely.points(*transformer.transform(crimes.longitud.to_numpy(), crimes.latitud.to_numpy()))
    tree = STRtree(lines)
    period_code = pd.Categorical(crimes.periodo, categories=PERIODS).codes
    weights = crimes.peso_delito.to_numpy(dtype=np.float32)
    panel = np.lib.format.open_memmap(TURN_PANEL, mode="w+", dtype=np.float32,
                                     shape=(len(PERIODS), 4, n, 2))
    rows = []
    for month, period in enumerate(PERIODS):
        indexes = np.flatnonzero(period_code == month)
        if len(indexes) != audit["monthly"][period]["geolocalizados"]:
            raise ValueError(f"Filas originales no reconciliadas en {period}")
        local, segments = tree.query(points[indexes], predicate="dwithin", distance=300.)
        events = indexes[local]
        distance = shapely.distance(points[events], lines[segments])
        weighted = weights[events] * np.exp(-.5 * (distance / 100.)**2)
        for turn in range(4):
            selected = turn_code[events] == turn
            panel[month, turn, :, 0] = np.bincount(segments[selected], minlength=n)
            panel[month, turn, :, 1] = np.bincount(segments[selected], weights=weighted[selected], minlength=n)
        if not np.array_equal(panel[month, :, :, 0].sum(axis=0), monthly[month, :, 0]):
            raise AssertionError(f"No se reproduce el conteo mensual original {period}")
        delta = float(np.max(np.abs(panel[month, :, :, 1].sum(axis=0)-monthly[month, :, 1])))
        if delta > .001:
            raise AssertionError(f"No se reproduce la gravedad mensual original {period}: {delta}")
        rows.append({"periodo": period, "delitos_originales": len(indexes),
                     "pares_delito_tramo": len(segments), "max_diferencia_gravedad": delta})
        if month % 6 == 5 or month == len(PERIODS)-1:
            panel.flush()
            print(f"PANEL TURNOS ORIGINAL {period}: {len(indexes):,} delitos; controles correctos", flush=True)
    panel.flush()
    meta = {"fingerprint": signature, "shape": list(panel.shape), "monthly": rows,
            "source_excel": original_meta["source_excel"], "only_original_coordinates": True,
            "original_crimes": len(crimes), "excluded_missing_coordinates": audit["totals"]["sin_coordenadas"],
            "turn_rule": "Hora del hecho: madrugada 00–05, mañana 06–11, tarde 12–17, noche 18–23",
            "monthly_panel_reproduced": True, "turns_corrected_from_event_hour": int(corrected.sum()),
            "turns_corrected_by_month": corrected_by_month}
    save_json(TURN_META, meta)
    del crimes, panel, points, lines, tree, monthly
    gc.collect()
    return meta


class CorrectedMonthlyPanel:
    """Vista del panel mensual que sustituye los cuatro conteos por turno."""
    def __init__(self, original, turns):
        self.original, self.turns = original, turns

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        if len(keys) == 3 and isinstance(keys[2], (int, np.integer)):
            column = keys[2]
            if column not in TURN_COLUMNS:
                return self.original[key]
            values = self.turns[keys[0], TURN_COLUMNS.index(column), :, 0]
            return values[keys[1]] if isinstance(keys[0], (int, np.integer)) else values[:, keys[1]]
        values = self.original[keys[0]].copy()
        for turn, column in enumerate(TURN_COLUMNS):
            values[..., column] = self.turns[keys[0], turn, :, 0]
        if len(keys) == 1:
            return values
        remaining = keys[1:] if isinstance(keys[0], (int, np.integer)) else (slice(None),)+keys[1:]
        return values[remaining]


class TurnData:
    def __init__(self):
        self.monthly = Features(panel_path=MONTHLY_PANEL)
        self.panel = np.load(TURN_PANEL, mmap_mode="r")
        self.monthly.panel = CorrectedMonthlyPanel(self.monthly.panel, self.panel)
        self.temporal = np.load(TEMPORAL, mmap_mode="r")
        self.light = np.load(LIGHT_HISTORY, mmap_mode="r")
        self.factor, self.static, self.n = self.monthly.factor, self.monthly.static, self.monthly.n
        self.neighbors = self.monthly._neighbors()

    def risk(self, month, turn):
        return self.panel[month, turn, :, 1] / self.factor

    def target(self, month, turn):
        risk = self.risk(month, turn)
        return np.where(risk >= THRESHOLD, 2, np.where(risk > 0, 1, 0)).astype(np.int8)

    @functools.lru_cache(maxsize=1)
    def monthly_base(self, month):
        return self.monthly.make(month, "contexto")

    def lighting(self, month, turn, ids):
        light = self.light[month, ids]
        return np.column_stack([light, light * float(turn in (0, 3))]).astype(np.float32)

    def xgb(self, month, turn, variant, ids):
        if month < 3:
            raise ValueError("Se requieren tres meses pasados")
        previous = np.stack([self.risk(month-lag, turn) for lag in (1, 2, 3)])
        recent_mean = previous.mean(axis=0)
        counts = self.monthly.panel[month-3:month, :, TURN_COLUMNS[turn]].mean(axis=0)/self.factor
        total = self.monthly.risk[month-3:month].mean(axis=0)
        neighbors = self.neighbors[ids, :8]
        extra = np.column_stack([
            np.full(len(ids), float(turn == k), dtype=np.float32) for k in range(4)
        ] + [previous[k, ids] for k in range(3)] + [
            recent_mean[ids], previous.std(axis=0)[ids], previous.max(axis=0)[ids],
            (previous > 0).mean(axis=0)[ids], (previous >= THRESHOLD).mean(axis=0)[ids],
            counts[ids], recent_mean[ids]/(total[ids]+.05),
            previous[0, neighbors].mean(axis=1), recent_mean[neighbors].mean(axis=1),
        ])
        x = np.column_stack([self.monthly_base(month)[ids], extra]).astype(np.float32)
        return np.column_stack([x, self.lighting(month, turn, ids)]) if variant == "con_alumbrado" else x

    @functools.lru_cache(maxsize=64)
    def temporal_turn(self, month, turn):
        risk = self.risk(month, turn)
        count = self.panel[month, turn, :, 0]/self.factor
        return np.log1p(np.column_stack([count, risk, risk[self.neighbors[:, :8]].mean(axis=1),
                                       risk[self.neighbors].mean(axis=1)])).astype(np.float16)

    def sequence(self, month, turn, ids):
        result = np.zeros((len(ids), WINDOW, CHANNELS), dtype=np.float32)
        start = max(0, month-WINDOW)
        for prior in range(start, month):
            position = WINDOW-month+prior
            result[:, position, :14] = self.temporal[prior, ids]
            for other, column in enumerate(TURN_COLUMNS):
                result[:, position, column] = np.log1p(self.panel[prior, other, ids, 0]/self.factor[ids])
            result[:, position, 14:] = self.temporal_turn(prior, turn)[ids]
        return result

    def context(self, month, turn, variant, ids):
        static = self.static[ids]
        season = np.broadcast_to([np.sin(2*np.pi*(month % 12)/12), np.cos(2*np.pi*(month % 12)/12)], (len(ids), 2))
        turn_hot = np.broadcast_to(np.eye(4, dtype=np.float32)[turn], (len(ids), 4))
        x = np.column_stack([static[:, :2], np.log1p(static[:, 2]), season, turn_hot]).astype(np.float32)
        return np.column_stack([x, self.lighting(month, turn, ids)]) if variant == "con_alumbrado" else x

    def selection(self, last):
        rng = np.random.default_rng(42)
        records, counts, digest = [], np.zeros(3, dtype=np.int64), hashlib.sha256()
        for month in range(FIRST, PERIODS.index(last)+1):
            for turn in range(4):
                y = self.target(month, turn)
                counts += np.bincount(y, minlength=3)
                for cls in range(3):
                    candidates = np.flatnonzero(y == cls)
                    keep = min(SAMPLE, len(candidates))
                    if keep:
                        ids = rng.choice(candidates, keep, replace=False)
                        digest.update(np.asarray([month, turn, cls], dtype=np.int32).tobytes())
                        digest.update(ids.astype(np.int32).tobytes())
                        records.append((month, turn, ids, cls, len(candidates)/keep))
        y = np.concatenate([np.full(len(ids), cls, dtype=np.int64) for _, _, ids, cls, _ in records])
        ratios = np.concatenate([np.full(len(ids), ratio, dtype=np.float32) for _, _, ids, _, ratio in records])
        weights = ratios*(counts.sum()/(3*counts[y]))**SPEC["balance"]
        weights = (weights/weights.mean()).astype(np.float32)
        return records, y, weights, {"first_target": PERIODS[FIRST], "last_target": last,
            "population_counts": counts.tolist(), "sampled_rows": len(y), "sample_sha256": digest.hexdigest(),
            "sampling_per_class_month_turn": SAMPLE, "earliest_history": "2018-01"}

    def training_xgb(self, records, variant):
        rows = []
        for month, turn in dict.fromkeys((r[0], r[1]) for r in records):
            ids = np.concatenate([r[2] for r in records if r[0] == month and r[1] == turn])
            rows.append(self.xgb(month, turn, variant, ids))
        return np.concatenate(rows)

    def training_lstm(self, records, variant):
        total = sum(len(r[2]) for r in records)
        seq = np.empty((total, WINDOW, CHANNELS), dtype=np.float32)
        context = np.empty((total, 17 if variant == "con_alumbrado" else 9), dtype=np.float32)
        offset = 0
        for month, turn, ids, _, _ in records:
            right = offset+len(ids)
            seq[offset:right] = self.sequence(month, turn, ids)
            context[offset:right] = self.context(month, turn, variant, ids)
            offset = right
        return seq, context


class TurnLSTM(nn.Module):
    def __init__(self, context_features):
        super().__init__()
        self.sequence = nn.LSTM(CHANNELS, HIDDEN, batch_first=True)
        self.head = nn.Sequential(nn.Linear(HIDDEN+context_features, 32), nn.ReLU(), nn.Dropout(.1), nn.Linear(32, 3))

    def forward(self, seq, context):
        _, (hidden, _) = self.sequence(seq)
        return self.head(torch.cat([hidden[-1], context], dim=1))


def normalize(seq, context):
    mean, std = seq.mean(axis=(0, 1)), np.maximum(seq.std(axis=(0, 1)), .05)
    mean[13], std[13] = 0, 1  # máscara de meses disponibles del historial mensual.
    cmean, cstd = context.mean(axis=0), np.maximum(context.std(axis=0), 1e-6)
    seq -= mean
    seq /= std
    context -= cmean
    context /= cstd
    return {"mean": mean, "std": std, "context_mean": cmean, "context_std": cstd}


def lstm_predict(model, data, month, turn, variant, scales):
    result = np.empty((data.n, 3), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for left in range(0, data.n, 8192):
            ids = np.arange(left, min(left+8192, data.n))
            seq = (data.sequence(month, turn, ids)-scales["mean"])/scales["std"]
            context = (data.context(month, turn, variant, ids)-scales["context_mean"])/scales["context_std"]
            result[ids] = torch.softmax(model(torch.from_numpy(seq), torch.from_numpy(context)), dim=1).numpy()
    return result


def eval_periods(model, data, family, variant, periods, scales=None, save=False):
    monthly, turns = [], {turn: np.zeros((3, 3), dtype=np.int64) for turn in TURNS}
    for period in periods:
        month = PERIODS.index(period)
        matrices = {}
        for turn, label in enumerate(TURNS):
            truth = data.target(month, turn)
            if family == "xgboost":
                probability = predictions(model, data.xgb(month, turn, variant, np.arange(data.n)))
            else:
                probability = lstm_predict(model, data, month, turn, variant, scales)
            metrics = evaluate(truth, probability)
            matrices[label] = metrics
            turns[label] += np.asarray(metrics["matriz_confusion"], dtype=np.int64)
            if save:
                np.savez_compressed(DATA / f"predicciones_{family}_{variant}_{period}_{label}.npz",
                                    truth=truth, probability=probability)
        total = matrix_metrics(sum(np.asarray(m["matriz_confusion"]) for m in matrices.values()))
        row = {"periodo": period, "metrics": total, "por_turno": matrices}
        monthly.append(row)
        if save:
            save_json(OUT / f"resultado_{family}_{variant}_{period}.json", row)
        print(f"EVAL TURNOS {family}/{variant} {period}: acc={total['accuracy']:.4f}, alto={total['recall'][2]:.4f}", flush=True)
    return {"monthly": monthly, "metrics": matrix_metrics(sum(turns.values())),
            "por_turno": {turn: matrix_metrics(matrix) for turn, matrix in turns.items()},
            "noche_madrugada": matrix_metrics(turns["noche"]+turns["madrugada"])}


def train_lstm(data, records, y, weights, variant, epochs, validate):
    seed()
    seq, context = data.training_lstm(records, variant)
    scales = normalize(seq, context)
    loader = DataLoader(TensorDataset(torch.from_numpy(seq), torch.from_numpy(context),
                        torch.from_numpy(y), torch.from_numpy(weights)), batch_size=2048, shuffle=True, num_workers=0)
    model = TurnLSTM(context.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    history, best = [], None
    for epoch in range(1, epochs+1):
        start = time.perf_counter()
        model.train()
        losses = []
        for x, ctx, target, weight in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = (F.cross_entropy(model(x, ctx), target, reduction="none")*weight).sum()/weight.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": time.perf_counter()-start}
        if validate:
            validation = eval_periods(model, data, "lstm", variant, VALIDATION, scales)
            row["validation"] = validation
            current = score(validation["metrics"])
            if best is None or current > best["score"]:
                best = {"epoch": epoch, "score": current, "validation": validation}
        history.append(row)
        print(f"LSTM TURNOS {variant} epoca={epoch}/{epochs}, loss={row['loss']:.4f}", flush=True)
    return model, scales, history, best


def run(data, family, variant):
    path = OUT / f"resultado_{family}_{variant}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    records, y, weights, selection = data.selection("2025-10")
    seed()
    print(f"ENTRENANDO TURNOS {family}/{variant}: {selection}", flush=True)
    epochs, history = None, None
    if family == "xgboost":
        x = data.training_xgb(records, variant)
        model = fit_model(SPEC, x, y, weights)
        validation = eval_periods(model, data, family, variant, VALIDATION)
        del x, model
    else:
        model, scales, history, chosen = train_lstm(data, records, y, weights, variant, MAX_EPOCHS, True)
        epochs, validation = chosen["epoch"], chosen["validation"]
        del model, scales
    del records, y, weights
    gc.collect()
    records, y, weights, final_selection = data.selection("2025-12")
    seed()
    print(f"AJUSTE FINAL TURNOS {family}/{variant}: {final_selection['sampled_rows']:,}; epocas={epochs}", flush=True)
    if family == "xgboost":
        x = data.training_xgb(records, variant)
        model = fit_model(SPEC, x, y, weights)
        scales = None
        joblib.dump({"model": model, "training": final_selection, "variant": variant,
                     "source": str(TURN_PANEL)}, DATA / f"modelo_{family}_{variant}.joblib", compress=3)
        del x
    else:
        model, scales, _, _ = train_lstm(data, records, y, weights, variant, epochs, False)
        torch.save({"state_dict": model.state_dict(), "scales": scales, "training": final_selection,
                    "variant": variant, "epochs": epochs, "architecture": {"window": WINDOW, "channels": CHANNELS,
                    "hidden": HIDDEN, "context_features": 17 if variant == "con_alumbrado" else 9}},
                   DATA / f"modelo_{family}_{variant}.pt")
    test = eval_periods(model, data, family, variant, TEST, scales, save=True)
    result = {"family": family, "variant": variant, "training_selection": selection,
              "training_final": final_selection, "validation": validation, "test": test,
              "epochs_selected_by_2025_validation": epochs, "validation_history": history,
              "test_used_for_selection": False}
    save_json(path, result)
    del model, records, y, weights
    gc.collect()
    return result


def main():
    seed()
    lighting = prepare_lighting()
    original = prepare_turns()
    protocol = {"script": str(Path(__file__).resolve()), "source": str(SOURCE),
        "only_original_coordinates": True, "spatial_method": "Buffer euclidiano 300 m; gaussiana sigma 100 m",
        "unit": "tramo × mes × turno", "high_threshold": THRESHOLD,
        "turn_rule": original["turn_rule"], "history_starts": "2018-01",
        "selection_train_targets": ["2018-04", "2025-10"], "validation": list(VALIDATION),
        "final_train_targets": ["2018-04", "2025-12"], "test": list(TEST),
        "models_fixed_during_test": True, "test_status": "Retrospectivo; estos meses ya fueron examinados",
        "sampling_per_class_month_turn": SAMPLE, "seed": 42, "class_balance_power": SPEC["balance"],
        "xgboost": SPEC, "lstm": {"window": WINDOW, "channels": CHANNELS, "hidden": HIDDEN, "max_epochs": MAX_EPOCHS},
        "light_variables": LIGHT_NAMES, "light_interactions": "Cuatro variables × indicador de noche o madrugada",
        "service_date_cutoff": "Puesta en servicio <= primer día del mes objetivo",
        "lighting_limitation": lighting["rule"], "monthly_metrics_not_directly_comparable": True,
        "label_note": "Umbral numérico anterior conservado; gravedad acumulada por turno, no por todo el mes",
        "test_used_for_selection": False}
    save_json(OUT / "protocolo.json", protocol)
    data = TurnData()
    data.monthly.verify_causality()
    results = {f"{family}_{variant}": run(data, family, variant)
               for family in ("xgboost", "lstm") for variant in ("sin_alumbrado", "con_alumbrado")}
    if len({tuple(r["test"]["metrics"]["support"]) for r in results.values()}) != 1:
        raise AssertionError("Los modelos no comparten las mismas etiquetas de prueba")
    if len({r["training_final"]["sample_sha256"] for r in results.values()}) != 1:
        raise AssertionError("Los modelos no comparten las mismas muestras de entrenamiento")
    save_json(OUT / "final_results.json", {"protocol": protocol, "source_audit": original, "results": results})
    for subset, suffix, subtitle in (("metrics", "todos_turnos", "Todos los turnos"),
                                     ("noche_madrugada", "noche_madrugada", "Solo noche y madrugada")):
        fig, axes = plt.subplots(2, 2, figsize=(16, 15), facecolor="white")
        for row, family in enumerate(("xgboost", "lstm")):
            for col, variant in enumerate(("sin_alumbrado", "con_alumbrado")):
                render_matrix(axes[row, col], results[f"{family}_{variant}"]["test"][subset],
                              family.upper()+" · "+variant.replace("_", " "))
        fig.suptitle("Buffer 300 m · Delitos originales · Predicción por turno", fontsize=20, weight="bold")
        fig.text(.5, .945, subtitle+" · Enero–mayo de 2026 · Historial desde 2018", ha="center", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .92), h_pad=4, w_pad=3)
        path = OUT / f"matrices_{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"IMAGEN {path}", flush=True)
    for name, result in results.items():
        m, dark = result["test"]["metrics"], result["test"]["noche_madrugada"]
        print(f"FINAL {name}: acc={m['accuracy']:.6f}; recall_medio={m['recall'][1]:.6f}; "
              f"recall_alto={m['recall'][2]:.6f}; precision_alto={m['precision'][2]:.6f}; "
              f"noche_madrugada_acc={dark['accuracy']:.6f}; noche_madrugada_alto={dark['recall'][2]:.6f}", flush=True)


if __name__ == "__main__":
    main()
