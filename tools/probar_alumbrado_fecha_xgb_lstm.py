"""Alumbrado fijo vs fecha de puesta en servicio, XGBoost y LSTM mensuales.

Delitos originales, buffer 300 m, cortes y muestras comunes desde 2018.
Ejecutar: .venv/Scripts/python tools/probar_alumbrado_fecha_xgb_lstm.py
"""
from __future__ import annotations

import gc
import json
import random
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from buffer300.data import PERIODS, THRESHOLD, load_static, save_json, DATA as BASE_DATA
from buffer300.features import Features
from buffer300.modelos import fit_model, predictions, evaluate, matrix_metrics, score
from buffer300.report import render_matrix
from probar_alumbrado_buffer_300 import projected
from probar_buffer_300_originales import PANEL, PANEL_META, OUT as ORIGINAL_OUT

DATA = ROOT / "Backend/data/experimentos_sidpol/alumbrado_fechas_originales_v1"
OUT = ROOT / "outputs/evaluacion-modelos-sidpol/alumbrado_fechas_xgb_lstm_v1"
SOURCE = Path(r"C:\Users\Alonso\Downloads\equipo_ap_luminaria_1.csv")
LIGHT_HISTORY = DATA / "alumbrado_por_fecha.npy"
LIGHT_STATIC = DATA / "alumbrado_estatico.npy"
TEMPORAL = DATA / "secuencias_delitos_originales.npy"
LIGHT_NAMES = ["luminarias_50m_log", "luminarias_100m_log", "distancia_luminaria_log_m", "luminaria_30m"]
FIRST = PERIODS.index("2018-04")
VALIDATION = ("2025-11", "2025-12")
TEST = tuple(f"2026-{m:02d}" for m in range(1, 6))
SEED, SAMPLE, WINDOW, CHANNELS, HIDDEN, MAX_EPOCHS = 42, 1200, 12, 14, 32, 4
SPEC = {"name": "contexto_historia_original", "features": "contexto", "balance": 1.25}


def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(6, torch.get_num_threads()))


def light_variables(xy, road_xy):
    tree = cKDTree(xy)
    distance, _ = tree.query(road_xy, k=1, workers=4)
    count50 = tree.query_ball_point(road_xy, r=50, return_length=True, workers=4)
    count100 = tree.query_ball_point(road_xy, r=100, return_length=True, workers=4)
    return np.column_stack([np.log1p(count50), np.log1p(count100),
                            np.log1p(np.minimum(distance, 500)), distance <= 30]).astype(np.float32)


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    source_stat = SOURCE.stat()
    signature = {"source": str(SOURCE), "size": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns}
    audit_path = OUT / "auditoria_fechas.json"
    marker = DATA / "alumbrado_complete.json"
    if marker.exists() and LIGHT_HISTORY.exists() and LIGHT_STATIC.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit["fingerprint"] != signature:
            raise ValueError("Cambió el inventario; cree otra versión")
    else:
        old_audit = json.loads((ROOT / "outputs/evaluacion-modelos-sidpol/alumbrado_buffer_300_v1/auditoria_alumbrado.json").read_text(encoding="utf-8"))
        bounds = old_audit["bbox"]
        chunks = []
        for chunk in pd.read_csv(SOURCE, usecols=["codemp", "codluminaria", "fecha_emision", "fecpuestaservicio", "coordenada_x", "coordenada_y"],
                                 dtype={"codemp": "string", "codluminaria": "string", "fecha_emision": "string", "fecpuestaservicio": "string"},
                                 chunksize=250000, encoding_errors="replace"):
            x = pd.to_numeric(chunk.coordenada_x, errors="coerce")
            y = pd.to_numeric(chunk.coordenada_y, errors="coerce")
            keep = x.between(bounds["lon_min"], bounds["lon_max"]) & y.between(bounds["lat_min"], bounds["lat_max"])
            if keep.any():
                chunks.append(chunk.loc[keep].copy())
        lights = pd.concat(chunks, ignore_index=True)
        dates = pd.to_datetime(lights.fecpuestaservicio, format="%d/%m/%Y", errors="coerce")
        if dates.isna().any() or lights.duplicated(["codemp", "codluminaria"]).any():
            raise ValueError("Fechas incompletas o luminarias duplicadas; revise la selección antes de comparar")
        raw_xy = lights[["coordenada_x", "coordenada_y"]].to_numpy(dtype=np.float64)
        static, _ = load_static()
        road_xy = projected(static[:, [1, 0]])
        xy = projected(raw_xy)
        all_lights = light_variables(xy, road_xy)
        reference = np.load(BASE_DATA / "alumbrado_estatico_v1.npy")
        if not np.array_equal(all_lights, reference):
            raise AssertionError("No se reprodujeron las variables del inventario fijo anterior")
        np.save(LIGHT_STATIC, all_lights)
        history = np.lib.format.open_memmap(LIGHT_HISTORY, mode="w+", dtype=np.float32,
                                           shape=(len(PERIODS), len(static), 4))
        rows = []
        for month, period in enumerate(PERIODS):
            as_of = pd.Timestamp(period + "-01")
            keep = (dates <= as_of).to_numpy()
            history[month] = all_lights if keep.all() else light_variables(xy[keep], road_xy)
            rows.append({"periodo": period, "known_put_into_service": int(keep.sum()), "as_of": str(as_of.date())})
            if month % 6 == 5 or month == len(PERIODS)-1:
                history.flush()
                print(f"ALUMBRADO POR FECHA {period}: {keep.sum():,}/{len(lights):,}", flush=True)
        if not np.array_equal(history[PERIODS.index("2025-01"):], np.broadcast_to(all_lights, history[PERIODS.index("2025-01"):].shape)):
            raise AssertionError("La equivalencia de alumbrado 2025–2026 no se cumple")
        audit = {"fingerprint": signature, "local_lights": len(lights), "valid_service_dates": int(dates.notna().sum()),
                 "service_min": str(dates.min().date()), "service_max": str(dates.max().date()),
                 "by_year": {str(k): int(v) for k, v in dates.dt.year.value_counts().sort_index().items()},
                 "most_common_dates": lights.fecpuestaservicio.value_counts().head(10).to_dict(),
                 "emission_dates": lights.fecha_emision.value_counts().to_dict(), "monthly": rows,
                 "current_rolling_protocol_equivalent": True,
                 "rule": "Puesta en servicio <= primer día del mes objetivo. Inventario actual con fecha de emisión 2024; no reconstruye retiros o averías.",
                 "static_reference_reproduced_exactly": True}
        save_json(audit_path, audit)
        save_json(marker, {"complete": True, "shape": list(history.shape)})
        del history, lights, chunks, xy, road_xy
        gc.collect()

    temporal_marker = DATA / "secuencias_complete.json"
    if not temporal_marker.exists() or not TEMPORAL.exists():
        panel = np.load(PANEL, mmap_mode="r")
        static, factor = load_static()
        neighbors = np.load(BASE_DATA / "vecinos_32.npy", mmap_mode="r")
        temporal = np.lib.format.open_memmap(TEMPORAL, mode="w+", dtype=np.float16,
                                             shape=(len(PERIODS), len(static), CHANNELS))
        for month, period in enumerate(PERIODS):
            raw = np.maximum(panel[month] / factor[:, None], 0)
            risk = raw[:, 1]
            temporal[month, :, :11] = np.log1p(raw).astype(np.float16)
            temporal[month, :, 11] = np.log1p(risk[neighbors[:, :8]].mean(axis=1)).astype(np.float16)
            temporal[month, :, 12] = np.log1p(risk[neighbors].mean(axis=1)).astype(np.float16)
            temporal[month, :, 13] = 1  # distingue historia observada de padding al inicio de 2018.
            if month % 12 == 11:
                temporal.flush()
                print(f"SECUENCIAS ORIGINALES {period}", flush=True)
        temporal.flush()
        save_json(temporal_marker, {"complete": True, "source_panel": str(PANEL),
                   "shape": list(temporal.shape), "fingerprint": json.loads(PANEL_META.read_text(encoding="utf-8"))["fingerprint"]})
        del temporal, panel
        gc.collect()
    return audit


class ModelData(Features):
    def __init__(self):
        super().__init__(panel_path=PANEL)
        self.fixed_light = np.load(LIGHT_STATIC)
        self.date_light = np.load(LIGHT_HISTORY, mmap_mode="r")
        self.temporal = np.load(TEMPORAL, mmap_mode="r")

    def lighting(self, month, variant):
        return self.fixed_light if variant == "estatico" else self.date_light[month]

    def xgb(self, month, variant, ids=None):
        x = self.make(month, "contexto")
        light = self.lighting(month, variant)
        if ids is not None:
            x, light = x[ids], light[ids]
        return np.column_stack([x, light]).astype(np.float32)

    def lstm_context(self, month, variant, ids):
        static = self.static[ids]
        season = np.broadcast_to([np.sin(2*np.pi*(month % 12)/12), np.cos(2*np.pi*(month % 12)/12)], (len(ids), 2))
        return np.column_stack([static[:, :2], np.log1p(static[:, 2]), self.lighting(month, variant)[ids], season]).astype(np.float32)

    def sequence(self, month, ids):
        start = max(0, month-WINDOW)
        result = np.zeros((len(ids), WINDOW, CHANNELS), dtype=np.float32)
        known = np.asarray(self.temporal[start:month][:, ids], dtype=np.float32).transpose(1, 0, 2)
        result[:, WINDOW-len(known[0]):] = known
        return result

    def selection(self, last):
        rng = np.random.default_rng(SEED)
        records, counts = [], np.zeros(3, dtype=np.int64)
        for month in range(FIRST, PERIODS.index(last)+1):
            y = self.target([month])
            counts += np.bincount(y, minlength=3)
            for cls in range(3):
                candidates = np.flatnonzero(y == cls)
                keep = min(SAMPLE, len(candidates))
                if keep:
                    records.append((month, rng.choice(candidates, keep, replace=False), cls, len(candidates)/keep))
        y = np.concatenate([np.full(len(ids), cls, dtype=np.int64) for _, ids, cls, _ in records])
        ratio = np.concatenate([np.full(len(ids), ratio, dtype=np.float32) for _, ids, _, ratio in records])
        weight = ratio*(counts.sum()/(3*counts[y]))**1.25
        weight = (weight/weight.mean()).astype(np.float32)
        return records, y, weight, {"first_target": PERIODS[FIRST], "last_target": last,
                                   "population_counts": counts.tolist(), "sampled_rows": len(y),
                                   "sampling_per_class_month": SAMPLE, "earliest_history": "2018-01"}

    def training_xgb(self, records, variant):
        rows = []
        for month in sorted({r[0] for r in records}):
            selected = [r[1] for r in records if r[0] == month]
            rows.append(self.xgb(month, variant, np.concatenate(selected)))
        return np.concatenate(rows)

    def training_lstm(self, records, variant):
        total = sum(len(ids) for _, ids, _, _ in records)
        seq = np.empty((total, WINDOW, CHANNELS), dtype=np.float32)
        sta = np.empty((total, 9), dtype=np.float32)
        offset = 0
        for month, ids, _, _ in records:
            right = offset+len(ids)
            seq[offset:right] = self.sequence(month, ids)
            sta[offset:right] = self.lstm_context(month, variant, ids)
            offset = right
        return seq, sta


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence = nn.LSTM(CHANNELS, HIDDEN, batch_first=True)
        self.head = nn.Sequential(nn.Linear(HIDDEN+9, 32), nn.ReLU(), nn.Dropout(.1), nn.Linear(32, 3))

    def forward(self, seq, context):
        _, (hidden, _) = self.sequence(seq)
        return self.head(torch.cat([hidden[-1], context], dim=1))


def normalize_training(seq, context):
    mean = seq.mean(axis=(0, 1))
    std = np.maximum(seq.std(axis=(0, 1)), .05)
    mean[-1], std[-1] = 0, 1
    smean = context.mean(axis=0)
    sstd = np.maximum(context.std(axis=0), 1e-6)
    seq -= mean
    seq /= std
    context -= smean
    context /= sstd
    return {"mean": mean, "std": std, "context_mean": smean, "context_std": sstd}


def lstm_predict(model, data, month, variant, scales):
    probability = np.empty((data.n, 3), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for left in range(0, data.n, 8192):
            ids = np.arange(left, min(left+8192, data.n))
            seq = (data.sequence(month, ids)-scales["mean"])/scales["std"]
            context = (data.lstm_context(month, variant, ids)-scales["context_mean"])/scales["context_std"]
            logits = model(torch.from_numpy(seq), torch.from_numpy(context))
            probability[ids] = torch.softmax(logits, dim=1).numpy()
    return probability


def eval_periods(model, data, variant, periods, family, scales=None, save=False):
    rows = []
    for period in periods:
        month = PERIODS.index(period)
        probability = predictions(model, data.xgb(month, variant)) if family == "xgboost" else lstm_predict(model, data, month, variant, scales)
        truth = data.target([month])
        row = {"periodo": period, "metrics": evaluate(truth, probability)}
        rows.append(row)
        if save:
            np.savez_compressed(DATA / f"predicciones_{family}_{variant}_{period}.npz", truth=truth, probability=probability)
            save_json(OUT / f"resultado_{family}_{variant}_{period}.json", row)
        print(f"EVAL {family}/{variant} {period}: acc={row['metrics']['accuracy']:.4f} alto={row['metrics']['recall'][2]:.4f}", flush=True)
    return {"monthly": rows, "metrics": matrix_metrics(sum(np.asarray(r["metrics"]["matriz_confusion"]) for r in rows))}


def train_lstm(data, records, y, weights, variant, epochs, validate):
    seed()
    seq, context = data.training_lstm(records, variant)
    scales = normalize_training(seq, context)
    dataset = TensorDataset(torch.from_numpy(seq), torch.from_numpy(context), torch.from_numpy(y), torch.from_numpy(weights))
    loader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=0)
    model = LSTM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    best, history = None, []
    for epoch in range(1, epochs+1):
        started = time.perf_counter()
        model.train()
        losses = []
        for x, sta, target, weight in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = (F.cross_entropy(model(x, sta), target, reduction="none")*weight).sum()/weight.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": time.perf_counter()-started}
        if validate:
            validation = eval_periods(model, data, variant, VALIDATION, "lstm", scales=scales)
            row["validation"] = validation
            current = score(validation["metrics"])
            if best is None or current > best["score"]:
                best = {"epoch": epoch, "score": current, "validation": validation}
        history.append(row)
        print(f"LSTM {variant} epoca={epoch}/{epochs}, loss={row['loss']:.4f}, seg={row['seconds']:.1f}", flush=True)
    return model, scales, history, best


def run(data, family, variant):
    output = OUT / f"resultado_{family}_{variant}.json"
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    records, y, weight, details = data.selection("2025-10")
    seed()
    print(f"ENTRENANDO VALIDACION {family}/{variant}: {details}", flush=True)
    if family == "xgboost":
        x = data.training_xgb(records, variant)
        model = fit_model(SPEC, x, y, weight)
        validation = eval_periods(model, data, variant, VALIDATION, family)
        epochs, history = None, None
        del x, model
    else:
        model, scales, history, chosen = train_lstm(data, records, y, weight, variant, MAX_EPOCHS, True)
        epochs, validation = chosen["epoch"], chosen["validation"]
        del model, scales
    del records, y, weight
    gc.collect()
    records, y, weight, final_details = data.selection("2025-12")
    seed()
    print(f"AJUSTE FINAL {family}/{variant}: {final_details}; epocas={epochs}", flush=True)
    if family == "xgboost":
        x = data.training_xgb(records, variant)
        model = fit_model(SPEC, x, y, weight)
        scales = None
        joblib.dump({"model": model, "variant": variant, "training": final_details, "source": str(PANEL)},
                    DATA / f"modelo_{family}_{variant}.joblib", compress=3)
        del x
    else:
        model, scales, _, _ = train_lstm(data, records, y, weight, variant, epochs, False)
        torch.save({"state_dict": model.state_dict(), "scales": scales, "training": final_details,
                    "architecture": {"window": WINDOW, "channels": CHANNELS, "hidden": HIDDEN, "context_features": 9},
                    "variant": variant, "epochs": epochs}, DATA / f"modelo_{family}_{variant}.pt")
    test = eval_periods(model, data, variant, TEST, family, scales=scales, save=True)
    result = {"family": family, "variant": variant, "training_selection": details,
              "training_final": final_details, "validation": validation, "test": test,
              "epochs_selected_by_2025_validation": epochs, "validation_training_history": history,
              "test_used_for_selection": False}
    save_json(output, result)
    del model, records, y, weight
    gc.collect()
    return result


def main():
    seed()
    audit = prepare()
    protocol = {"source_panel": str(PANEL), "source_origin": "Solo coordenadas originales del Excel limpio",
                "buffer_m": 300, "sigma_m": 100, "high_threshold": THRESHOLD, "unit": "tramo x mes",
                "selection_train_targets": ["2018-04", "2025-10"], "validation": list(VALIDATION),
                "final_train_targets": ["2018-04", "2025-12"], "test": list(TEST),
                "history_starts": "2018-01", "models_fixed_during_test": True,
                "models": {"xgboost": SPEC, "lstm": {"window": WINDOW, "hidden": HIDDEN, "max_epochs": MAX_EPOCHS}},
                "sampling_per_class_month": SAMPLE, "class_balance_power": 1.25,
                "light_variables": LIGHT_NAMES, "service_date_cutoff": "Primer dia del mes objetivo",
                "test_status": "Evaluacion retrospectiva; estos meses ya fueron examinados",
                "current_3m_protocol_dates_equal_static": audit["current_rolling_protocol_equivalent"]}
    save_json(OUT / "protocolo.json", protocol)
    data = ModelData()
    data.verify_causality()
    results = {}
    for family in ("xgboost", "lstm"):
        for variant in ("estatico", "por_fecha"):
            results[f"{family}_{variant}"] = run(data, family, variant)
    supports = {tuple(result["test"]["metrics"]["support"]) for result in results.values()}
    if len(supports) != 1:
        raise AssertionError("Los cuatro modelos no tienen las mismas etiquetas de prueba")
    current = json.loads((ORIGINAL_OUT / "final_results.json").read_text(encoding="utf-8"))["results"]["con_alumbrado"]["benchmark"]["metrics"]
    save_json(OUT / "final_results.json", {"protocol": protocol, "lighting_audit": audit, "results": results,
              "current_xgboost_3m_static_and_date_identical": current})
    fig, axes = plt.subplots(2, 2, figsize=(16, 15), facecolor="white")
    for row, family in enumerate(("xgboost", "lstm")):
        for col, variant in enumerate(("estatico", "por_fecha")):
            render_matrix(axes[row, col], results[f"{family}_{variant}"]["test"]["metrics"],
                          family.upper()+(" · alumbrado fijo" if variant == "estatico" else " · según puesta en servicio"))
    fig.suptitle("Buffer 300 m · Delitos originales · Historial desde 2018", fontsize=20, weight="bold")
    fig.text(.5, .945, "Misma prueba: enero–mayo de 2026 · tramo × mes · mismos cortes de entrenamiento", ha="center", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .92), h_pad=4, w_pad=3)
    image = OUT / "matrices_xgboost_lstm_alumbrado_fecha.png"
    fig.savefig(image, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"IMAGEN {image}", flush=True)
    for name, result in results.items():
        m = result["test"]["metrics"]
        print(f"FINAL {name}: acc={m['accuracy']:.6f}; recall_medio={m['recall'][1]:.6f}; "
              f"recall_alto={m['recall'][2]:.6f}; precision_alto={m['precision'][2]:.6f}", flush=True)


if __name__ == "__main__":
    main()
