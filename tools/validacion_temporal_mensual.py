"""Protocolo A/B/C y registro prospectivo D, independiente del despliegue.

run: selecciona solo en B, congela y confirma en C.
forecast: registra antes del mes objetivo las predicciones D (panel actualizado).
evaluate: evalúa una única vez el mes D registrado, al disponer de sus etiquetas.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

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
from buffer300.modelos import fit_model, model_parameters, predictions, evaluate, matrix_metrics
from buffer300.report import render_matrix
from probar_buffer_300_originales import PANEL, PANEL_META, SOURCE, fingerprint
from probar_alumbrado_fecha_xgb_lstm import TEMPORAL, DATA as TEMPORAL_DATA, seed, normalize_training
from modelos_mensuales import MonthlyData, MonthlyLSTM, lstm_predict, RF_PARAMETERS
from compatibilidad_experimentos import huellas_compatibles

OUT = ROOT / "outputs/evaluacion-modelos-sidpol/protocolo_rolling_2018_2026_v1"
DATA = ROOT / "Backend/data/experimentos_sidpol/protocolo_rolling_2018_2026_v1"
FAMILIES = ("xgboost", "lstm", "random_forest")
WINDOWS = ("3m", "12m", "desde2018")
# Cada origen tiene horizonte de un mes; se exploran cuatro orígenes por año.
B_PERIODS = tuple(f"{year}-{month:02d}" for year in range(2022, 2025) for month in (3, 6, 9, 12))
C_PERIODS = tuple(p for p in PERIODS if "2025-01" <= p <= "2026-08")
CAP = 90000
BALANCE = 1.25
EPOCHS = 4
SPEC = {"features": "contexto", "balance": BALANCE}
LAMBDA_ASTAR = 10.0
DEPENDENCIES = (
    Path(__file__).resolve(), ROOT / "tools/buffer300/data.py", ROOT / "tools/buffer300/features.py",
    ROOT / "tools/buffer300/modelos.py", ROOT / "tools/buffer300/report.py",
    ROOT / "tools/probar_buffer_300_originales.py", ROOT / "tools/probar_alumbrado_fecha_xgb_lstm.py",
    ROOT / "tools/modelos_mensuales.py", ROOT / "tools/comparar_factores_67.py",
    ROOT / "Backend/app/flujo_entrenamiento/riesgo.py", ROOT / "Backend/app/services/pesos_delito.py",
    ROOT / "Backend/config_optimizacion_buffer_300.json",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_hashes():
    # Reconoce los renombres y comentarios aprobados sin modificar el registro histórico.
    return huellas_compatibles(ROOT, DEPENDENCIES)


# SELECCIÓN: combina accuracy y F1 de Medio/Alto para decidir la configuración en la Fase B.
def decision_score(metrics):
    return .4 * metrics["accuracy"] + .6 * metrics["f1_medio_alto"]


def risk_cost(distance_m, probability):
    """Coste prospectivo registrado: lambda fijo, sin búsqueda posterior de rutas."""
    p = np.asarray(probability)
    risk = np.clip(.5 * p[..., 1] + p[..., 2], 0, 1)
    return np.asarray(distance_m) * (1 + LAMBDA_ASTAR * risk)


class RollingData(MonthlyData):
    def __init__(self, panel=PANEL, temporal=TEMPORAL, periods=PERIODS):
        Features.__init__(self, panel_path=panel)
        self.temporal = np.load(temporal, mmap_mode="r")
        self.periods = tuple(periods)
        if self.panel.shape != (len(self.periods), self.n, 11):
            raise ValueError("Panel, meses o número de tramos incompatibles")
        if self.temporal.shape != (len(self.periods), self.n, 14):
            raise ValueError("Secuencias incompatibles con el panel")

    def selected_features(self, month, ids):
        """Las mismas 74 variables, calculando únicamente las filas solicitadas."""
        if month < 3:
            raise ValueError("Se requieren tres meses previos")
        static = self.static[ids]
        factor = self.factor[ids]
        cols = [static[:, 0], static[:, 1], np.log1p(static[:, 2]), np.full(len(ids), min(month, 12))]
        for lag in (1, 2, 3):
            rows = self.panel[month-lag, ids] / factor[:, None]
            cols.extend(rows[:, var] for var in range(11))
        for window in (3, 6, 12):
            hist = self.panel[max(0, month-window):month, ids]
            risk = self.risk[max(0, month-window):month, ids]
            cols.extend([hist[:, :, 0].mean(axis=0)/factor, risk.mean(axis=0), risk.std(axis=0),
                         risk.max(axis=0), (risk > 0).mean(axis=0), (risk >= THRESHOLD).mean(axis=0)])
        cols.extend([
            self.risk[month-6, ids] if month >= 6 else np.full(len(ids), np.nan),
            self.risk[month-12, ids] if month >= 12 else np.full(len(ids), np.nan),
            self.risk[month-1, ids] - self.risk[month-2, ids],
            self.risk[month-3:month, ids].mean(axis=0) - self.risk[max(0, month-6):max(1, month-3), ids].mean(axis=0),
            (self.risk[month-1, ids]+.05)/(self.risk[max(0, month-6):month, ids].mean(axis=0)+.05),
            np.full(len(ids), np.sin(2*np.pi*(month % 12)/12)),
            np.full(len(ids), np.cos(2*np.pi*(month % 12)/12)),
        ])
        neighbors = self._neighbors()[ids]
        recent = self.risk[month-1]
        mean3 = self.risk[month-3:month].mean(axis=0)
        for k in (8, 32):
            cols.extend([recent[neighbors[:, :k]].mean(axis=1), mean3[neighbors[:, :k]].mean(axis=1),
                         (recent[neighbors[:, :k]] > 0).mean(axis=1)])
        for lag in (1, 2, 3):
            cols.extend([np.full(len(ids), float((self.risk[month-lag] > 0).mean())),
                         np.full(len(ids), float((self.risk[month-lag] >= THRESHOLD).mean()))])
        return np.nan_to_num(np.column_stack(cols).astype(np.float32), nan=0.)

    def training_tabular(self, entries):
        return np.concatenate([self.selected_features(month, ids) for month, ids in entries])

    def selection(self, month, window):
        first = 3 if window == "desde2018" else max(3, month-int(window[:-1]))
        months = list(range(first, month))
        if not months:
            raise ValueError("Sin meses de entrenamiento anteriores al objetivo")
        limit = min(18000, CAP//(3*len(months)))
        rng = np.random.default_rng(42)
        counts = np.zeros(3, dtype=np.int64)
        entries, targets, ratios = [], [], []
        h = hashlib.sha256()
        for prior in months:
            ym = self.target([prior])
            counts += np.bincount(ym, minlength=3)
            selected, expansion = [], []
            for cls in range(3):
                candidates = np.flatnonzero(ym == cls)
                keep = min(limit, len(candidates))
                if keep:
                    selected.append(rng.choice(candidates, keep, replace=False))
                    expansion.append(np.full(keep, len(candidates)/keep, dtype=np.float32))
            ids = np.concatenate(selected)
            order = np.argsort(ids)
            ids = ids[order]
            entries.append((prior, ids))
            targets.append(ym[ids])
            ratios.append(np.concatenate(expansion)[order])
            h.update(np.asarray(prior, dtype=np.int32).tobytes())
            h.update(ids.astype(np.int32).tobytes())
        y = np.concatenate(targets).astype(np.int64)
        weights = np.concatenate(ratios)*(counts.sum()/(3*counts[y]))**BALANCE
        weights = (weights/weights.mean()).astype(np.float32)
        training = {"first_target": self.periods[first], "last_target": self.periods[month-1],
                    "target_months": len(months), "train_units": len(months)*self.n,
                    "train_samples": len(y), "class_counts": counts.tolist(),
                    "sample_counts": np.bincount(y, minlength=3).tolist(),
                    "sample_sha256": h.hexdigest(), "per_class_month_cap": limit,
                    "class_balance_power": BALANCE, "history_starts": self.periods[max(0, first-12)]}
        assert max(m for m, _ in entries) < month
        return entries, y, weights, training


def aggregate(rows):
    return matrix_metrics(sum(np.asarray(r["metrics"]["matriz_confusion"], dtype=np.int64) for r in rows))


def row_metrics(data, month, probability):
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probabilidades inválidas")
    return evaluate(data.target([month]), probability)


# LSTM DEL PROTOCOLO: usa la arquitectura mensual y aprende solo con meses anteriores.
def fit_lstm(data, entries, y, weights, month, epochs, validation=False):
    seed()
    seq, context = data.training_lstm(entries)
    # La normalización se calcula con el entrenamiento para no utilizar información futura.
    scales = normalize_training(seq, context)
    loader = DataLoader(TensorDataset(torch.from_numpy(seq), torch.from_numpy(context),
                        torch.from_numpy(y), torch.from_numpy(weights)), batch_size=2048,
                        shuffle=True, num_workers=0)
    model = MonthlyLSTM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        for x, ctx, target, w in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = (F.cross_entropy(model(x, ctx), target, reduction="none")*w).sum()/w.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        record = {"epoch": epoch, "loss": float(np.mean(losses))}
        if validation:
            record["metrics"] = row_metrics(data, month, lstm_predict(model, data, month, scales))
        history.append(record)
        print(f"LSTM {data.periods[month] if month < len(data.periods) else 'D'} época {epoch}/{epochs}", flush=True)
    return model, scales, history


def register():
    path = OUT / "registro_inicial.json"
    if path.exists():
        registry = read(path)
        if registry["code_sha256"] != code_hashes() or registry["source_fingerprint"] != fingerprint():
            raise ValueError("Código o fuente cambió; use otra versión de protocolo")
        return registry
    registry = {"registered_at_utc": now(), "source_fingerprint": fingerprint(),
        "source_panel": str(PANEL), "source_csv": str(SOURCE), "source_max_month": PERIODS[-1],
        "code_sha256": code_hashes(), "unit": "tramo × mes", "horizon_months": 1,
        "coordinates": "solo originales; se excluyen las completadas/simuladas y los delitos sin coordenadas",
        "buffer_m": 300, "sigma_m": 100, "lighting": False, "turn_conditioned_prediction": False,
        "spatial_method_is_fixed_not_searched": True,
        "labels": {"low": "riesgo == 0", "medium": f"0 < riesgo < {THRESHOLD}",
                   "high": f"riesgo >= {THRESHOLD}", "risk": "severidad gaussiana / max(longitud_m/100, 1)",
                   "threshold_origin": "umbral de experimentos anteriores conservado; no se vuelve a estimar con C"},
        "phase_A": {"periods": ["2018-01", "2021-12"], "evaluated": False,
                    "role": "historial; etiquetas desde abril 2018 en candidatos expansivos; enero-marzo solo lags"},
        "phase_B": {"origins": list(B_PERIODS), "cadence": "trimestral: pronóstico de un mes por origen",
                    "families": list(FAMILIES), "training_windows": list(WINDOWS),
                    "lstm_epoch_candidates": list(range(1, EPOCHS+1)), "training_sample_cap": CAP,
                    "sampling": "estratificado por mes/clase, mismas filas para las tres familias, pesos de expansión",
                    "seed": 42, "balance_power": BALANCE,
                    "selection_score": "0.4 accuracy + 0.6 F1 medio/alto, agregado de orígenes B",
                    "high_risk_guard": "P y recall alto >= 90% del control 3m de cada familia; LSTM control 3m epoch4",
                    "refit": "desde cero en cada origen; solo etiquetas anteriores al objetivo",
                    "features": "tabular contexto 74 (NaN iniciales->0); LSTM secuencia12x14 + contexto5",
                    "xgboost": model_parameters(SPEC), "random_forest": RF_PARAMETERS,
                    "lstm": {"hidden": 32, "dropout": .1, "lr": .001, "weight_decay": .001,
                             "batch": 2048, "gradient_clip": 1., "epochs_max": EPOCHS}},
        "phase_C": {"periods": list(C_PERIODS), "label": "validación retrospectiva",
                    "selection_allowed": False, "cadence": "mensual, horizonte un mes",
                    "refit": "permitido mensualmente según ventana congelada; meses C pasados pueden entrenar el siguiente",
                    "independence": "semi-independiente: estos años y el umbral ya aparecieron en experimentos anteriores"},
        "phase_D": {"requested_start": "2026-09", "earliest_prospective_month": "2026-10",
                    "status": "pendiente: sin fuente posterior a agosto 2026",
                    "rule": "primer mes >= octubre 2026 con pronóstico registrado antes de su inicio y mes anterior completo",
                    "period_length_months": 1, "evaluate_once": True,
                    "primary_model": "ganador exclusivamente por B; C no cambia selección",
                    "online_retraining": "regla/ventana/épocas congeladas; ajuste solo con meses ya finalizados",
                    "lambda_astar": LAMBDA_ASTAR, "risk_score": "0.5 P(medio)+P(alto)",
                    "edge_cost": "longitud_m*(1+10*riesgo_score)",
                    "astar_status": "configuración de experimento; lambda no optimizado ni validado mediante clasificación",
                    "deployment_changed": False,
                    "registration_limitation": "se registra el 30 septiembre 2026; septiembre no se puede presentar como prospectivo"}}
    save_json(path, registry)
    for path in DEPENDENCIES:
        dest = DATA / "codigo_congelado" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    return registry


# FASE B: compara ventanas y épocas; aquí se eligen las configuraciones de los tres modelos.
def phase_b(data):
    for number, period in enumerate(B_PERIODS, 1):
        month = PERIODS.index(period)
        for window in WINDOWS:
            paths = {f: OUT / "B" / f"{f}_{window}_{period}.json" for f in FAMILIES}
            if all(p.exists() for p in paths.values()):
                continue
            print(f"FASE B {number}/{len(B_PERIODS)} {period} ventana={window}", flush=True)
            entries, y, weights, training = data.selection(month, window)
            tabular = [f for f in ("xgboost", "random_forest") if not paths[f].exists()]
            if tabular:
                x = data.training_tabular(entries)
                current = data.selected_features(month, np.arange(data.n))
                for family in tabular:
                    seed()
                    start = time.perf_counter()
                    if family == "xgboost":
                        model = fit_model(SPEC, x, y, weights)
                    else:
                        model = RandomForestClassifier(**RF_PARAMETERS).fit(x, y, sample_weight=weights)
                    metrics = row_metrics(data, month, predictions(model, current))
                    save_json(paths[family], {"phase": "B", "period": period, "family": family,
                        "window": window, "training": training, "metrics": metrics,
                        "seconds": time.perf_counter()-start})
                    print(f"B {family} {period} {window}: acc={metrics['accuracy']:.4f} alto={metrics['recall'][2]:.4f}", flush=True)
                    del model
                    gc.collect()
                del x, current
            if not paths["lstm"].exists():
                start = time.perf_counter()
                model, scales, history = fit_lstm(data, entries, y, weights, month, EPOCHS, True)
                save_json(paths["lstm"], {"phase": "B", "period": period, "family": "lstm",
                    "window": window, "training": training, "history": history,
                    "seconds": time.perf_counter()-start})
                del model, scales
            del entries, y, weights
            gc.collect()
    candidates, selected = {}, {}
    for family in FAMILIES:
        candidates[family] = []
        for window in WINDOWS:
            rows = [read(OUT / "B" / f"{family}_{window}_{p}.json") for p in B_PERIODS]
            for epoch in (range(1, EPOCHS+1) if family == "lstm" else (None,)):
                eval_rows = ([{"metrics": r["history"][epoch-1]["metrics"]} for r in rows]
                             if epoch is not None else rows)
                m = aggregate(eval_rows)
                candidates[family].append({"family": family, "window": window, "epochs": epoch,
                                            "metrics": m, "score": decision_score(m)})
        control = next(c for c in candidates[family] if c["window"] == "3m" and c["epochs"] in (None, EPOCHS))
        for c in candidates[family]:
            c["eligible_high_guard"] = (c["metrics"]["precision"][2] >= .9*control["metrics"]["precision"][2]
                                       and c["metrics"]["recall"][2] >= .9*control["metrics"]["recall"][2])
        selected[family] = max((c for c in candidates[family] if c["eligible_high_guard"]), key=lambda c: c["score"])
    winner = max(selected.values(), key=lambda c: c["score"])["family"]
    result = {"frozen_at_utc": now(), "selection_only_B": True, "primary_family_D": winner,
              "selected": selected, "candidates": candidates, "code_sha256": code_hashes(),
              "registry_sha256": digest(OUT / "registro_inicial.json"), "confirmation_consulted": False}
    path = OUT / "seleccion_congelada_antes_C.json"
    if path.exists():
        previous = read(path)
        if previous["selected"] != selected or previous["code_sha256"] != result["code_sha256"]:
            raise ValueError("No se permite cambiar la selección congelada")
        return previous
    save_json(path, result)
    print(f"SELECCIÓN CONGELADA B: {[(f, c['window'], c['epochs']) for f, c in selected.items()]}; ganador={winner}", flush=True)
    return result


# FASE C: reentrena cada mes con la configuración elegida en B y evalúa el mes siguiente.
def phase_c(data, selection):
    frozen_hash = digest(OUT / "seleccion_congelada_antes_C.json")
    for number, period in enumerate(C_PERIODS, 1):
        month = PERIODS.index(period)
        grouped = {}
        for family in FAMILIES:
            path = OUT / "C" / f"{family}_{period}.json"
            if not path.exists():
                grouped.setdefault(selection["selected"][family]["window"], []).append(family)
        for window, families in grouped.items():
            entries, y, weights, training = data.selection(month, window)
            x = current = None
            for family in families:
                print(f"FASE C {number}/{len(C_PERIODS)} {family} {period} ventana={window}", flush=True)
                seed()
                start = time.perf_counter()
                if family == "lstm":
                    # LSTM aprende con secuencias mensuales y las épocas previamente seleccionadas.
                    epochs = selection["selected"][family]["epochs"]
                    model, scales, _ = fit_lstm(data, entries, y, weights, month, epochs)
                    p = lstm_predict(model, data, month, scales)
                    del scales
                else:
                    if x is None:
                        x = data.training_tabular(entries)
                        current = data.selected_features(month, np.arange(data.n))
                    # XGBoost y Random Forest aprenden con las variables tabulares del mismo periodo.
                    model = (fit_model(SPEC, x, y, weights) if family == "xgboost" else
                             RandomForestClassifier(**RF_PARAMETERS).fit(x, y, sample_weight=weights))
                    p = predictions(model, current)
                metrics = row_metrics(data, month, p)
                save_json(OUT / "C" / f"{family}_{period}.json", {"phase": "C", "period": period,
                    "label": "validación retrospectiva", "family": family, "window": window,
                    "training": training, "metrics": metrics, "seconds": time.perf_counter()-start,
                    "selection_sha256": frozen_hash, "hyperparameters_changed_in_C": False})
                print(f"C {family} {period}: acc={metrics['accuracy']:.4f} alto={metrics['recall'][2]:.4f}", flush=True)
                del model, p
                gc.collect()
            del x, current, entries, y, weights
            gc.collect()
    results = {}
    for family in FAMILIES:
        rows = [read(OUT / "C" / f"{family}_{p}.json") for p in C_PERIODS]
        groups = {"2025": [r for r in rows if r["period"].startswith("2025")],
                  "2026_enero_agosto": [r for r in rows if r["period"].startswith("2026")],
                  "2026_enero_mayo_comparable": [r for r in rows if "2026-01" <= r["period"] <= "2026-05"]}
        results[family] = {"selected_in_B": selection["selected"][family], "monthly": rows,
                           "metrics": aggregate(rows), "subperiods": {g: aggregate(v) for g, v in groups.items()}}
    for period in C_PERIODS:
        rows = [next(r for r in results[f]["monthly"] if r["period"] == period) for f in FAMILIES]
        if len({tuple(r["metrics"]["support"]) for r in rows}) != 1:
            raise ValueError("Las etiquetas no coinciden entre familias")
    save_json(OUT / "resultados_finales_ABC.json", {"registry": read(OUT / "registro_inicial.json"),
        "selection": selection, "confirmation": results,
        "B_fits": len(B_PERIODS)*len(WINDOWS)*len(FAMILIES), "C_fits": len(C_PERIODS)*len(FAMILIES),
        "D_evaluated": False, "D_status": "pendiente de datos y pronóstico prospectivo registrado"})
    for group, name, subtitle in (
        (None, "matrices_validacion_retrospectiva_2025_2026.png", "Enero 2025–agosto 2026 · 20 orígenes mensuales"),
        ("2026_enero_agosto", "matrices_validacion_retrospectiva_2026.png", "Enero–agosto 2026 · 8 orígenes mensuales"),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(22, 8), facecolor="white")
        for ax, family, title in zip(axes, FAMILIES, ("XGBoost", "LSTM", "Random Forest")):
            m = results[family]["metrics"] if group is None else results[family]["subperiods"][group]
            render_matrix(ax, m, title + " · " + selection["selected"][family]["window"])
        fig.suptitle("Fase C · Validación retrospectiva · Buffer 300 m", fontsize=21, weight="bold")
        fig.text(.5, .91, subtitle + " · Coordenadas originales · Sin alumbrado", ha="center", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .82), w_pad=3)
        fig.savefig(OUT / name, dpi=135, bbox_inches="tight")
        plt.close(fig)
    for family, row in results.items():
        m = row["metrics"]
        print(f"FINAL C {family} accuracy={m['accuracy']:.6f} medio={m['recall'][1]:.6f} alto={m['recall'][2]:.6f}", flush=True)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    registry = register()
    signature = fingerprint()
    if read(PANEL_META)["fingerprint"] != signature or read(TEMPORAL_DATA / "secuencias_complete.json")["fingerprint"] != signature:
        raise ValueError("La fuente original y los paneles no coinciden")
    if (OUT / "resultados_finales_ABC.json").exists():
        print("ABC ya completado; se conservan sus resultados", flush=True)
        return
    data = RollingData()
    data.verify_causality()
    ids = np.arange(0, data.n, 233)
    for month in (3, 15, PERIODS.index("2022-03")):
        reference = np.nan_to_num(data.make(month, "contexto")[ids], nan=0.)
        if not np.allclose(reference, data.selected_features(month, ids), atol=1e-6):
            raise ValueError("Las variables seleccionadas difieren del cálculo completo")
    save_json(OUT / "controles_causalidad.json", {"at_utc": now(), "features_causal": True,
        "sampled_features_equal_full": True, "only_original_coordinates": True,
        "label_threshold_reused": THRESHOLD, "registration_at_utc": registry["registered_at_utc"]})
    selection = phase_b(data)
    phase_c(data, selection)


def future_data(args):
    if not all((args.panel, args.temporal, args.periods)):
        raise ValueError("D requiere --panel --temporal --periods con meses completos desde enero 2018")
    periods = read(args.periods)
    if not isinstance(periods, list) or periods != [str(p) for p in __import__("pandas").period_range("2018-01", periods[-1], freq="M")]:
        raise ValueError("La lista de periodos no es mensual y continua desde enero 2018")
    return RollingData(args.panel, args.temporal, periods)


def verify_frozen():
    selection = read(OUT / "seleccion_congelada_antes_C.json")
    if selection["code_sha256"] != code_hashes():
        raise ValueError("El pipeline ha cambiado desde el registro")
    return selection


def forecast_d(args):
    selection = verify_frozen()
    if (OUT / "D_pronostico_registrado.json").exists():
        raise ValueError("Ya se registró el mes D; no se reemplaza el pronóstico")
    import pandas as pd
    target = pd.Period(args.month, freq="M")
    deadline = datetime(target.year, target.month, 1, 5, tzinfo=timezone.utc)  # medianoche Lima
    if str(target) < "2026-10" or datetime.now(timezone.utc) >= deadline:
        raise ValueError("Un pronóstico D debe registrarse antes del inicio del mes, desde octubre 2026")
    data = future_data(args)
    if data.periods[-1] != str(target-1):
        raise ValueError("Se requiere el mes inmediatamente anterior completo; el objetivo aún no debe estar en el panel")
    month = len(data.periods)
    selected_files = {}
    for family in FAMILIES:
        chosen = selection["selected"][family]
        entries, y, weights, training = data.selection(month, chosen["window"])
        seed()
        if family == "lstm":
            model, scales, _ = fit_lstm(data, entries, y, weights, month, chosen["epochs"])
            p = lstm_predict(model, data, month, scales)
            torch.save({"state_dict": model.state_dict(), "scales": scales, "training": training}, DATA / "D_lstm.pt")
            del scales
        else:
            x = data.training_tabular(entries)
            model = (fit_model(SPEC, x, y, weights) if family == "xgboost" else
                     RandomForestClassifier(**RF_PARAMETERS).fit(x, y, sample_weight=weights))
            p = predictions(model, data.selected_features(month, np.arange(data.n)))
            joblib.dump({"model": model, "training": training}, DATA / f"D_{family}.joblib", compress=3)
            del x
        path = DATA / f"D_{family}_{target}.npz"
        np.savez_compressed(path, probability=p, risk_score=.5*p[:, 1]+p[:, 2])
        selected_files[family] = {"path": str(path), "sha256": digest(path), "training": training}
        del model, p, entries, y, weights
        gc.collect()
    if datetime.now(timezone.utc) >= deadline:
        raise ValueError("Los ajustes terminaron después del plazo; estos archivos no califican para D")
    save_json(OUT / "D_pronostico_registrado.json", {"registered_at_utc": now(), "month": str(target),
        "selection_sha256": digest(OUT / "seleccion_congelada_antes_C.json"), "models": selected_files,
        "primary_family": selection["primary_family_D"], "lambda_astar": LAMBDA_ASTAR,
        "panel_sha256": digest(args.panel), "temporal_sha256": digest(args.temporal),
        "periods": list(data.periods), "labels_seen": False})


def evaluate_d(args):
    selection = verify_frozen()
    if (OUT / "D_evaluacion_final_unica.json").exists():
        raise ValueError("D ya fue evaluada; no se vuelve a evaluar ni se ajusta el pipeline")
    forecast = read(OUT / "D_pronostico_registrado.json")
    data = future_data(args)
    month = data.periods.index(forecast["month"])
    if tuple(data.periods[:month]) != tuple(forecast["periods"]):
        raise ValueError("Cambió la cronología del panel de pronóstico")
    original_panel = np.load(args.forecast_panel, mmap_mode="r") if args.forecast_panel else None
    if original_panel is None or digest(args.forecast_panel) != forecast["panel_sha256"]:
        raise ValueError("Se requiere --forecast-panel intacto para comprobar el historial registrado")
    if not np.array_equal(original_panel, data.panel[:month]):
        raise ValueError("El historial fue revisado después del pronóstico; requiere otro análisis fuera de D")
    results = {}
    for family, item in forecast["models"].items():
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Cambió un pronóstico registrado")
        results[family] = row_metrics(data, month, np.load(item["path"])["probability"])
    save_json(OUT / "D_evaluacion_final_unica.json", {"at_utc": now(), "month": forecast["month"],
        "primary_family": selection["primary_family_D"], "results": results,
        "label": "test prospectivo final", "evaluated_once": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=("run", "forecast", "evaluate"), default="run")
    parser.add_argument("--month")
    parser.add_argument("--panel")
    parser.add_argument("--temporal")
    parser.add_argument("--periods")
    parser.add_argument("--forecast-panel")
    args = parser.parse_args()
    {"run": run, "forecast": lambda: forecast_d(args), "evaluate": lambda: evaluate_d(args)}[args.mode]()
