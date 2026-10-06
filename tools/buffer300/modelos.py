from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from xgboost import XGBClassifier

from comparar_factores_67 import _submuestrear_clase_baja, OUT as PREVIOUS
from .data import DATA, META, OUT, PERIODS, ROOT, THRESHOLD, save_json
from .features import Features

CONFIG_PATH = ROOT / "Backend/config_optimizacion_buffer_300.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def month_ids(periods):
    return [PERIODS.index(p) for p in periods]


def months_between(start, end):
    return list(range(PERIODS.index(start), PERIODS.index(end) + 1))


def matrix_metrics(matrix):
    a = np.asarray(matrix, dtype=np.int64)
    tp = np.diag(a)
    support = a.sum(axis=1)
    p = np.divide(tp, a.sum(axis=0), out=np.zeros(3), where=a.sum(axis=0) != 0)
    r = np.divide(tp, support, out=np.zeros(3), where=support != 0)
    f = np.divide(2*p*r, p+r, out=np.zeros(3), where=p+r != 0)
    return {"accuracy": float(tp.sum()/a.sum()), "precision": p.tolist(),
            "recall": r.tolist(), "f1": f.tolist(), "f1_macro": float(f.mean()),
            "f1_medio_alto": float(f[1:].mean()), "support": support.tolist(),
            "matriz_confusion": a.tolist()}


# EVALUACIÓN: compara la clase predicha con la observada y calcula la matriz de confusión.
def evaluate(y, probability, multiplier=None):
    if multiplier is None:
        multiplier = np.ones(3)
    pred = (probability * multiplier).argmax(axis=1)
    matrix = np.bincount(y.astype(np.int64)*3+pred, minlength=9).reshape(3, 3)
    return matrix_metrics(matrix)


def score(metrics):
    return CONFIG["selection"]["accuracy_weight"]*metrics["accuracy"] + CONFIG["selection"]["f1_middle_high_weight"]*metrics["f1_medio_alto"]


def select_bias(y, p):
    best = (-1, [1., 1., 1.])
    for medium in (.7, .85, 1., 1.15, 1.35, 1.6, 1.9, 2.3):
        for high in (.7, .85, 1., 1.15, 1.35, 1.6):
            bias = [1., medium, high]
            s = score(evaluate(y, p, bias))
            if s > best[0]:
                best = s, bias
    return best[1]


def fit_calibrator(y, probability):
    estimator = LogisticRegression(C=.3, max_iter=160, solver="lbfgs")
    estimator.fit(np.log(np.clip(probability, 1e-7, 1)), y)
    return estimator


def calibrate(estimator, probability):
    return estimator.predict_proba(np.log(np.clip(probability, 1e-7, 1))).astype(np.float32)


def adapt_prior(probability, features, months, reference, beta):
    if not beta:
        return probability
    result = probability.copy()
    for index, month in enumerate(months):
        previous = features.target([month-1])
        prior = np.bincount(previous, minlength=3)/len(previous)
        multiplier = ((prior+1e-6)/(np.asarray(reference)+1e-6)) ** beta
        part = result[index*features.n:(index+1)*features.n]
        part *= multiplier
        part /= part.sum(axis=1, keepdims=True)
    return result


# SALIDA: obtiene las probabilidades de Bajo, Medio y Alto para cada fila de entrada.
def predictions(model, x):
    if isinstance(model, list):
        pm = model[0].predict_proba(x)[:, 1]
        ph = model[1].predict_proba(x)[:, 1]
        # Proyección de dos probabilidades acumuladas al orden requerido.
        swapped = ph > pm
        mean = (pm + ph) / 2
        pm[swapped], ph[swapped] = mean[swapped], mean[swapped]
        return np.column_stack([1-pm, pm-ph, ph]).astype(np.float32)
    return model.predict_proba(x).astype(np.float32)


# XGBOOST: configura el número de árboles, su profundidad y la regularización.
# En el caso mensual de referencia usa 320 árboles, profundidad 5 y tasa 0,055.
def model_parameters(spec, binary=False):
    return dict(n_estimators=spec.get("trees", 320), max_depth=spec.get("depth", 5),
                learning_rate=spec.get("learning_rate", .055),
                min_child_weight=spec.get("child", 20), reg_lambda=spec.get("reg", 8),
                reg_alpha=spec.get("alpha", .1), subsample=.85, colsample_bytree=.85,
                max_bin=128, tree_method="hist", random_state=CONFIG["seed"], n_jobs=CONFIG["n_jobs"],
                objective="binary:logistic" if binary else "multi:softprob",
                eval_metric="logloss" if binary else "mlogloss")


# ENTRENAMIENTO XGBOOST: aprende la relación entre las variables históricas y el riesgo.
def fit_model(spec, x, y, weight):
    if spec.get("ordinal"):
        models = []
        for threshold in (1, 2):
            model = XGBClassifier(**model_parameters(spec, binary=True))
            model.fit(x, (y >= threshold).astype(np.int8), sample_weight=weight)
            models.append(model)
        return models
    params = model_parameters(spec)
    if spec.get("legacy"):
        params.update(n_estimators=260, max_depth=6, learning_rate=.06, min_child_weight=1,
                      reg_lambda=1, reg_alpha=0, max_bin=256)
    # El objetivo multi:softprob produce una probabilidad para cada una de las tres clases.
    model = XGBClassifier(**params)
    # Los pesos ajustan la contribución de cada muestra al aprendizaje.
    model.fit(x, y, sample_weight=weight)
    return model


class TrainingData:
    def __init__(self, feature_builder):
        self.feature_builder = feature_builder
        self.cached = None
        self.key = None

    def get(self, spec, end):
        start = spec.get("start", "2025-04")
        months = months_between(start, end)
        key = (start, end, spec["features"], spec.get("legacy", False), spec.get("full", False))
        f = self.feature_builder
        if key != self.key:
            self.cached = None
            gc.collect()
            y_all = f.target(months)
            actual_counts = np.bincount(y_all, minlength=3)
            rng = np.random.default_rng(CONFIG["seed"])
            selected, ratios, month_codes = [], [], []
            if spec.get("legacy"):
                indexes = _submuestrear_clase_baja(y_all, CONFIG["seed"])
                selected = [indexes]
                ratios = [np.ones(len(indexes), dtype=np.float32)]
                month_codes = [np.array(months)[indexes // f.n]]
            else:
                limit = min(CONFIG["sampling"]["per_class_month"],
                            CONFIG["sampling"]["maximum_training_rows"] // (3*len(months)))
                for j, m in enumerate(months):
                    ym = y_all[j*f.n:(j+1)*f.n]
                    for c in range(3):
                        indexes = np.flatnonzero(ym == c)
                        count = len(indexes)
                        keep = count if spec.get("full") else min(count, limit)
                        if not keep:
                            continue
                        indexes = rng.choice(indexes, size=keep, replace=False)
                        selected.append(indexes+j*f.n)
                        ratios.append(np.full(keep, count/keep, dtype=np.float32))
                        month_codes.append(np.full(keep, m, dtype=np.int16))
            selected = np.concatenate(selected)
            ordering = np.argsort(selected)
            selected = selected[ordering]
            ratios = np.concatenate(ratios)[ordering]
            month_codes = np.concatenate(month_codes)[ordering]
            y = y_all[selected]
            if spec.get("legacy"):
                actual_counts = np.bincount(y, minlength=3)
            x = None
            for j, m in enumerate(months):
                left, right = np.searchsorted(selected, [j*f.n, (j+1)*f.n])
                features = f.make(m, spec["features"])
                if x is None:
                    x = np.empty((len(selected), features.shape[1]), dtype=np.float32)
                x[left:right] = features[selected[left:right]-j*f.n]
                del features
            self.key = key
            self.cached = (x, y, ratios, month_codes, actual_counts, len(y_all))
        x, y, ratios, month_codes, counts, total = self.cached
        exponent = spec.get("balance", .5)
        weight = ratios * (counts.sum()/(3*counts[y])) ** exponent
        weight *= np.asarray(spec.get("class_boost", [1, 1, 1]), dtype=np.float32)[y]
        half_life = spec.get("half_life")
        if half_life:
            weight *= np.exp2(-(months[-1]-month_codes)/half_life)
        weight = (weight/weight.mean()).astype(np.float32)
        return x, y, weight, {"train_units": total, "train_samples": len(y),
                              "class_counts": counts.tolist(), "first_month": start,
                              "last_month": end, "n_features": x.shape[1]}


def base_candidates():
    specs = [{"name": "control_legacy", "features": "legacy", "legacy": True, "balance": 1},
             {"name": "legacy_natural", "features": "legacy", "balance": 0},
             {"name": "legacy_suave", "features": "legacy", "balance": .5}]
    for feature in ("temporal", "contexto", "sin_geografia"):
        for label, balance in (("natural", 0), ("suave", .5), ("balanceado", 1)):
            specs.append({"name": f"{feature}_{label}", "features": feature, "balance": balance})
    for depth in (3, 8):
        specs.append({"name": f"contexto_profundidad{depth}", "features": "contexto", "depth": depth,
                      "trees": 420, "balance": .5})
    specs += [{"name": "contexto_ordinal", "features": "contexto", "ordinal": True, "balance": .5},
              {"name": "contexto_completo", "features": "contexto", "balance": .5, "full": True},
              {"name": "contexto_reciente", "features": "contexto", "balance": .5, "half_life": 3},
              {"name": "contexto_regularizado", "features": "contexto", "balance": .5,
               "depth": 4, "trees": 600, "reg": 30, "child": 60, "learning_rate": .04}]
    return specs


def choose_result(rows, control):
    rules = CONFIG["selection"]
    eligible = [r for r in rows if r["metrics"]["precision"][2] >= control["precision"][2]*rules["minimum_high_precision_relative_to_control"]
                and r["metrics"]["recall"][2] >= control["recall"][2]*rules["minimum_high_recall_relative_to_control"]]
    dominating = [r for r in eligible if r["metrics"]["accuracy"] >= control["accuracy"]
                  and r["metrics"]["f1_medio_alto"] >= control["f1_medio_alto"]]
    return max(dominating or eligible or rows, key=lambda r: score(r["metrics"]))


def search(refinement=False):
    if not META.exists():
        raise ValueError("Ejecute primero prepare")
    if (OUT / "selection.json").exists() and not refinement:
        print("Selección ya completada; reutilizando resultados", flush=True)
        return
    if refinement and (OUT / "final_results.json").exists():
        raise ValueError("La prueba ya fue evaluada; una búsqueda adicional requiere otra versión del protocolo")
    f = Features()
    f.verify_causality()
    save_json(OUT / "configuration.json", CONFIG)
    save_json(OUT / "checks.json", {"baseline_panel_equal": True, "causal_features": True,
                                    "all_labels_fixed": True, "test_used_for_selection": False})
    training = TrainingData(f)
    cal_months = month_ids(CONFIG["development"]["calibration"])
    val_months = month_ids(CONFIG["development"]["validation"])
    ycal, yval = f.target(cal_months), f.target(val_months)
    cal_cache, val_cache = {}, {}
    rows, specs = [], base_candidates()
    control = None
    count = 0
    while count < len(specs):
        spec = specs[count]
        result_path = OUT / "trials" / f"{spec['name']}.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            start = time.perf_counter()
            print(f"ENSAYO {count+1}/{len(specs)} {spec['name']}", flush=True)
            x, y, w, details = training.get(spec, CONFIG["development"]["train_end"])
            model = fit_model(spec, x, y, w)
            kind = spec["features"]
            if kind not in cal_cache:
                cal_cache[kind] = f.matrix(cal_months, kind)
                val_cache[kind] = f.matrix(val_months, kind)
            pcal = predictions(model, cal_cache[kind])
            pval = predictions(model, val_cache[kind])
            estimator = fit_calibrator(ycal, pcal)
            calibrated_cal, calibrated_val = calibrate(estimator, pcal), calibrate(estimator, pval)
            alternatives = []
            for method in CONFIG["probability_methods"]:
                pc, pv = (calibrated_cal, calibrated_val) if method.startswith("calibrated") else (pcal, pval)
                bias = select_bias(ycal, pc) if method.endswith("bias") else [1., 1., 1.]
                metrics = evaluate(yval, pv, bias)
                alternatives.append({"name": spec["name"], "method": method, "bias": bias,
                                     "metrics": metrics, "calibration_metrics": evaluate(ycal, pc, bias),
                                     "spec": spec})
                print(f"  {method}: acc={metrics['accuracy']:.4f} F1MA={metrics['f1_medio_alto']:.4f} "
                      f"alto P/R={metrics['precision'][2]:.3f}/{metrics['recall'][2]:.3f}", flush=True)
            result = {"spec": spec, "training": details, "seconds": time.perf_counter()-start,
                      "alternatives": alternatives}
            save_json(result_path, result)
            (DATA / "validation").mkdir(exist_ok=True)
            np.savez_compressed(DATA / "validation" / f"{spec['name']}.npz", calibration=pcal, validation=pval)
            del model, pcal, pval, calibrated_cal, calibrated_val, estimator
            gc.collect()
        rows.extend(result["alternatives"])
        if spec["name"] == "control_legacy":
            control = result["alternatives"][0]["metrics"]
        count += 1
        if count == len(base_candidates()):
            best = choose_result(rows, control)
            chosen = best["spec"]
            for first, half in (("2024-01", None), ("2022-01", 12), ("2018-04", None), ("2018-04", 18)):
                candidate = {k: v for k, v in chosen.items() if k not in ("legacy", "full", "start", "half_life", "name")}
                candidate.update(name=f"historia_{first}_{half or 'todos'}", start=first, half_life=half)
                specs.append(candidate)
            save_json(OUT / "candidate_registry.json", specs)
        if count == len(base_candidates()) + 4 and refinement:
            # La ampliación usa solo la validación 2025; aún no se consulta 2026.
            specs.extend([
                {"name": "temporal_parametros_originales", "features": "temporal", "legacy": True, "balance": 1},
                {"name": "contexto_parametros_originales", "features": "contexto", "legacy": True, "balance": 1},
                {"name": "contexto_flexible", "features": "contexto", "depth": 8, "child": 1, "reg": 1, "alpha": 0, "trees": 450, "balance": 1},
                {"name": "temporal_flexible", "features": "temporal", "depth": 6, "child": 1, "reg": 1, "alpha": 0, "trees": 500, "balance": 1},
                {"name": "contexto_historia_natural", "features": "contexto", "start": "2018-04", "balance": 0, "depth": 6, "trees": 450},
                {"name": "contexto_historia_suave", "features": "contexto", "start": "2018-04", "balance": .5, "depth": 6, "trees": 450},
                {"name": "contexto_historia_balanceado", "features": "contexto", "start": "2018-04", "balance": 1, "depth": 6, "trees": 450},
                {"name": "contexto_historia_reciente", "features": "contexto", "start": "2018-04", "half_life": 18, "balance": .5, "depth": 6, "trees": 450},
                {"name": "temporal_historia_suave", "features": "temporal", "start": "2018-04", "balance": .5, "depth": 6, "trees": 450},
                {"name": "contexto_historia_ordinal", "features": "contexto", "start": "2018-04", "ordinal": True, "balance": .5, "depth": 6, "trees": 350}
                ,{"name": "contexto_balance_fuerte", "features": "contexto", "balance": 1.25}
                ,{"name": "contexto_enfasis_medio_alto", "features": "contexto", "balance": 1, "class_boost": [1, 1.3, 1.4]}
                ,{"name": "temporal_original_medio", "features": "temporal", "legacy": True, "balance": 1, "class_boost": [1, 1.25, 1]}
                ,{"name": "contexto_historia_enfasis", "features": "contexto", "start": "2018-04", "balance": 1, "depth": 6, "trees": 450, "class_boost": [1, 1.3, 1.4]}
                ,{"name": "aumentado_original", "features": "aumentado", "legacy": True, "balance": 1}
                ,{"name": "aumentado_medio", "features": "aumentado", "legacy": True, "balance": 1, "class_boost": [1, 1.25, 1]}
                ,{"name": "aumentado_contexto_original", "features": "aumentado_contexto", "legacy": True, "balance": 1}
                ,{"name": "aumentado_contexto_regular", "features": "aumentado_contexto", "balance": 1}
            ])
            save_json(OUT / "candidate_registry.json", specs)
    selected = choose_result(rows, control)
    save_json(OUT / "selection.json", {"selected": selected, "control_validation": control,
                                      "trials": len(specs), "decision_variants": len(rows),
                                      "selection_periods": CONFIG["development"], "all_results": rows,
                                      "test_was_not_used_for_selection": True,
                                      "refinement_completed": refinement})
    save_json(DATA / "feature_names.json", f.names)
    print(f"SELECCIONADO {selected['name']} / {selected['method']}: {selected['metrics']}", flush=True)


def finalize():
    if (OUT / "final_results.json").exists():
        print("Evaluación final existente; no se vuelve a usar la prueba para ajustar", flush=True)
        return
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    selected, f = selection["selected"], Features()
    spec, method = selected["spec"], selected["method"]
    training = TrainingData(f)
    print(f"AJUSTE FINAL {selected['name']} / {method}", flush=True)
    needs_calibration_holdout = method.startswith("calibrated") or method.endswith("bias")
    final_train_end = CONFIG["final"]["train_end"] if needs_calibration_holdout else "2025-12"
    x, y, w, details = training.get(spec, final_train_end)
    model = fit_model(spec, x, y, w)
    del x, y, w, training
    gc.collect()
    calibration_months = month_ids(CONFIG["final"]["calibration"])
    used_calibration = CONFIG["final"]["calibration"] if needs_calibration_holdout else []
    ycal = f.target(calibration_months)
    pcal = predictions(model, f.matrix(calibration_months, spec["features"]))
    estimator = fit_calibrator(ycal, pcal) if method.startswith("calibrated") else None
    if estimator is not None:
        pcal = calibrate(estimator, pcal)
    bias = select_bias(ycal, pcal) if method.endswith("bias") else [1., 1., 1.]
    prior_reference = (np.bincount(ycal, minlength=3)/len(ycal)).tolist()
    prior_beta = selected.get("prior_beta", 0)
    package = {"models": model, "calibrator": estimator, "decision_multiplier": bias,
               "features": spec["features"], "feature_names": f.names[spec["features"]],
               "spec": spec, "training": details, "calibration_periods": used_calibration,
               "radius_m": 300, "sigma_m": 100, "high_threshold": THRESHOLD,
               "prior_reference": prior_reference, "prior_beta": prior_beta,
               "source_panel": str(DATA / "panel_mensual_300m.npy"), "selection": selected,
               "coordinate_origin": "Fuente mixta: observadas y copiadas; escenario experimental"}
    joblib.dump(package, DATA / "modelo_seleccionado.joblib", compress=3)
    restored = joblib.load(DATA / "modelo_seleccionado.joblib")
    probe = f.make(calibration_months[0], spec["features"])[:128]
    if not np.allclose(predictions(restored["models"], probe), predictions(model, probe)):
        raise AssertionError("El modelo guardado no reproduce las probabilidades")
    del restored, probe
    output = {"selected": selected, "training": details, "calibration": used_calibration,
              "calibration_metrics": evaluate(ycal, pcal, bias) if used_calibration else None, "final_bias": bias,
              "prior_reference": prior_reference, "prior_beta": prior_beta,
              "model_path": str(DATA / "modelo_seleccionado.joblib"), "evaluations": {}}
    baseline = joblib.load(PREVIOUS / "modelo_b_base_nueva_buffer_300m_mes.joblib")["pipeline"]
    for evaluation in ("benchmark", "additional_evaluation"):
        months = month_ids(CONFIG["final"][evaluation])
        all_p, monthly, baseline_p = [], [], []
        for month in months:
            probability = predictions(model, f.make(month, spec["features"]))
            if estimator is not None:
                probability = calibrate(estimator, probability)
            probability = adapt_prior(probability, f, [month], prior_reference, prior_beta)
            if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1, atol=1e-5):
                raise AssertionError("Probabilidades inválidas")
            ym = f.target([month])
            metrics = evaluate(ym, probability, bias)
            monthly.append({"periodo": PERIODS[month], **metrics})
            all_p.append(probability)
            original_probability = baseline.predict_proba(f.make(month, "legacy")).astype(np.float32)
            baseline_p.append(original_probability)
            monthly[-1]["baseline_metrics"] = evaluate(ym, original_probability)
            print(f"PRUEBA {PERIODS[month]} acc={metrics['accuracy']:.4f} F1MA={metrics['f1_medio_alto']:.4f}", flush=True)
        p = np.concatenate(all_p)
        truth = f.target(months)
        metrics = evaluate(truth, p, bias)
        metrics["average_precision_alto"] = float(average_precision_score(truth == 2, p[:, 2]))
        metrics["log_loss"] = float(log_loss(truth, p))
        output["evaluations"][evaluation] = {"periods": CONFIG["final"][evaluation],
                                                "metrics": metrics, "monthly": monthly,
                                                "baseline_metrics": evaluate(truth, np.concatenate(baseline_p))}
        if evaluation == "benchmark":
            expected = json.loads((PREVIOUS / "resultado_b_base_nueva_buffer_300m_mes.json").read_text(encoding="utf-8"))["metrics"]["matriz_confusion"]
            actual = output["evaluations"][evaluation]["baseline_metrics"]["matriz_confusion"]
            if actual != expected:
                raise AssertionError("No se reprodujo exactamente la matriz original")
        np.savez_compressed(DATA / f"predicciones_{evaluation}.npz", truth=truth, probability=p,
                            predicted=(p*bias).argmax(axis=1).astype(np.int8))
    save_json(OUT / "final_results.json", output)
    print("EVALUACION FINAL COMPLETA", flush=True)


def adapt():
    """Compara correcciones de distribución conocidas antes del mes objetivo."""
    if (OUT / "final_results.json").exists():
        raise ValueError("La prueba ya fue evaluada")
    selection_path = OUT / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("prior_adaptation_completed"):
        return
    save_json(OUT / "selection_before_prior.json", selection)
    f = Features()
    cal_months = month_ids(CONFIG["development"]["calibration"])
    val_months = month_ids(CONFIG["development"]["validation"])
    ycal, yval = f.target(cal_months), f.target(val_months)
    reference = np.bincount(ycal, minlength=3)/len(ycal)
    rows = list(selection["all_results"])
    names = sorted({row["name"] for row in rows})
    for name in names:
        original = [row for row in selection["all_results"] if row["name"] == name]
        with np.load(DATA / "validation" / f"{name}.npz") as saved:
            pcal, pval = saved["calibration"], saved["validation"]
        estimator = fit_calibrator(ycal, pcal)
        calibrated_val = calibrate(estimator, pval)
        for source_row in original:
            base = calibrated_val if source_row["method"].startswith("calibrated") else pval
            for beta in (.5, 1.):
                p = adapt_prior(base, f, val_months, reference, beta)
                metrics = evaluate(yval, p, source_row["bias"])
                # Mantener el sufijo bias permite reutilizar el mismo procedimiento
                # de ajuste de decisión al recalibrar antes de la evaluación final.
                original_method = source_row["method"]
                prefix = "calibrated" if original_method.startswith("calibrated") else "raw"
                suffix = "_bias" if original_method.endswith("bias") else ""
                method = f"{prefix}_prior{beta}{suffix}"
                rows.append({**source_row, "method": method, "prior_beta": beta, "metrics": metrics})
        print(f"ADAPTACION {name}: revisada solo en noviembre–diciembre 2025", flush=True)
    selected = choose_result(rows, selection["control_validation"])
    selection.update(selected=selected, all_results=rows, decision_variants=len(rows),
                     prior_adaptation_completed=True,
                     prior_adaptation="Ajuste opcional con proporciones observadas en el mes anterior; nunca usa etiquetas del mes predicho")
    save_json(selection_path, selection)
    print(f"SELECCION DEFINITIVA {selected['name']} {selected['method']}: {selected['metrics']}", flush=True)


def run(stage):
    if stage == "search":
        search()
    elif stage == "refine":
        search(refinement=True)
    elif stage == "adapt":
        adapt()
    elif stage == "finalize":
        finalize()
    elif stage == "report":
        from .report import generate
        generate()
