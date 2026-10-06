"""Ajusta la decisión mensual de 300 m manteniendo detección de riesgo alto.

Reutiliza probabilidades guardadas. Enero–mayo de 2026 es desarrollo para
escoger multiplicadores; junio–agosto de 2026 es evaluación cronológica
posterior. Los meses de 2026 se han examinado antes: ensayo retrospectivo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from buffer300.decision import predict_frozen_high
BASE = ROOT / "Backend/data/experimentos_sidpol/buffer_300_v1/mensual_v2"
LIGHT = ROOT / "outputs/evaluacion-modelos-sidpol/alumbrado_buffer_300_v1"
OUT = ROOT / "outputs/evaluacion-modelos-sidpol/ajuste_decision_300m_v3"
DEV = ("2026-01", "2026-02", "2026-03", "2026-04", "2026-05")
LATER = ("2026-06", "2026-07", "2026-08")


def load_data(arm, months):
    truth, probs = [], []
    for period in months:
        path = BASE / f"prediccion_{period}.npz" if arm == "sin_alumbrado" else LIGHT / f"probabilidades_{period}.npz"
        with np.load(path) as data:
            truth.append(data["truth"])
            probs.append(data["probability"] if arm == "sin_alumbrado" else data["proba"])
    return np.concatenate(truth), np.concatenate(probs)


def metrics_from_matrix(matrix):
    a = np.asarray(matrix, dtype=np.int64)
    support = a.sum(axis=1)
    predicted = a.sum(axis=0)
    diagonal = a.diagonal()
    p = np.divide(diagonal, predicted, out=np.zeros(3), where=predicted > 0)
    r = np.divide(diagonal, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2*diagonal, support+predicted, out=np.zeros(3), where=(support+predicted) > 0)
    return {"accuracy": float(diagonal.sum()/a.sum()), "precision": p.tolist(),
            "recall": r.tolist(), "f1": f1.tolist(), "f1_medio_alto": float(f1[1:].mean()),
            "support": support.tolist(), "matriz_confusion": a.tolist()}


def measure(truth, probability, med=1., high=1.):
    pred = np.argmax(probability*np.array([1., med, high], dtype=np.float32), axis=1)
    matrix = np.bincount(truth.astype(np.int64)*3+pred, minlength=9).reshape(3, 3)
    return metrics_from_matrix(matrix)


def measure_frozen_high(truth, base_probability, light_probability, blend, medium_multiplier):
    """Retiene cada predicción original de alto; decide bajo/medio en las demás."""
    pred = predict_frozen_high(base_probability, light_probability, blend, medium_multiplier)
    matrix = np.bincount(truth.astype(np.int64)*3+pred, minlength=9).reshape(3, 3)
    return metrics_from_matrix(matrix)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def search_global():
    """Registra la búsqueda previa de multiplicadores para las tres clases."""
    truth, base = load_data("sin_alumbrado", DEV)
    light_truth, light = load_data("con_alumbrado", DEV)
    if not np.array_equal(truth, light_truth):
        raise AssertionError("Cambió la verdad entre modelos")
    reference = measure(truth, base)
    guard = {"high_precision_min": reference["precision"][2],
             "high_recall_min": reference["recall"][2],
             "medium_f1_min": reference["f1"][1]-.005}
    rows = []
    for arm, probability in (("sin_alumbrado", base), ("con_alumbrado", light)):
        for medium in np.round(np.arange(.75, 1.301, .05), 2):
            for high in np.round(np.arange(.80, 1.351, .05), 2):
                result = measure(truth, probability, medium, high)
                eligible = (result["precision"][2] >= guard["high_precision_min"] and
                            result["recall"][2] >= guard["high_recall_min"] and
                            result["f1"][1] >= guard["medium_f1_min"])
                rows.append({"arm": arm, "medium_multiplier": float(medium),
                             "high_multiplier": float(high), "eligible": bool(eligible),
                             "metrics": result})
    save_json(OUT / "busqueda_umbrales_globales.json",
              {"development": DEV, "reference": reference, "guard": guard,
               "candidate_count": len(rows), "eligible_count": sum(row["eligible"] for row in rows),
               "candidates": rows})
    print(f"UMBRALES GLOBALES {len(rows)} candidatos; "
          f"{sum(row['eligible'] for row in rows)} elegibles", flush=True)


def search():
    OUT.mkdir(parents=True, exist_ok=True)
    base_truth, base_prob = load_data("sin_alumbrado", DEV)
    baseline = measure(base_truth, base_prob, 1., 1.)
    light_truth, light_prob = load_data("con_alumbrado", DEV)
    if not np.array_equal(base_truth, light_truth):
        raise AssertionError("Cambió la verdad entre los dos modelos")
    del light_truth
    # Las predicciones de alto se congelan fila por fila: precisión y recall
    # permanecen idénticos por construcción, también en la evaluación posterior.
    guard = {"high_precision_min": baseline["precision"][2],
             "high_recall_min": baseline["recall"][2],
             "medium_f1_min": baseline["f1"][1]}
    media = np.round(np.arange(.70, 1.401, .025), 3)
    blends = (0., .25, .5, .75, 1.)
    rows = []
    for blend in blends:
        for med in media:
            m = measure_frozen_high(base_truth, base_prob, light_prob, blend, float(med))
            allowed = m["f1"][1] >= guard["medium_f1_min"]
            rows.append({"blend_lighting": blend, "medium_multiplier": float(med),
                         "high_predictions_frozen": True, "metrics": m, "eligible": bool(allowed)})
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise AssertionError("La decisión original debía satisfacer sus propios límites")
    selected = max(eligible, key=lambda row: (row["metrics"]["accuracy"], row["metrics"]["f1_medio_alto"]))
    save_json(OUT / "selection.json", {"development": DEV, "later_evaluation": LATER,
              "baseline": baseline, "guard": guard, "candidate_count": len(rows),
              "eligible_count": len(eligible), "selected": selected,
              "all_candidates": rows,
              "test_used_in_selection": False,
              "note": "Enero–mayo 2026 se usa aquí como desarrollo, aunque fue prueba en ensayos anteriores."})
    print(f"SELECCIONADO mezcla={selected['blend_lighting']} M={selected['medium_multiplier']} alto congelado "
          f"acc={selected['metrics']['accuracy']:.4f} P/R alto={selected['metrics']['precision'][2]:.4f}/"
          f"{selected['metrics']['recall'][2]:.4f}; elegibles {len(eligible)}/{len(rows)}", flush=True)


def evaluate_later():
    selected = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))["selected"]
    baseline_y, baseline_p = load_data("sin_alumbrado", LATER)
    ref = measure(baseline_y, baseline_p, 1., 1.)
    light_y, light_p = load_data("con_alumbrado", LATER)
    if not np.array_equal(baseline_y, light_y):
        raise AssertionError("Cambió la verdad de la evaluación posterior")
    light = measure(light_y, light_p, 1., 1.)
    chosen = measure_frozen_high(baseline_y, baseline_p, light_p,
                                 selected["blend_lighting"], selected["medium_multiplier"])
    save_json(OUT / "final_results.json", {"selection": selected,
              "later_periods": LATER, "later_metrics": {"baseline": ref,
              "light_default": light, "selected": chosen},
              "improves_accuracy_and_keeps_high": bool(chosen["accuracy"] > ref["accuracy"] and
                                                      chosen["precision"][2] >= ref["precision"][2] and
                                                      chosen["recall"][2] >= ref["recall"][2]),
              "retrospective": True})
    print(f"JUNIO–AGOSTO: acc {ref['accuracy']:.4f}→{chosen['accuracy']:.4f}; "
          f"P/R alto {ref['precision'][2]:.4f}/{ref['recall'][2]:.4f}→"
          f"{chosen['precision'][2]:.4f}/{chosen['recall'][2]:.4f}", flush=True)


def train_current():
    """Guarda los dos XGBoost y la decisión seleccionada para septiembre de 2026."""
    from buffer300.data import PERIODS
    from buffer300.modelos import TrainingData, fit_model, predictions
    from probar_alumbrado_buffer_300 import LightingFeatures, NAMES, LIGHTS

    if not (OUT / "final_results.json").exists():
        raise ValueError("Complete antes search y evaluate")
    if not LIGHTS.exists():
        raise ValueError("Faltan las variables de alumbrado estático")
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))["selected"]
    source = BASE / "modelo_para_2026_09.joblib"
    base = joblib.load(source)
    if base["target_period"] != "2026-09" or base["candidate"]["name"] != "temporal_3m":
        raise ValueError("Cambió el modelo mensual base")
    features = LightingFeatures()
    month = len(PERIODS)
    spec = {**base["candidate"], "features": "contexto_luz", "start": PERIODS[month-3]}
    training = TrainingData(features)
    x, y, weight, details = training.get(spec, PERIODS[-1])
    light_model = fit_model(spec, x, y, weight)
    base_x = features.make(month, "contexto")
    light_x = features.make(month, "contexto_luz")
    base_p = predictions(base["model"], base_x)
    light_p = predictions(light_model, light_x)
    risk = predict_frozen_high(base_p, light_p,
                               selection["blend_lighting"], selection["medium_multiplier"])
    if not np.isfinite(base_p).all() or not np.isfinite(light_p).all():
        raise AssertionError("Probabilidades no válidas para septiembre")
    destination = BASE.parent / "mensual_v3" / "modelo_para_2026_09.joblib"
    destination.parent.mkdir(parents=True, exist_ok=True)
    package = {"base_model": base["model"], "light_model": light_model,
               "base_feature_names": base["feature_names"],
               "light_feature_names": features.names["contexto_luz"],
               "decision": {"kind": "freeze_base_high_then_blend_low_medium",
                            "blend_lighting": selection["blend_lighting"],
                            "medium_multiplier": selection["medium_multiplier"]},
               "training_base": base["training"], "training_light": details,
               "target_period": "2026-09", "fixed_radius_m": 300,
               "static_lighting": True, "light_variables": NAMES,
               "retrospective_selection": True,
               "source_note": base["source_note"]}
    joblib.dump(package, destination, compress=3)
    loaded = joblib.load(destination)
    reproduced = predict_frozen_high(predictions(loaded["base_model"], base_x[:128]),
                                     predictions(loaded["light_model"], light_x[:128]),
                                     loaded["decision"]["blend_lighting"],
                                     loaded["decision"]["medium_multiplier"])
    if not np.array_equal(reproduced, risk[:128]):
        raise AssertionError("El artefacto guardado no reproduce las clases")
    print(f"ARTEFACTO {destination} · predicciones septiembre: "
          f"{np.bincount(risk, minlength=3).tolist()}", flush=True)


def report():
    result = json.loads((OUT / "final_results.json").read_text(encoding="utf-8"))
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    light_original = json.loads((LIGHT / "final_results.json").read_text(encoding="utf-8"))["groups"]["benchmark"]["alumbrado"]
    base = result["later_metrics"]["baseline"]
    other = result["later_metrics"]["selected"]
    fields = [("Accuracy", "accuracy", None), ("Precisión medio", "precision", 1),
              ("Detección medio", "recall", 1), ("F1 medio", "f1", 1),
              ("Precisión alto", "precision", 2), ("Detección alto", "recall", 2),
              ("F1 alto", "f1", 2), ("F1 medio+alto", "f1_medio_alto", None)]
    lines = ["# Ajuste de decisión para riesgo mensual, buffer 300 m", "",
             "El modelo y la fuente no cambian. Se comparan probabilidades guardadas del XGBoost mensual con y sin "
             "alumbrado fijo. Se conservan exactamente las predicciones de alto del modelo base. "
             "Solo se ajusta la decisión bajo/medio con un multiplicador y una mezcla opcional de probabilidades.", "",
             "La elección usa enero–mayo de 2026 y exige que el F1 medio no baje frente al "
             "modelo mensual sin alumbrado. Precisión y detección de alto son idénticas por construcción. "
             "Junio–agosto de 2026 se consulta después de fijar la elección.", "",
             f"Selección: mezcla con alumbrado {result['selection']['blend_lighting']:.2f}, multiplicador medio "
             f"{result['selection']['medium_multiplier']:.3f}, alto congelado. "
             f"Candidatos: {selection['candidate_count']}; elegibles: {selection['eligible_count']}.", "",
             "## Enero–mayo de 2026: periodo de ajuste", "",
             "| Métrica | Sin alumbrado | Con alumbrado de la imagen | Ajuste elegido |",
             "|---|---:|---:|---:|"]
    for title, key, index in fields:
        a = selection["baseline"][key] if index is None else selection["baseline"][key][index]
        b = light_original[key] if index is None else light_original[key][index]
        c = selection["selected"]["metrics"][key] if index is None else selection["selected"]["metrics"][key][index]
        lines.append(f"| {title} | {a*100:.2f} % | {b*100:.2f} % | {c*100:.2f} % |")
    lines += ["", "Estas cifras intervinieron en la elección y no son una prueba independiente.", "",
             "## Evaluación junio–agosto de 2026", "",
             "| Métrica | Sin alumbrado | Con alumbrado de la imagen | Ajuste elegido |",
             "|---|---:|---:|---:|"]
    for title, key, index in fields:
        a = base[key] if index is None else base[key][index]
        b = result["later_metrics"]["light_default"][key] if index is None else result["later_metrics"]["light_default"][key][index]
        c = other[key] if index is None else other[key][index]
        lines.append(f"| {title} | {a*100:.2f} % | {b*100:.2f} % | {c*100:.2f} % |")
    lines += ["", "Resultado de la condición principal: " + ("sí" if result["improves_accuracy_and_keeps_high"] else "no") + ".", "",
              "La evaluación es retrospectiva: estos meses ya fueron inspeccionados en otros ensayos. "
              "No se cambió el umbral espacial de 300 m ni el objetivo `tramo × mes`. Los porcentajes de enero–mayo "
              "sirvieron para elegir la decisión y no se presentan como una nueva prueba independiente.", ""]
    (OUT / "informe.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8), facecolor="white")
    from buffer300.report import render_matrix
    render_matrix(axes[0], base, "Referencia")
    render_matrix(axes[1], other, "Decisión ajustada")
    fig.suptitle("Buffer 300 m · junio–agosto de 2026", y=.98, fontsize=18, weight="bold")
    fig.text(.5, .91, "Evaluación posterior a la elección · porcentajes por clase real", ha="center", fontsize=12)
    fig.subplots_adjust(top=.78, bottom=.14, left=.06, right=.98, wspace=.25)
    fig.savefig(OUT / "matrices_ajuste_decision.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(OUT / "informe.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("global", "search", "evaluate", "current", "report"))
    {"global": search_global, "search": search, "evaluate": evaluate_later, "current": train_current,
     "report": report}[parser.parse_args().stage]()
