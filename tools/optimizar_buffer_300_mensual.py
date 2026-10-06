"""Evaluación cronológica de reentrenamiento mensual con buffer 300 m fijo."""

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
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from buffer300.data import CACHE, DATA, OUT as V1_OUT, PERIODS, THRESHOLD, save_json
from buffer300.modelos import TrainingData, evaluate, fit_model, matrix_metrics, predictions, choose_result, score
from buffer300.features import Features
from buffer300.report import pct, render_matrix

OUT = ROOT / "outputs/evaluacion-modelos-sidpol/optimizacion_buffer_300_mensual_v2"
MODEL_DIR = DATA / "mensual_v2"
VALIDATION = [PERIODS.index(p) for p in ("2025-11", "2025-12")]
TEST = [PERIODS.index(p) for p in ("2026-01", "2026-02", "2026-03", "2026-04", "2026-05")]
ADDITIONAL = [PERIODS.index(p) for p in ("2026-06", "2026-07", "2026-08")]
CANDIDATES = [
    {"name": "legado_3m", "features": "legacy", "legacy": True, "balance": 1, "window": 3},
    {"name": "legado_6m", "features": "legacy", "legacy": True, "balance": 1, "window": 6},
    {"name": "temporal_3m", "features": "contexto", "balance": 1.25, "window": 3},
    {"name": "temporal_6m", "features": "contexto", "balance": 1.25, "window": 6},
]


def sample_spec(candidate, target_month):
    return {**candidate, "start": PERIODS[target_month-candidate["window"]]}


def train_for_month(candidate, month, features, training):
    spec = sample_spec(candidate, month)
    x, y, weight, details = training.get(spec, PERIODS[month-1])
    model = fit_model(spec, x, y, weight)
    return model, details


def collect_metrics(month_files):
    matrices = [np.array(item["metrics"]["matriz_confusion"]) for item in month_files]
    return matrix_metrics(sum(matrices))


def validate():
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if (OUT / "selection.json").exists():
        print("Selección mensual existente", flush=True)
        return
    features, training = Features(), None
    features.verify_causality()
    training = TrainingData(features)
    results = []
    for candidate in CANDIDATES:
        months = []
        for month in VALIDATION:
            model, details = train_for_month(candidate, month, features, training)
            y = features.target([month])
            p = predictions(model, features.make(month, candidate["features"]))
            months.append({"periodo": PERIODS[month], "metrics": evaluate(y, p), "training": details})
            print(f"VALIDACION {candidate['name']} {PERIODS[month]}: "
                  f"acc={months[-1]['metrics']['accuracy']:.4f}, "
                  f"F1MA={months[-1]['metrics']['f1_medio_alto']:.4f}", flush=True)
        metrics = collect_metrics(months)
        results.append({"name": candidate["name"], "spec": candidate, "metrics": metrics,
                        "months": months, "score": score(metrics)})
        save_json(OUT / "validacion_parcial.json", results)
    control = next(row["metrics"] for row in results if row["name"] == "legado_6m")
    selected = choose_result(results, control)
    save_json(OUT / "selection.json", {"selected": selected, "candidates": results,
                                      "control": control, "validation": [PERIODS[m] for m in VALIDATION],
                                      "selection_uses_2026": False,
                                      "protocol": "Reentrenar cada mes con los 3 o 6 meses anteriores; etiquetas del mes objetivo excluidas; buffer 300 m fijo"})
    print(f"SELECCIONADO {selected['name']}: acc={selected['metrics']['accuracy']:.4f}, "
          f"F1MA={selected['metrics']['f1_medio_alto']:.4f}", flush=True)


def evaluate_months():
    if (OUT / "final_results.json").exists():
        print("Evaluación cronológica ya completada", flush=True)
        return
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    candidate = selection["selected"]["spec"]
    features, training = Features(), None
    features.verify_causality()
    training = TrainingData(features)
    records = {}
    for month in TEST + ADDITIONAL:
        period = PERIODS[month]
        saved = MODEL_DIR / f"prediccion_{period}.npz"
        record = OUT / f"resultado_{period}.json"
        if saved.exists() and record.exists():
            records[period] = json.loads(record.read_text(encoding="utf-8"))
            continue
        model, details = train_for_month(candidate, month, features, training)
        truth = features.target([month])
        probability = predictions(model, features.make(month, candidate["features"]))
        if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1, atol=1e-5):
            raise AssertionError(f"Probabilidades inválidas en {period}")
        metrics = evaluate(truth, probability)
        records[period] = {"periodo": period, "training": details, "metrics": metrics}
        save_json(record, records[period])
        np.savez_compressed(saved, truth=truth, probability=probability)
        print(f"PRUEBA MENSUAL {period}: acc={metrics['accuracy']:.4f}, "
              f"F1MA={metrics['f1_medio_alto']:.4f}, "
              f"P/R alto={metrics['precision'][2]:.3f}/{metrics['recall'][2]:.3f}", flush=True)
    previous = json.loads((V1_OUT / "final_results.json").read_text(encoding="utf-8"))
    evaluations = {}
    for name, months in (("benchmark", TEST), ("additional_evaluation", ADDITIONAL)):
        period_names = [PERIODS[m] for m in months]
        monthly = [records[p] for p in period_names]
        metrics = collect_metrics(monthly)
        truth_chunks, proba_chunks = [], []
        for period in period_names:
            with np.load(MODEL_DIR / f"prediccion_{period}.npz") as data:
                truth_chunks.append(data["truth"])
                proba_chunks.append(data["probability"])
        truth, probability = np.concatenate(truth_chunks), np.concatenate(proba_chunks)
        metrics["average_precision_alto"] = float(average_precision_score(truth == 2, probability[:, 2]))
        original = previous["evaluations"][name]["baseline_metrics"]
        if metrics["support"] != original["support"]:
            raise AssertionError("La prueba mensual cambió las etiquetas del ensayo anterior")
        evaluations[name] = {"periods": period_names, "metrics": metrics,
                             "monthly": monthly, "original_frozen": original}
    save_json(OUT / "final_results.json", {"selected": selection["selected"],
              "validation": selection["validation"], "evaluations": evaluations,
              "predictive_information": "Para cada objetivo solo se entrenó con meses anteriores. Los objetivos previos de 2026 son conocidos al predecir un mes posterior.",
              "test_used_for_selection": False})
    print("EVALUACION MENSUAL COMPLETA", flush=True)


def train_current():
    """Guarda el modelo que usaría agosto completo para pronosticar septiembre."""
    destination = MODEL_DIR / "modelo_para_2026_09.joblib"
    if destination.exists():
        print(destination, flush=True)
        return
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    candidate = selection["selected"]["spec"]
    features, training = Features(), None
    training = TrainingData(features)
    month = len(PERIODS)
    model, details = train_for_month(candidate, month, features, training)
    data = features.make(month, candidate["features"])
    probability = predictions(model, data)
    if not np.isfinite(probability).all():
        raise AssertionError("El pronóstico de septiembre contiene valores no válidos")
    package = {"model": model, "candidate": candidate, "training": details,
               "features": candidate["features"], "feature_names": features.names[candidate["features"]],
               "target_period": "2026-09", "panel": str(CACHE),
               "fixed_radius_m": 300, "fixed_high_threshold": THRESHOLD,
               "source_note": "Escenario de coordenadas observadas y completadas"}
    joblib.dump(package, destination, compress=3)
    loaded = joblib.load(destination)
    if not np.allclose(predictions(loaded["model"], data[:128]), probability[:128]):
        raise AssertionError("El modelo guardado no reproduce sus probabilidades")
    print(destination, flush=True)


def report():
    result = json.loads((OUT / "final_results.json").read_text(encoding="utf-8"))
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    ev = result["evaluations"]["benchmark"]
    original, current = ev["original_frozen"], ev["metrics"]
    fields = [("Accuracy", "accuracy", None), ("Precisión medio", "precision", 1),
              ("Detección medio", "recall", 1), ("F1 medio", "f1", 1),
              ("Precisión alto", "precision", 2), ("Detección alto", "recall", 2),
              ("F1 alto", "f1", 2), ("F1 medio+alto", "f1_medio_alto", None)]
    lines = ["# Evaluación del reentrenamiento mensual, buffer 300 m", "",
             "| Métrica | Modelo anterior fijo | Reentrenamiento mensual | Cambio (puntos) |",
             "|---|---:|---:|---:|"]
    for title, key, index in fields:
        a = original[key] if index is None else original[key][index]
        b = current[key] if index is None else current[key][index]
        lines.append(f"| {title} | {pct(a)} | {pct(b)} | {(b-a)*100:+.2f} |")
    lines += ["", "## Selección en noviembre–diciembre de 2025", "",
              "| Configuración | Accuracy | F1 medio+alto | Precisión alto | Detección alto |",
              "|---|---:|---:|---:|---:|"]
    for row in selection["candidates"]:
        m = row["metrics"]
        lines.append(f"| {row['name']} | {pct(m['accuracy'])} | {pct(m['f1_medio_alto'])} | {pct(m['precision'][2])} | {pct(m['recall'][2])} |")
    lines += ["", "## Evaluación cronológica 2026", "",
              "| Periodo | Accuracy anterior | Accuracy mensual | Detección medio mensual | Precisión alto mensual | Detección alto mensual |",
              "|---|---:|---:|---:|---:|---:|"]
    old_months = {m["periodo"]: m for m in json.loads((V1_OUT / "final_results.json").read_text(encoding="utf-8"))["evaluations"]["benchmark"]["monthly"]}
    for m in ev["monthly"]:
        s = m["metrics"]
        lines.append(f"| {m['periodo']} | {pct(old_months[m['periodo']]['baseline_metrics']['accuracy'])} | {pct(s['accuracy'])} | {pct(s['recall'][1])} | {pct(s['precision'][2])} | {pct(s['recall'][2])} |")
    extra = result["evaluations"]["additional_evaluation"]
    lines += ["", "## Junio–agosto de 2026", "",
              f"Accuracy anterior {pct(extra['original_frozen']['accuracy'])}; mensual {pct(extra['metrics']['accuracy'])}. "
              f"F1 medio+alto anterior {pct(extra['original_frozen']['f1_medio_alto'])}; mensual {pct(extra['metrics']['f1_medio_alto'])}.", "",
              "## Interpretación", "",
              f"Se seleccionó `{selection['selected']['name']}` con los mismos tramos, radio de 300 m, pesos de gravedad delictiva y umbral del ensayo anterior.",
              "Cada mes se entrenó desde cero con la ventana elegida inmediatamente anterior. Los hechos de meses de prueba previos pasan a ser historia conocida para los meses siguientes; ninguna etiqueta del mes objetivo entra en su entrenamiento.",
              "La validación y la prueba son retrospectivas: los meses de 2026 ya habían sido consultados en otros experimentos. La fuente incluye coordenadas copiadas y el objetivo representa el entorno de 300 m, con varios tramos por delito.",
              "Las variantes fueron seleccionadas con 2025. Se conservaron también los resultados que no mejoraron, sin recalibrar decisiones con las pruebas de 2026.",
              "", "Archivos: `selection.json`, `final_results.json`, `matrices_antes_despues.png`, `modelo_para_2026_09.joblib` en la carpeta de modelos.", ""]
    (OUT / "informe.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8), facecolor="white")
    render_matrix(axes[0], original, "Modelo anterior fijo")
    render_matrix(axes[1], current, "Reentrenamiento mensual")
    fig.suptitle("Buffer 300 m · mismas etiquetas y tramos", y=.98, fontsize=19, weight="bold")
    fig.text(.5, .91, "Prueba enero–mayo de 2026 · porcentajes por clase real", ha="center", fontsize=12)
    fig.subplots_adjust(top=.78, bottom=.14, left=.06, right=.98, wspace=.25)
    fig.savefig(OUT / "matrices_antes_despues.png", dpi=180, facecolor="white")
    plt.close(fig)
    np.savetxt(OUT / "matriz_confusion_mensual.csv", np.array(current["matriz_confusion"]),
               fmt="%d", delimiter=",", header="pred_bajo,pred_medio,pred_alto", comments="")
    print(OUT / "informe.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validate", "evaluate", "current", "report"))
    stage = parser.parse_args().stage
    {"validate": validate, "evaluate": evaluate_months, "current": train_current, "report": report}[stage]()
