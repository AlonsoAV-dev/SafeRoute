"""Compara usar 12, 24 y toda la historia en el XGBoost mensual de 300 m.

Mismas etiquetas, variables, pesos, parámetros y meses de validación que
temporal_3m. Solo cambia el tramo temporal del entrenamiento y, para una
variante, el decaimiento temporal. Guarda resultados mes a mes para reanudar.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from buffer300.data import PERIODS, save_json
from buffer300.modelos import TrainingData, evaluate, fit_model, matrix_metrics, predictions
from buffer300.features import Features

OUT = ROOT / "outputs/evaluacion-modelos-sidpol/comparacion_historia_mensual_300_v1"
REFERENCE = ROOT / "outputs/evaluacion-modelos-sidpol/optimizacion_buffer_300_mensual_v2/selection.json"
VALIDATION = ("2025-11", "2025-12")
EVALUATION = ("2026-01", "2026-02", "2026-03", "2026-04", "2026-05",
              "2026-06", "2026-07", "2026-08")
CANDIDATES = (
    {"name": "historia_12m", "window": 12},
    {"name": "historia_24m", "window": 24},
    {"name": "historia_2018_completa", "start": "2018-04"},
    {"name": "historia_2018_decaimiento_24m", "start": "2018-04", "half_life": 24},
)


def specification(candidate, period):
    target = PERIODS.index(period)
    start = candidate.get("start") or PERIODS[target-candidate["window"]]
    return {"name": candidate["name"], "features": "contexto", "balance": 1.25,
            "start": start, **({"half_life": candidate["half_life"]} if "half_life" in candidate else {})}


def run_periods(periods):
    OUT.mkdir(parents=True, exist_ok=True)
    features = Features()
    features.verify_causality()
    training = TrainingData(features)
    for period in periods:
        month = PERIODS.index(period)
        truth = features.target([month])
        for candidate in CANDIDATES:
            destination = OUT / "resultados" / f"{candidate['name']}_{period}.json"
            if destination.exists():
                print(f"EXISTENTE {candidate['name']} {period}", flush=True)
                continue
            spec = specification(candidate, period)
            print(f"ENTRENANDO {candidate['name']} {spec['start']}–{PERIODS[month-1]} → {period}", flush=True)
            x, y, weight, details = training.get(spec, PERIODS[month-1])
            model = fit_model(spec, x, y, weight)
            probability = predictions(model, features.make(month, "contexto"))
            if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1, atol=1e-5):
                raise AssertionError(f"Probabilidades inválidas: {candidate['name']} {period}")
            result = {"candidate": candidate, "period": period, "training": details,
                      "metrics": evaluate(truth, probability)}
            save_json(destination, result)
            print(f"TERMINADO {candidate['name']} {period}: acc={result['metrics']['accuracy']:.4f} "
                  f"F1MA={result['metrics']['f1_medio_alto']:.4f} "
                  f"P/R alto={result['metrics']['precision'][2]:.4f}/{result['metrics']['recall'][2]:.4f}", flush=True)
            del model, probability
            gc.collect()


def aggregate(periods):
    rows = []
    for candidate in CANDIDATES:
        source = [json.loads((OUT / "resultados" / f"{candidate['name']}_{p}.json").read_text(encoding="utf-8"))
                  for p in periods]
        matrix = sum((np.asarray(item["metrics"]["matriz_confusion"]) for item in source),
                     np.zeros((3, 3), dtype=np.int64))
        rows.append({"candidate": candidate, "periods": periods, "metrics": matrix_metrics(matrix),
                     "monthly": source})
    return rows


def report():
    validation = aggregate(VALIDATION)
    previous = json.loads(REFERENCE.read_text(encoding="utf-8"))
    reference = next(item for item in previous["candidates"] if item["name"] == "temporal_3m")
    original = reference["metrics"]
    lines = ["# Historia larga frente a tres meses, modelo mensual de buffer 300 m", "",
             "Se conservan la fuente, los tramos, las etiquetas, las 74 variables de contexto, "
             "XGBoost y el protocolo mensual. Solo cambia cuántos meses anteriores "
             "se usan como ejemplos de entrenamiento. La historia completa empieza en abril "
             "de 2018 porque los primeros tres meses se necesitan para formar rezagos.", "",
             "## Validación noviembre–diciembre de 2025", "",
             "| Ventana | Accuracy | F1 medio+alto | Precisión alto | Detección alto |",
             "|---|---:|---:|---:|---:|"]
    def row(name, m):
        return (f"| {name} | {m['accuracy']*100:.2f} % | {m['f1_medio_alto']*100:.2f} % | "
                f"{m['precision'][2]*100:.2f} % | {m['recall'][2]*100:.2f} % |")
    lines.append(row("3 meses (referencia)", original))
    lines.extend(row(item["candidate"]["name"], item["metrics"]) for item in validation)
    lines += ["", "Las filas de 2018–2024 sí entran al ajuste de las variantes de historia "
              "completa. Esta tabla no prueba por sí sola que el mejor modelo en años futuros "
              "sea el de tres meses; se requieren observaciones posteriores sin ajustar a ellas.", ""]
    if all((OUT / "resultados" / f"{c['name']}_{p}.json").exists() for c in CANDIDATES for p in EVALUATION):
        baseline = json.loads((ROOT / "outputs/evaluacion-modelos-sidpol/optimizacion_buffer_300_mensual_v2/final_results.json").read_text(encoding="utf-8"))
        grouped = {}
        for title, months, names in (
            ("enero–mayo", EVALUATION[:5], ("benchmark",)),
            ("junio–agosto", EVALUATION[5:], ("additional_evaluation",)),
            ("enero–agosto", EVALUATION, ("benchmark", "additional_evaluation")),
        ):
            evaluation = aggregate(months)
            pieces = [np.asarray(baseline["evaluations"][name]["metrics"]["matriz_confusion"])
                      for name in names]
            reference_metrics = matrix_metrics(sum(pieces))
            grouped[title] = {"periods": months, "reference": reference_metrics, "variants": evaluation}
            lines += [f"## Evaluación retrospectiva {title} de 2026", "",
                      "| Ventana | Accuracy | F1 medio+alto | Precisión alto | Detección alto |",
                      "|---|---:|---:|---:|---:|",
                      row("3 meses (referencia)", reference_metrics)]
            lines.extend(row(item["candidate"]["name"], item["metrics"]) for item in evaluation)
            lines.append("")
        lines += ["", "Los meses de 2026 ya se han consultado en ensayos anteriores: esta evaluación "
                  "es exploratoria, no una prueba virgen.", ""]
        save_json(OUT / "evaluacion.json", grouped)
    save_json(OUT / "validacion.json", validation)
    (OUT / "informe.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "informe.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validate", "evaluate", "report"))
    stage = parser.parse_args().stage
    {"validate": lambda: run_periods(VALIDATION),
     "evaluate": lambda: run_periods(EVALUATION), "report": report}[stage]()
