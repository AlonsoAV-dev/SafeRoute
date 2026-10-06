"""Recupera probabilidades de C (enero-agosto 2026) sin cambiar el protocolo.

Repite las configuraciones congeladas de B y conserva las evidencias originales.
PR-AUC Alto se calcula como Average Precision sobre P(Alto), agrupando los meses.
"""
from __future__ import annotations

import csv
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import validacion_temporal_mensual as original

OUT = ROOT / "outputs/resultados_modelos_buffer300_80pct_2026_10_02/pr_auc_rolling_2026"
PERIODS = tuple(f"2026-{m:02}" for m in range(1, 9))
NAMES = {"random_forest": "Random Forest", "xgboost": "XGBoost", "lstm": "LSTM"}


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    selection = original.verify_frozen()
    frozen_hash = original.digest(original.OUT / "seleccion_congelada_antes_C.json")
    old_results = original.read(original.OUT / "resultados_finales_ABC.json")
    signature = original.fingerprint()
    if (original.read(original.PANEL_META)["fingerprint"] != signature or
            original.read(original.TEMPORAL_DATA / "secuencias_complete.json")["fingerprint"] != signature):
        raise ValueError("Los datos difieren de los utilizados en el protocolo original")
    inputs = {str(path): original.digest(path) for path in (original.PANEL, original.TEMPORAL)}
    original.save_json(OUT / "registro_reproduccion.json", {
        "started_at_utc": original.now(), "periods": list(PERIODS),
        "original_results": str(original.OUT / "resultados_finales_ABC.json"),
        "selection_sha256": frozen_hash, "selected": selection["selected"],
        "original_code_sha256": original.code_hashes(), "input_sha256": inputs,
        "reproduction_script_sha256": original.digest(Path(__file__)),
        "versions": {lib: importlib.metadata.version(lib) for lib in
                     ("numpy", "scikit-learn", "torch", "xgboost")},
        "definition": "sklearn.metrics.average_precision_score(y == 2, P[:, 2])",
        "evaluation": "validación retrospectiva; mismos ocho orígenes y selección congelada de B",
        "original_outputs_modified": False,
    })
    data = original.RollingData()
    for number, period in enumerate(PERIODS, 1):
        month = original.PERIODS.index(period)
        groups = {}
        for family in original.FAMILIES:
            groups.setdefault(selection["selected"][family]["window"], []).append(family)
        for window, families in groups.items():
            entries, y, weights, training = data.selection(month, window)
            x = current = None
            for family in families:
                previous = original.read(original.OUT / "C" / f"{family}_{period}.json")
                if training != previous["training"]:
                    raise ValueError(f"Cambió la muestra de entrenamiento: {family}, {period}")
                path = OUT / f"probabilidades_{family}_{period}.npz"
                row_path = OUT / f"resultado_{family}_{period}.json"
                if path.exists() and row_path.exists():
                    existing = original.read(row_path)
                    if existing["selection_sha256"] != frozen_hash or existing["training"] != training:
                        raise ValueError("La reproducción parcial no corresponde al protocolo congelado")
                    print(f"REUTILIZADO {family} {period}", flush=True)
                    continue
                print(f"REPRODUCCIÓN {number}/8 {family} {period} ventana={window}", flush=True)
                original.seed()
                fit_start = time.perf_counter()
                if family == "lstm":
                    chosen_epochs = selection["selected"][family]["epochs"]
                    model, scales, history = original.fit_lstm(
                        data, entries, y, weights, month, chosen_epochs)
                    probability = original.lstm_predict(model, data, month, scales)
                    del scales, history
                else:
                    if x is None:
                        x = data.training_tabular(entries)
                        current = data.selected_features(month, np.arange(data.n))
                    model = (original.fit_model(original.SPEC, x, y, weights)
                             if family == "xgboost" else
                             RandomForestClassifier(**original.RF_PARAMETERS).fit(x, y, sample_weight=weights))
                    probability = original.predictions(model, current)
                truth = data.target([month])
                metrics = original.row_metrics(data, month, probability)
                exact = metrics["matriz_confusion"] == previous["metrics"]["matriz_confusion"]
                ap = float(average_precision_score(truth == 2, probability[:, 2]))
                np.savez_compressed(path, truth=truth, probability=probability)
                original.save_json(row_path, {
                    "period": period, "family": family, "window": window, "training": training,
                    "metrics": metrics, "pr_auc_alto_average_precision": ap,
                    "original_matrix_identical": exact,
                    "original_matrix": previous["metrics"]["matriz_confusion"],
                    "selection_sha256": frozen_hash, "probability_sha256": original.digest(path),
                    "seconds": time.perf_counter() - fit_start,
                })
                print(f"LISTO {family} {period}: AP_Alto={ap:.6f}; matriz_original_igual={exact}", flush=True)
                del model, probability, truth
                gc.collect()
            del x, current, entries, y, weights
            gc.collect()

    aggregate = {}
    curve_data = {}
    monthly = []
    for family, name in NAMES.items():
        truths, probabilities = [], []
        for period in PERIODS:
            path = OUT / f"probabilidades_{family}_{period}.npz"
            with np.load(path) as archive:
                truths.append(archive["truth"])
                probabilities.append(archive["probability"])
            monthly.append(original.read(OUT / f"resultado_{family}_{period}.json"))
        truth, probability = np.concatenate(truths), np.concatenate(probabilities)
        metrics = original.evaluate(truth, probability)
        expected = old_results["confirmation"][family]["subperiods"]["2026_enero_agosto"]
        exact = metrics["matriz_confusion"] == expected["matriz_confusion"]
        precision, recall, _ = precision_recall_curve(truth == 2, probability[:, 2])
        # Curvas completas para reproducir el gráfico; la métrica usa todas las observaciones.
        np.savez_compressed(OUT / f"curva_precision_recall_{family}_alto.npz",
                            precision=precision, recall=recall)
        curve_data[family] = (precision, recall)
        aggregate[family] = {
            "model": name, "pr_auc_alto_average_precision": float(average_precision_score(truth == 2, probability[:, 2])),
            "metrics": metrics, "observations": int(len(truth)), "high_support": int((truth == 2).sum()),
            "high_prevalence": float((truth == 2).mean()), "original_matrix_identical": exact,
            "all_monthly_matrices_identical": all(r["original_matrix_identical"] for r in monthly if r["family"] == family),
        }
        print(f"FINAL {name}: AP_Alto={aggregate[family]['pr_auc_alto_average_precision']:.8f}; matriz_original_igual={exact}", flush=True)
        del truths, probabilities, truth, probability
        gc.collect()
    result = {
        "completed_at_utc": original.now(), "periods": list(PERIODS), "buffer_m": 300,
        "lighting": False, "coordinates": "originales", "unit": "segmento vial-mes",
        "definition": "Average Precision (AP) one-vs-rest de Alto; los ocho meses se agrupan antes del cálculo",
        "selection_sha256": frozen_hash, "input_sha256": inputs, "models": aggregate,
        "monthly": monthly, "seconds_total": time.perf_counter() - started,
        "original_outputs_modified": False,
    }
    original.save_json(OUT / "pr_auc_alto_tres_modelos.json", result)
    with (OUT / "pr_auc_alto_tres_modelos.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Modelo", "PR_AUC_Alto_AP", "Accuracy", "Precision_Alto", "Recall_Alto", "F1_Alto", "Observaciones", "Matrices_originales_iguales"])
        for family, row in aggregate.items():
            m = row["metrics"]
            writer.writerow([row["model"], row["pr_auc_alto_average_precision"], m["accuracy"], m["precision"][2], m["recall"][2], m["f1"][2], row["observations"], row["all_monthly_matrices_identical"]])
    plt = original.plt
    fig, ax = plt.subplots(figsize=(8.8, 6.3))
    colors = {"random_forest": "#286daf", "xgboost": "#ce8731", "lstm": "#7c60a7"}
    for family, (precision, recall) in curve_data.items():
        # Solo el dibujo se reduce; AP se calcula con todas las probabilidades.
        indexes = np.unique(np.linspace(0, len(precision)-1, min(len(precision), 8000), dtype=int))
        ap = aggregate[family]["pr_auc_alto_average_precision"]
        ax.plot(recall[indexes], precision[indexes], color=colors[family], linewidth=2.1,
                label=f"{NAMES[family]} · AP = {100*ap:.2f} %")
    prevalence = aggregate["random_forest"]["high_prevalence"]
    ax.axhline(prevalence, color="#7d8795", linestyle="--", label=f"Referencia sin discriminación = {100*prevalence:.2f} %")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall de riesgo Alto", ylabel="Precision de riesgo Alto")
    ax.set_title("Precision–Recall de riesgo Alto\nBuffer 300 m · Enero–agosto de 2026 · Sin alumbrado", fontweight="bold", pad=16)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(OUT / "curvas_pr_auc_alto.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({f: r["pr_auc_alto_average_precision"] for f, r in aggregate.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
