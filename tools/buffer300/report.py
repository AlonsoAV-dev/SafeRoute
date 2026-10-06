from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import DATA, OUT


def pct(value):
    return f"{100*value:.2f}".replace(".", ",") + " %"


def render_matrix(ax, metrics, title):
    matrix = np.array(metrics["matriz_confusion"])
    share = matrix / matrix.sum(axis=1, keepdims=True) * 100
    ax.imshow(share, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(3), ["Bajo", "Medio", "Alto"], fontsize=12)
    ax.set_yticks(range(3), ["Bajo", "Medio", "Alto"], fontsize=12)
    ax.set_xlabel("Riesgo predicho", fontsize=12, labelpad=8)
    ax.set_ylabel("Riesgo real", fontsize=12, labelpad=8)
    ax.set_title(title + "\nAccuracy: " + pct(metrics["accuracy"]), fontsize=16, weight="bold", pad=15)
    ax.set_xticks(np.arange(-.5, 3), minor=True)
    ax.set_yticks(np.arange(-.5, 3), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="both", length=0)
    for i in range(3):
        for j in range(3):
            color = "white" if share[i, j] >= 55 else "#15344b"
            ax.text(j, i-.08, f"{matrix[i,j]:,}".replace(",", "."), ha="center", va="center",
                    fontsize=18, weight="bold", color=color)
            ax.text(j, i+.23, f"{share[i,j]:.1f} %".replace(".", ","), ha="center", va="center",
                    fontsize=13, color=color)


def generate():
    result = json.loads((OUT / "final_results.json").read_text(encoding="utf-8"))
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    evaluation = result["evaluations"]["benchmark"]
    before, after = evaluation["baseline_metrics"], evaluation["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8), facecolor="white")
    render_matrix(axes[0], before, "Modelo anterior · 300 m")
    render_matrix(axes[1], after, "Modelo optimizado · 300 m")
    fig.suptitle("Mismas etiquetas, mismos tramos, mismos meses de prueba", fontsize=19, weight="bold", y=.98)
    fig.text(.5, .91, "Enero–mayo de 2026 · tramo vial × mes · porcentajes por clase real",
             ha="center", fontsize=12, color="#42617a")
    fig.subplots_adjust(top=.78, bottom=.14, left=.06, right=.98, wspace=.25)
    fig.text(.5, .035, "Evaluación retrospectiva. La fuente incluye coordenadas completadas.",
             ha="center", fontsize=11, color="#42617a")
    fig.savefig(OUT / "matrices_antes_despues.png", dpi=180, facecolor="white")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    render_matrix(ax, after, "Modelo optimizado · buffer 300 m")
    fig.subplots_adjust(top=.78, bottom=.12, left=.15, right=.95)
    fig.text(.5, .94, "Prueba enero–mayo de 2026 · tramo vial × mes", ha="center", fontsize=12)
    fig.savefig(OUT / "matriz_confusion_optimizada.png", dpi=180, facecolor="white")
    plt.close(fig)
    pd.DataFrame(after["matriz_confusion"], index=["real_bajo", "real_medio", "real_alto"],
                 columns=["pred_bajo", "pred_medio", "pred_alto"]).to_csv(OUT / "matriz_confusion_optimizada.csv")
    rows = []
    for row in selection["all_results"]:
        m = row["metrics"]
        rows.append({"ensayo": row["name"], "decision": row["method"], "accuracy": m["accuracy"],
                     "f1_medio_alto": m["f1_medio_alto"], "precision_medio": m["precision"][1],
                     "recall_medio": m["recall"][1], "precision_alto": m["precision"][2],
                     "recall_alto": m["recall"][2], "seleccionado": row["name"] == result["selected"]["name"] and row["method"] == result["selected"]["method"]})
    pd.DataFrame(rows).to_csv(OUT / "resultados_validacion.csv", index=False)
    for group, ev in result["evaluations"].items():
        pd.DataFrame(ev["monthly"]).to_json(OUT / f"metricas_mensuales_{group}.json", orient="records", indent=2)
    fields = [("Accuracy", "accuracy", None), ("Precisión medio", "precision", 1),
              ("Detección medio", "recall", 1), ("F1 medio", "f1", 1),
              ("Precisión alto", "precision", 2), ("Detección alto", "recall", 2),
              ("F1 alto", "f1", 2), ("F1 medio+alto", "f1_medio_alto", None)]
    lines = ["# Optimización del modelo de buffer de 300 m", "",
             "## Comparación sobre la misma prueba", "",
             "| Métrica | Anterior | Optimizado | Cambio (puntos porcentuales) |",
             "|---|---:|---:|---:|"]
    for name, key, index in fields:
        a = before[key] if index is None else before[key][index]
        b = after[key] if index is None else after[key][index]
        lines.append(f"| {name} | {pct(a)} | {pct(b)} | {(b-a)*100:+.2f} |")
    lines += ["", "## Resultados por mes", "",
              "| Mes | Accuracy anterior | Accuracy nuevo | Detección medio nueva | Precisión alto nueva | Detección alto nueva |",
              "|---|---:|---:|---:|---:|---:|"]
    for m in evaluation["monthly"]:
        lines.append(f"| {m['periodo']} | {pct(m['baseline_metrics']['accuracy'])} | {pct(m['accuracy'])} | {pct(m['recall'][1])} | {pct(m['precision'][2])} | {pct(m['recall'][2])} |")
    lines += ["", "## Protocolo y selección", "",
              f"- Configuración seleccionada: `{result['selected']['name']}`; decisión `{result['selected']['method']}`.",
              f"- Se entrenaron {selection['trials']} configuraciones y se evaluaron {selection['decision_variants']} variantes de decisión en validación.",
              "- Selección: entrenamiento hasta agosto de 2025, calibración septiembre–octubre y validación noviembre–diciembre de 2025.",
              f"- Ajuste final: {result['training']['first_month']} a {result['training']['last_month']}; calibración/decisión en {', '.join(result['calibration']) if result['calibration'] else 'ningún mes adicional (regla argmax)'}.",
              "- Prueba común: enero–mayo de 2026. Evaluación adicional: junio–agosto de 2026.",
              "- El modelo anterior entrenó abril–diciembre de 2025. Ambos disponen de etiquetas solo hasta diciembre de 2025 antes de la prueba.",
              "- Horizonte de un mes: las variables se actualizan con los delitos de meses ya transcurridos, sin reentrenar con las etiquetas de prueba.",
              "- Buffer 300 m, decaimiento σ=100 m, umbral alto 2,3442396250791653 y etiquetas del ensayo anterior conservados exactamente.",
              "- Se reprodujo exactamente la matriz de confusión del modelo anterior. Se comprobó que las variables solo consultan meses anteriores.",
              "- Se comprobó que el modelo guardado reproduce las probabilidades del modelo en memoria y que estas son válidas.",
              f"- Adaptación por proporciones del mes anterior: beta={result.get('prior_beta', 0)}.",
              "", "## Evaluación adicional", "",
              "| Modelo | Accuracy | F1 medio+alto | Precisión alto | Detección alto |",
              "|---|---:|---:|---:|---:|"]
    extra = result["evaluations"]["additional_evaluation"]
    for name, m in (("Anterior", extra["baseline_metrics"]), ("Optimizado", extra["metrics"])):
        lines.append(f"| {name} | {pct(m['accuracy'])} | {pct(m['f1_medio_alto'])} | {pct(m['precision'][2])} | {pct(m['recall'][2])} |")
    lines += ["", "## Alcance y límites", "",
              "La distribución de coordenadas cambia sustancialmente al final de 2025. Se mantiene la fuente recibida; el análisis mensual está en `auditoria_fuente_mensual.csv`.",
              "La fuente contiene ubicaciones copiadas de otros delitos. Estos resultados evalúan ese escenario y no verifican la localización real de todos los hechos.",
              "El control de causalidad cubre las variables del modelo. No se rehizo ni se verificó temporalmente la selección original de donantes de coordenadas para cada fecha.",
              "La prueba 2026 había sido consultada en trabajos anteriores. En esta búsqueda no se usó para elegir configuraciones, pero sigue siendo una evaluación retrospectiva exploratoria.",
              "El objetivo es el riesgo del entorno de 300 m del tramo. Un mismo delito contribuye a varios tramos. Las métricas no demuestran precisión de ubicación sobre una calle individual ni seguridad de una ruta.",
              "El artefacto del modelo es independiente del servicio de rutas y está disponible para una integración posterior.",
              "", "## Archivos", "", f"- Modelo: `{result['model_path']}`.",
              "- Resultados completos: `final_results.json`.", "- Búsqueda y selección: `selection.json` y `resultados_validacion.csv`.",
              "- Comparación visual: `matrices_antes_despues.png`.", "- Matriz nueva: `matriz_confusion_optimizada.csv`.",
              "- Ejecución: `.venv/Scripts/python tools/optimizar_buffer_300.py prepare`, después `search`, `finalize` y `report`.", ""]
    (OUT / "informe.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "informe.md", flush=True)
