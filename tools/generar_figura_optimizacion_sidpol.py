"""Genera una comparación visual de los experimentos de riesgo alto SIDPOL."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Backend" / "data" / "procesados_sidpol_v2"
OUTPUT = ROOT / "outputs" / "evaluacion-modelos-sidpol" / "comparacion_modelos_optimizados_2026.png"
MODELS = ("Random Forest", "XGBoost", "Modelo combinado")
COLORS = ("#2b62ce", "#078f89", "#ef8a34")


def percent(value: float) -> str:
    return f"{100 * value:.1f}".replace(".", ",") + " %"


def main() -> None:
    base = json.loads((DATA / "resumen_entrenamiento.json").read_text(encoding="utf-8"))
    tuned = json.loads((DATA / "experimentos_alto_binario" / "ajuste_comparacion.json").read_text(encoding="utf-8"))
    august = json.loads((DATA / "experimentos_alto_binario" / "holdout_agosto.json").read_text(encoding="utf-8"))
    apr_jul = [
        base["assessments"]["random_forest"]["test"],
        base["assessments"]["xgboost"]["test"],
        tuned["test_combined_with_spatial_rf_for_medium"],
    ]
    aug = [august["models"][key] for key in (
        "random_forest", "xgboost", "alto_binario_medio_rf_vecindad"
    )]
    for evaluation in apr_jul[:2]:
        matrix = np.asarray(evaluation["matriz_confusion"])
        tp = matrix[2, 2]
        precision = tp / matrix[:, 2].sum()
        recall = tp / matrix[2].sum()
        evaluation["precision_alto"] = precision
        evaluation["f1_alto"] = 2 * precision * recall / (precision + recall)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    fig.text(0.055, 0.94, "Riesgo alto: resultados antes y después del ajuste",
             fontsize=25, fontweight="bold", color="#1b2e4f")
    fig.text(0.055, 0.895,
             "Prueba temporal 2026  ·  unidad: segmento vial × mes × turno  ·  mejor configuración elegida en validación 2024",
             fontsize=12, color="#5a6b82")
    fig.add_artist(plt.Line2D([0.055, 0.945], [0.872, 0.872], color="#dce5f1", lw=1.5))

    panels = [
        (fig.add_axes([0.065, 0.40, 0.42, 0.30]), apr_jul, "Abril–julio de 2026", 13_337),
        (fig.add_axes([0.555, 0.40, 0.42, 0.30]), aug, "Agosto de 2026", 2_964),
    ]
    metrics = (("precision_alto", "Precisión"), ("recall_riesgo_alto", "Detección"), ("f1_alto", "F1 alto"))
    x = np.arange(3)
    for ax, evaluations, title, support in panels:
        for index, (name, color) in enumerate(zip(MODELS, COLORS)):
            values = [evaluations[index][field] * 100 for field, _ in metrics]
            bars = ax.bar(x + (index - 1) * 0.235, values, width=0.22, color=color, label=name)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7,
                        f"{value:.1f}".replace(".", ","), ha="center", va="bottom", fontsize=9,
                        color="#263953")
        ax.set_title(f"{title}\n{support:,} casos reales de riesgo alto".replace(",", " "),
                     fontsize=16, fontweight="bold", color="#1b2e4f", loc="left", pad=23)
        ax.set_xticks(x, [label for _, label in metrics], fontsize=12)
        ax.set_ylim(0, 43)
        ax.set_ylabel("Porcentaje (%)", color="#52657e")
        ax.grid(axis="y", color="#e3eaf3", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#dce5f1")
        ax.tick_params(axis="y", length=0, colors="#52657e")
        ax.tick_params(axis="x", length=0, pad=8, colors="#243b59")

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS]
    fig.legend(handles, MODELS, loc="upper center", bbox_to_anchor=(0.5, 0.85),
               ncol=3, frameon=False, fontsize=12)

    header = ["Período", "Modelo", "Casos altos detectados", "F1 macro", "PR-AUC alto"]
    rows = []
    for period, evaluations in (("Abr–jul", apr_jul), ("Agosto", aug)):
        for name, assessment in zip(MODELS, evaluations):
            rows.append([
                period,
                name,
                f"{assessment['matriz_confusion'][2][2]:,}".replace(",", " "),
                f"{assessment['f1_macro']:.3f}".replace(".", ","),
                f"{assessment['pr_auc_riesgo_alto']:.3f}".replace(".", ","),
            ])
    table_ax = fig.add_axes([0.065, 0.18, 0.91, 0.18])
    table_ax.axis("off")
    table = table_ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center",
                           colWidths=[0.12, 0.27, 0.25, 0.15, 0.19], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dce5f1")
        if row == 0:
            cell.set_facecolor("#1b2e4f")
            cell.set_text_props(color="white", weight="bold")
        elif row in (3, 6):
            cell.set_facecolor("#fff4ea")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f9fe")

    fig.text(0.065, 0.115,
             "El modelo combinado mejora el F1 alto en ambos períodos, pero todavía detecta solo alrededor de 15 % de los casos altos.",
             fontsize=12, fontweight="bold", color="#a65b1c")
    fig.text(0.065, 0.075,
             "Abr–jul: modelos entrenados hasta 2023. Agosto: modelos reentrenados hasta julio de 2026; agosto quedó fuera del ajuste.",
             fontsize=10.5, color="#5a6b82")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
