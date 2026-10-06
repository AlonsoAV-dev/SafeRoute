"""Genera una figura legible de la evaluación temporal SIDPOL."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Backend/data/procesados_sidpol_v2/resumen_entrenamiento.json"
OUTPUT = ROOT / "outputs/evaluacion-modelos-sidpol/evaluacion_modelos_abril_julio_2026.png"

NAVY = "#172B4D"
MUTED = "#50627A"
BLUE = "#2563EB"
TEAL = "#0D9488"
PALE = "#F2F6FB"
GRID = "#D9E3EF"
ORANGE = "#B45309"
LABELS = ("Bajo", "Medio", "Alto")


def percent(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f} %".replace(".", ",")


def decimal(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def plot_confusion(ax, matrix: np.ndarray, title: str, accent: str) -> None:
    support = matrix.sum(axis=1)
    normalized = np.divide(matrix, support[:, None], out=np.zeros_like(matrix, dtype=float), where=support[:, None] > 0) * 100
    cmap = LinearSegmentedColormap.from_list("matrix", ["#F3F7FB", "#8AB6E8", "#16509B"])
    ax.imshow(normalized, vmin=0, vmax=100, cmap=cmap)
    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", color=accent, pad=18)
    ax.set_xticks(range(3), LABELS, fontsize=11, color=NAVY)
    ax.set_yticks(range(3), [f"{label}\n(n={count:,.0f})" for label, count in zip(LABELS, support)], fontsize=10, color=NAVY)
    ax.set_xlabel("Riesgo predicho", fontsize=11, color=MUTED, labelpad=12)
    ax.set_ylabel("Riesgo real", fontsize=11, color=MUTED, labelpad=12)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(3):
        for column in range(3):
            share = normalized[row, column]
            color = "white" if share >= 57 else NAVY
            ax.text(column, row - 0.09, percent(share / 100, 1), ha="center", va="center", fontsize=12, fontweight="bold", color=color)
            ax.text(column, row + 0.21, f"{matrix[row, column]:,.0f}", ha="center", va="center", fontsize=9, color=color)


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    results = data["assessments"]
    rf = results["random_forest"]["test"]
    xgb = results["xgboost"]["test"]
    support = np.asarray(rf["soporte_clases"], dtype=int)
    total = int(support.sum())
    low_baseline_accuracy = support[0] / total
    high_prevalence = support[2] / total
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "white"})
    figure = plt.figure(figsize=(16, 10.5), dpi=170)
    grid = figure.add_gridspec(3, 2, height_ratios=[0.9, 1.65, 6.0], hspace=0.18, wspace=0.29, left=0.105, right=0.96, top=0.965, bottom=0.115)

    header = figure.add_subplot(grid[0, :])
    header.axis("off")
    header.text(0, 0.85, "Evaluación de los modelos de riesgo vial", fontsize=25, fontweight="bold", color=NAVY, transform=header.transAxes)
    header.text(0, 0.28, "Prueba temporal externa: abril–julio de 2026  ·  Unidad: segmento vial × mes × turno", fontsize=13, color=MUTED, transform=header.transAxes)
    header.plot([0, 1], [0.04, 0.04], color=GRID, lw=1.5, transform=header.transAxes)

    summary = figure.add_subplot(grid[1, :])
    summary.axis("off")
    summary.text(0, 0.99, "Resultados en el conjunto de prueba", fontsize=14, fontweight="bold", color=NAVY, va="top", transform=summary.transAxes)
    names = ("Random Forest", "XGBoost")
    evaluations = (rf, xgb)
    rows = []
    for name, evaluation in zip(names, evaluations):
        matrix = np.asarray(evaluation["matriz_confusion"], dtype=int)
        precision_high = matrix[2, 2] / matrix[:, 2].sum()
        rows.append([
            name,
            percent(evaluation["accuracy"]),
            decimal(evaluation["f1_macro"]),
            percent(evaluation["recall_riesgo_alto"]),
            percent(precision_high),
            decimal(evaluation["pr_auc_riesgo_alto"]),
        ])
    table = summary.table(
        cellText=rows,
        colLabels=["Modelo", "Accuracy", "F1 macro", "Recall alto", "Precisión alto", "PR-AUC alto"],
        colWidths=[0.22, 0.14, 0.14, 0.16, 0.18, 0.16],
        cellLoc="center", bbox=[0, 0.04, 1, 0.77],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F5F9FF" if row == 1 else "#F2FAF8")
            cell.get_text().set_color(NAVY)
            if column == 0:
                cell.get_text().set_weight("bold")
                cell.get_text().set_color(BLUE if row == 1 else TEAL)

    plot_confusion(figure.add_subplot(grid[2, 0]), np.asarray(rf["matriz_confusion"], dtype=int), "Random Forest · matriz de confusión", BLUE)
    plot_confusion(figure.add_subplot(grid[2, 1]), np.asarray(xgb["matriz_confusion"], dtype=int), "XGBoost · matriz de confusión", TEAL)

    figure.text(0.105, 0.052, f"Clase alto: {support[2]:,} de {total:,} casos ({percent(high_prevalence)}). Un modelo que siempre predice «bajo» logra {percent(low_baseline_accuracy)} de accuracy.", fontsize=10.8, color=ORANGE)
    figure.text(0.105, 0.028, "Cada fila de la matriz suma 100 %. Random Forest fue elegido por la validación de 2024; ambas matrices corresponden a la prueba de 2026.", fontsize=10, color=MUTED)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=170, facecolor="white")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
