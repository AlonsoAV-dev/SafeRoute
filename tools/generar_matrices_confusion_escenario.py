"""Dibuja las matrices de confusión del escenario de ubicaciones copiadas."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Backend" / "data" / "experimentos_sidpol" / "escenario_distrito"
OUT = ROOT / "outputs" / "evaluacion-modelos-sidpol" / "escenario_distrito"
CLASSES = ("Bajo", "Medio", "Alto")


def format_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def panel(ax, title: str, assessment: dict):
    counts = np.asarray(assessment["matriz_confusion"], dtype=np.int64)
    rates = counts / counts.sum(axis=1, keepdims=True)
    image = ax.imshow(rates * 100, vmin=0, vmax=100, cmap="Blues")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=15)
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Riesgo real")
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(3):
        for col in range(3):
            percentage = f"{rates[row, col] * 100:.1f}".replace(".", ",")
            label = f"{format_int(counts[row, col])}\n{percentage} %"
            ax.text(col, row, label, ha="center", va="center", fontsize=10.5,
                    color="white" if rates[row, col] >= 0.5 else "#15314b", fontweight="bold")
    return image


def main():
    controlled = json.loads((DATA / "modelos" / "evaluacion_observada.json").read_text(encoding="utf-8"))
    recent = json.loads((DATA / "reciente" / "comparacion_entrenamiento_2025.json").read_text(encoding="utf-8"))
    xgb = controlled["candidates"]["xgb_multi_recent"]
    cases = [
        ("A · Original\nEntrena hasta 2023 · selecciona 2024", xgb["original_retrospective"]),
        ("B · Copiadas\nEntrena hasta 2023 · selecciona 2024", xgb["retrospective"]),
        ("C · Copiadas\nEntrena hasta 2023 · selecciona 2026", recent["candidates"]["hasta_2023"]["retrospective"]),
        ("D · Copiadas\nEntrena hasta 2025 · selecciona 2026", recent["candidates"]["hasta_2025"]["retrospective"]),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 12.8), facecolor="white")
    fig.suptitle("Matrices de confusión · XGBoost con ventanas", fontsize=19, fontweight="bold", y=0.975)
    for ax, (title, assessment) in zip(axes.flat, cases):
        panel(ax, title, assessment)
        ax.text(0.5, -0.22,
                f"Aciertos: medio {format_int(assessment['matriz_confusion'][1][1])} / 42.104; "
                f"alto {format_int(assessment['matriz_confusion'][2][2])} / 16.301",
                transform=ax.transAxes, ha="center", fontsize=10, color="#496077")
    fig.text(0.5, 0.055,
             "Cada celda: cantidad de segmentos × mes × turno y porcentaje de su fila real. "
             "La diagonal indica aciertos.",
             ha="center", fontsize=11, color="#3f5268")
    fig.text(0.5, 0.032,
             "A–B comparten la selección de 2024; C–D comparten calibración y selección de 2026. "
             "Prueba: abril–agosto de 2026.",
             ha="center", fontsize=10, color="#677a8b")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.87, bottom=0.18, hspace=0.68, wspace=0.30)
    output = OUT / "matrices_confusion_xgb_escenario.png"
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    for title, assessment in cases:
        tag = title[0].lower()
        matrix = pd.DataFrame(assessment["matriz_confusion"], index=CLASSES, columns=CLASSES)
        matrix.index.name = "riesgo_real"
        matrix.to_csv(OUT / f"matriz_confusion_{tag}.csv", encoding="utf-8-sig")
    names = [
        ("Random Forest", "base_rf", "rf"),
        ("XGBoost base", "base_xgb", "xgb_base"),
        ("XGBoost ventanas", "xgb_multi_recent", "xgb_ventanas"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5), facecolor="white")
    fig.suptitle("Matrices de confusión por modelo · abril–agosto de 2026", fontsize=19, fontweight="bold", y=0.97)
    for col, (label, key, slug) in enumerate(names):
        assessment = controlled["candidates"][key]
        for row, (source, value) in enumerate((("Original", assessment["original_retrospective"]),
                                               ("Copiadas", assessment["retrospective"]))):
            panel(axes[row, col], f"{label} · {source}", value)
            matrix = pd.DataFrame(value["matriz_confusion"], index=CLASSES, columns=CLASSES)
            matrix.index.name = "riesgo_real"
            matrix.to_csv(OUT / f"matriz_confusion_{slug}_{source.lower()}.csv", encoding="utf-8-sig")
    fig.text(0.5, 0.035, "Filas: clase real · columnas: predicción · cada celda muestra cantidad y porcentaje de su fila. "
             "Selección de umbrales: 2024.", ha="center", fontsize=11, color="#526b81")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.10, hspace=0.37, wspace=0.26)
    all_models_output = OUT / "matrices_confusion_todos_modelos.png"
    fig.savefig(all_models_output, dpi=170, facecolor="white")
    plt.close(fig)
    print(output)
    print(all_models_output)


if __name__ == "__main__":
    main()
