from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / "Backend" / "data" / "procesados" / "evaluacion_2026" / "tuning"
OUTPUT = ROOT / "Documentos" / "figuras"
MATRIX_PATH = TUNING / "matriz_confusion_final_argmax_xgb_shallow.csv"
MODEL_PATH = TUNING / "modelo_final_argmax_2025_xgb_shallow.joblib"

LABELS = ["Bajo", "Medio", "Alto"]
FEATURE_LABELS = {
    "latitud": "Latitud",
    "longitud": "Longitud",
    "longitud_m": "Longitud del tramo",
    "frecuencia_delitos_hist": "Frecuencia histórica",
    "suma_pesos_hist": "Gravedad histórica acumulada",
    "delitos_graves_hist": "Delitos graves",
    "hurtos_hist": "Hurtos",
    "robos_hist": "Robos",
    "extorsiones_hist": "Extorsiones",
    "homicidios_hist": "Homicidios",
    "delitos_manana_hist": "Delitos en la mañana",
    "delitos_tarde_hist": "Delitos en la tarde",
    "delitos_noche_hist": "Delitos en la noche",
    "delitos_madrugada_hist": "Delitos en la madrugada",
    "densidad_delictiva_100m": "Densidad delictiva por 100 m",
    "densidad_gravedad_100m": "Densidad de gravedad por 100 m",
}


def plot_confusion_matrix() -> None:
    matrix = pd.read_csv(MATRIX_PATH, index_col=0).to_numpy(dtype=int)
    percentages = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor="white")
    image = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=1)
    for row in range(3):
        for column in range(3):
            value = percentages[row, column]
            color = "white" if value >= 0.55 else "#0f172a"
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}\n{value:.1%}",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="semibold",
            )
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Nivel predicho")
    ax.set_ylabel("Nivel real")
    ax.set_title("Matriz de confusión de XGBoost", fontsize=14, weight="bold", pad=14)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.04)
    colorbar.set_label("Proporción dentro de la clase real")
    fig.tight_layout()
    fig.savefig(OUTPUT / "figura_6_1_matriz_confusion_xgboost.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance() -> None:
    artifact = joblib.load(MODEL_PATH)
    variables = artifact["variables"]
    importances = artifact["pipeline"].named_steps["modelo"].feature_importances_
    frame = pd.DataFrame({"variable": variables, "importancia": importances})
    frame["etiqueta"] = frame["variable"].map(FEATURE_LABELS).fillna(frame["variable"])
    frame = frame.nlargest(10, "importancia").sort_values("importancia")

    fig, ax = plt.subplots(figsize=(8.6, 5.9), facecolor="white")
    bars = ax.barh(frame["etiqueta"], frame["importancia"], color="#2563eb")
    for bar, value in zip(bars, frame["importancia"]):
        ax.text(
            value + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
            color="#334155",
        )
    ax.set_xlim(0, max(frame["importancia"].max() * 1.18, 0.1))
    ax.set_xlabel("Importancia relativa")
    ax.set_title("Importancia de variables de XGBoost", fontsize=14, weight="bold", pad=12)
    ax.grid(axis="x", color="#cbd5e1", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figura_6_2_importancia_variables_xgboost.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix()
    plot_feature_importance()
    source_map = OUTPUT / "figura_5_3_riesgo_segmento.png"
    target_map = OUTPUT / "figura_6_3_riesgo_predicho_segmento.png"
    target_map.write_bytes(source_map.read_bytes())
    print(OUTPUT)


if __name__ == "__main__":
    main()
