"""Dibuja la matriz de confusión de la base nueva por radio de buffer."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "evaluacion-modelos-sidpol" / "comparacion_67"
def main(radius: int):
    if radius == 200:
        frame = pd.read_csv(OUT / "matriz_confusion_b.csv", index_col=0)
        counts = frame.to_numpy(dtype=np.int64)
        result = json.loads((OUT / "resultado_b_base_nueva_buffer_mes.json").read_text(encoding="utf-8"))
    else:
        result = json.loads((OUT / f"resultado_b_base_nueva_buffer_{radius}m_mes.json").read_text(encoding="utf-8"))
        counts = np.asarray(result["metrics"]["matriz_confusion"], dtype=np.int64)
    destination = OUT / f"matriz_confusion_buffer_{radius}m.png"
    share = counts / counts.sum(axis=1, keepdims=True) * 100
    labels = ["Bajo", "Medio", "Alto"]

    fig, ax = plt.subplots(figsize=(10.6, 8.0), facecolor="white")
    fig.subplots_adjust(top=0.79, bottom=0.22, left=0.15, right=0.88)
    image = ax.imshow(share, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(np.arange(3), labels)
    ax.set_yticks(np.arange(3), labels)
    ax.tick_params(axis="both", labelsize=15, length=0, pad=11)
    ax.set_xlabel("Riesgo predicho", fontsize=16, labelpad=13)
    ax.set_ylabel("Riesgo real", fontsize=16, labelpad=13)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(3):
        for j in range(3):
            color = "white" if share[i, j] >= 55 else "#172d45"
            ax.text(j, i - .08, f"{counts[i, j]:,}".replace(",", "."),
                    ha="center", va="center", fontsize=20, weight="bold", color=color)
            ax.text(j, i + .24, f"{share[i, j]:.1f}".replace(".", ",") + " % de la fila",
                    ha="center", va="center", fontsize=12, color=color)

    cbar = fig.colorbar(image, ax=ax, fraction=.045, pad=.035)
    cbar.set_label("Porcentaje dentro de la clase real", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.suptitle(f"Matriz de confusión · Excel 2018–2026 + buffer {radius} m",
                 fontsize=20, weight="bold", y=.96)
    fig.text(.5, .88, "Prueba enero–mayo de 2026  ·  Unidad: tramo vial × mes",
             ha="center", fontsize=13, color="#516b82")
    recall = result["metrics"]["recall"][2]
    precision = result["metrics"]["precision"][2]
    footer = (f"Riesgo alto: {recall:.2%} detectado  ·  "
              f"{precision:.2%} de precisión en las alertas").replace(".", ",")
    fig.text(.5, .045, footer,
             ha="center", fontsize=13, weight="bold", color="#315d8c")
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--radio", type=int, choices=(100, 200, 300), default=200)
    main(parser.parse_args().radio)
