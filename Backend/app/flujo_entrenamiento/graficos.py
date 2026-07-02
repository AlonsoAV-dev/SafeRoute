from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.flujo_entrenamiento.riesgo import VARIABLES_MODELO


NIVELES = ["bajo", "medio", "alto"]


def generar_graficos(
    metricas: dict,
    reporte: pd.DataFrame,
    matriz: np.ndarray,
    predicciones: pd.DataFrame,
    modelo,
    salida: Path,
    comparacion_radios: pd.DataFrame | None = None,
) -> list[str]:
    """Genera visualizaciones reproducibles de la evaluación y las predicciones."""
    salida.mkdir(parents=True, exist_ok=True)
    archivos = [
        _grafico_metricas(metricas, salida / "01_metricas_random_forest.png"),
        _grafico_matriz(matriz, salida / "02_matriz_confusion.png"),
        _grafico_clases(reporte, salida / "03_metricas_por_clase.png"),
        _grafico_distribucion(
            predicciones, salida / "04_distribucion_predicciones.png"
        ),
        _grafico_importancia(
            modelo, salida / "05_importancia_variables.png"
        ),
    ]
    if comparacion_radios is not None and not comparacion_radios.empty:
        archivos.append(
            _grafico_radios(
                comparacion_radios, salida / "06_comparacion_radios.png"
            )
        )
    return [archivo.name for archivo in archivos]


def generar_desde_archivos(directorio: Path) -> list[str]:
    metricas = pd.read_csv(directorio / "metricas_random_forest.csv").iloc[0].to_dict()
    reporte = pd.read_csv(
        directorio / "classification_report_random_forest.csv", index_col=0
    )
    matriz = pd.read_csv(
        directorio / "matriz_confusion_random_forest.csv", index_col=0
    ).to_numpy()
    predicciones = pd.read_csv(
        directorio / "predicciones_tramos.csv",
        usecols=["nivel_riesgo", "riesgo_score"],
    )
    artefacto = joblib.load(directorio / "modelo_random_forest.joblib")
    ruta_radios = directorio / "comparacion_radios.csv"
    comparacion_radios = pd.read_csv(ruta_radios) if ruta_radios.exists() else None
    return generar_graficos(
        metricas,
        reporte,
        matriz,
        predicciones,
        artefacto["pipeline"],
        directorio / "graficos",
        comparacion_radios,
    )


def _grafico_metricas(metricas: dict, ruta: Path) -> Path:
    claves = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Accuracy balanceada"),
        ("precision_macro", "Precisión macro"),
        ("recall_macro", "Recall macro"),
        ("f1_macro", "F1 macro"),
        ("recall_riesgo_alto", "Recall alto"),
        ("pr_auc_riesgo_alto", "PR-AUC alto"),
    ]
    etiquetas = [etiqueta for clave, etiqueta in claves]
    valores = [float(metricas[clave]) for clave, _ in claves]
    colores = ["#2563eb", "#0891b2", "#059669", "#65a30d", "#ca8a04", "#ea580c", "#dc2626"]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    barras = ax.bar(etiquetas, valores, color=colores)
    ax.bar_label(barras, labels=[f"{valor:.3f}" for valor in valores], padding=4)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Puntuación")
    ax.set_title("Resultados de Random Forest en el periodo de prueba", weight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return ruta


def _grafico_matriz(matriz: np.ndarray, ruta: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    imagen = ax.imshow(matriz, cmap="YlOrRd")
    limite = float(matriz.max()) / 2 if matriz.size else 0
    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            valor = int(matriz[fila, columna])
            ax.text(
                columna,
                fila,
                f"{valor:,}",
                ha="center",
                va="center",
                color="white" if valor > limite else "#111827",
                weight="bold",
            )
    ax.set_xticks(range(3), [nivel.capitalize() for nivel in NIVELES])
    ax.set_yticks(range(3), [nivel.capitalize() for nivel in NIVELES])
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusión", weight="bold")
    fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return ruta


def _grafico_clases(reporte: pd.DataFrame, ruta: Path) -> Path:
    datos = reporte.loc[NIVELES, ["precision", "recall", "f1-score"]].astype(float)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    datos.plot.bar(ax=ax, color=["#2563eb", "#f59e0b", "#16a34a"], width=0.78)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Nivel de riesgo")
    ax.set_ylabel("Puntuación")
    ax.set_title("Métricas por clase", weight="bold")
    ax.set_xticklabels([nivel.capitalize() for nivel in NIVELES], rotation=0)
    ax.legend(["Precisión", "Recall", "F1"], frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return ruta


def _grafico_distribucion(predicciones: pd.DataFrame, ruta: Path) -> Path:
    conteos = predicciones["nivel_riesgo"].value_counts().reindex(NIVELES, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5.8))
    barras = ax.bar(
        [nivel.capitalize() for nivel in NIVELES],
        conteos.values,
        color=["#16a34a", "#f59e0b", "#dc2626"],
    )
    ax.bar_label(barras, labels=[f"{int(valor):,}" for valor in conteos], padding=4)
    ax.set_ylabel("Cantidad de tramos")
    ax.set_title("Distribución del riesgo predicho", weight="bold")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return ruta


def _grafico_importancia(modelo, ruta: Path) -> Path:
    importancias = modelo.named_steps["modelo"].feature_importances_
    datos = pd.Series(importancias, index=VARIABLES_MODELO).nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(datos.index, datos.values, color="#0f766e")
    ax.set_xlabel("Importancia")
    ax.set_title("Variables más importantes del modelo", weight="bold")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return ruta


def _grafico_radios(comparacion: pd.DataFrame, ruta: Path) -> Path:
    datos = comparacion.sort_values("radio_buffer_m").set_index("radio_buffer_m")
    columnas = ["f1_macro", "recall_riesgo_alto", "pr_auc_riesgo_alto"]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    datos[columnas].plot.bar(
        ax=ax,
        color=["#2563eb", "#f97316", "#16a34a"],
        width=0.78,
    )
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Radio del buffer (metros)")
    ax.set_ylabel("Puntuación")
    ax.set_title("Comparación temporal de radios espaciales", weight="bold")
    ax.set_xticklabels([f"{int(valor)} m" for valor in datos.index], rotation=0)
    ax.legend(["F1 macro", "Recall alto", "PR-AUC alto"], frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return ruta


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera gráficos del Random Forest entrenado.")
    parser.add_argument(
        "--directorio",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "procesados",
    )
    argumentos = parser.parse_args()
    for archivo in generar_desde_archivos(argumentos.directorio):
        print(argumentos.directorio / "graficos" / archivo)


if __name__ == "__main__":
    main()
