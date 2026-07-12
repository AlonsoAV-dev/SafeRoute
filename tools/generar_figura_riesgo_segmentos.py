from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Backend" / "data" / "procesados" / "evaluacion_2026"
PREDICTIONS = DATA / "tuning" / "predicciones_final_argmax_2026_xgb_shallow.csv"
SEGMENTS = DATA / "tramos_osm.csv"
OUTPUT = ROOT / "Documentos" / "figuras" / "figura_5_3_riesgo_segmento.png"
TARGET_PERIOD = "2026-05"


def main() -> None:
    predictions = pd.read_csv(
        PREDICTIONS,
        usecols=["tramo_id", "periodo_objetivo", "riesgo_score"],
    )
    predictions = predictions.loc[
        predictions["periodo_objetivo"].eq(TARGET_PERIOD),
        ["tramo_id", "riesgo_score"],
    ]
    segments = pd.read_csv(SEGMENTS, usecols=["tramo_id", "geometria"])
    data = segments.merge(predictions, on="tramo_id", how="inner")
    geometry = gpd.GeoSeries.from_wkt(data.pop("geometria"), crs="EPSG:4326")
    roads = gpd.GeoDataFrame(data, geometry=geometry)
    roads = roads.cx[-77.20:-76.88, -12.25:-11.82].copy()

    roads["nivel"] = pd.cut(
        roads["riesgo_score"],
        bins=[-0.001, 0.34, 0.66, 1.001],
        labels=["Bajo", "Medio", "Alto"],
        include_lowest=True,
        right=False,
    )

    styles = {
        "Bajo": {"color": "#65a30d", "linewidth": 0.22, "alpha": 0.24},
        "Medio": {"color": "#f59e0b", "linewidth": 0.38, "alpha": 0.58},
        "Alto": {"color": "#dc2626", "linewidth": 0.62, "alpha": 0.88},
    }

    fig, ax = plt.subplots(figsize=(11.2, 8.1), facecolor="white")
    ax.set_facecolor("#f8fafc")
    for level in ("Bajo", "Medio", "Alto"):
        subset = roads.loc[roads["nivel"].eq(level)]
        subset.plot(ax=ax, **styles[level])

    ax.set_xlim(-77.20, -76.88)
    ax.set_ylim(-12.25, -11.82)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#cbd5e1", linewidth=0.35, alpha=0.45)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.tick_params(labelsize=8, colors="#475569")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")
        spine.set_linewidth(0.6)

    legend = [
        Line2D([0], [0], color=styles[level]["color"], lw=3, label=level)
        for level in ("Bajo", "Medio", "Alto")
    ]
    ax.legend(
        handles=legend,
        title="Nivel de riesgo predicho",
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
        framealpha=0.95,
        fontsize=9,
        title_fontsize=9,
    )
    ax.text(
        0.995,
        0.012,
        "Periodo evaluado: mayo de 2026 | Modelo: XGBoost",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#475569",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92, "pad": 4},
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{OUTPUT}|segmentos={len(roads)}")


if __name__ == "__main__":
    main()
