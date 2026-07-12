from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from shapely.geometry import LineString


ROOT = Path(__file__).resolve().parents[1]
API_URL = "http://127.0.0.1:8000/api/route/calculate"
OUTPUT = ROOT / "Backend" / "data" / "procesados" / "resultados_rutas"
FIGURE = ROOT / "Documentos" / "figuras" / "figura_6_4_comparacion_rutas.png"
SEGMENTS = ROOT / "Backend" / "data" / "procesados" / "evaluacion_2026" / "tramos_osm.csv"

CASES = [
    {
        "id": "caso_1",
        "title": "Caso 1: Centro de Lima - Gamarra",
        "origin_name": "Plaza San Martín",
        "destination_name": "Gamarra",
        "origin": [-12.0516, -77.0347],
        "destination": [-12.0652, -77.0132],
    },
    {
        "id": "caso_2",
        "title": "Caso 2: San Miguel - Miraflores",
        "origin_name": "San Miguel",
        "destination_name": "Parque Kennedy",
        "origin": [-12.0770, -77.0920],
        "destination": [-12.1211, -77.0297],
    },
    {
        "id": "caso_3",
        "title": "Caso 3: Plaza Norte - Miraflores",
        "origin_name": "Plaza Norte",
        "destination_name": "Parque Kennedy",
        "origin": [-11.9940, -77.0610],
        "destination": [-12.1211, -77.0297],
    },
]


def request_route(case: dict) -> dict:
    payload = {
        "origin": case["origin"],
        "destination": case["destination"],
        "routePreference": "safe",
        "modelo_riesgo": "xgboost",
        "beta": 10,
        "buffer_m": 200,
        "risk_mode": "predicted",
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    return {"case": case, "request": payload, "response": result}


def save_results(results: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in results:
        case = item["case"]
        response = item["response"]
        (OUTPUT / f"{case['id']}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fast = response["traditional_route"]
        safe = response["safe_route"]
        for route_name, route in (("Ruta corta", fast), ("Ruta segura", safe)):
            rows.append(
                {
                    "caso": case["id"],
                    "origen": case["origin_name"],
                    "destino": case["destination_name"],
                    "ruta": route_name,
                    "distancia_km": route["distance_km"],
                    "tiempo_min": route["time_min"],
                    "riesgo_acumulado": route["risk_total"],
                    "riesgo_promedio": route["risk_average"],
                    "reduccion_riesgo_pct": (
                        response["risk_reduction"] if route_name == "Ruta segura" else ""
                    ),
                    "beta": response["parametros_a_star"]["beta_ruta_segura"],
                    "modelo": response["modelo_usado"],
                    "periodo_prediccion": response["periodo_prediccion"],
                }
            )
    with (OUTPUT / "resumen_casos.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def route_line(route: dict) -> LineString:
    return LineString([(point["lng"], point["lat"]) for point in route["route"]])


def load_background(results: list[dict]) -> pd.DataFrame:
    bounds = []
    for item in results:
        for route in (item["response"]["traditional_route"], item["response"]["safe_route"]):
            line = route_line(route)
            bounds.append(line.bounds)
    min_lng = min(bound[0] for bound in bounds) - 0.012
    min_lat = min(bound[1] for bound in bounds) - 0.012
    max_lng = max(bound[2] for bound in bounds) + 0.012
    max_lat = max(bound[3] for bound in bounds) + 0.012
    segments = pd.read_csv(
        SEGMENTS,
        usecols=["latitud", "longitud", "geometria"],
    )
    return segments.loc[
        segments["latitud"].between(min_lat, max_lat)
        & segments["longitud"].between(min_lng, max_lng)
    ].copy()


def plot_case(ax, item: dict, segments: pd.DataFrame) -> None:
    case = item["case"]
    response = item["response"]
    fast = response["traditional_route"]
    safe = response["safe_route"]
    fast_line = route_line(fast)
    safe_line = route_line(safe)
    bounds = [fast_line.bounds, safe_line.bounds]
    min_lng = min(bound[0] for bound in bounds) - 0.006
    min_lat = min(bound[1] for bound in bounds) - 0.006
    max_lng = max(bound[2] for bound in bounds) + 0.006
    max_lat = max(bound[3] for bound in bounds) + 0.006

    if case["id"] == "caso_3":
        center_lng = (min_lng + max_lng) / 2
        desired_width = (max_lat - min_lat) * 1.55
        min_lng = center_lng - desired_width / 2
        max_lng = center_lng + desired_width / 2

    local = segments.loc[
        segments["latitud"].between(min_lat, max_lat)
        & segments["longitud"].between(min_lng, max_lng)
    ].copy()
    geometry = gpd.GeoSeries.from_wkt(local.pop("geometria"), crs="EPSG:4326")
    gpd.GeoDataFrame(local, geometry=geometry).plot(
        ax=ax, color="#cbd5e1", linewidth=0.35, alpha=0.72
    )
    gpd.GeoSeries([fast_line], crs="EPSG:4326").plot(
        ax=ax, color="#334155", linewidth=2.8, linestyle="--", alpha=0.95
    )
    gpd.GeoSeries([safe_line], crs="EPSG:4326").plot(
        ax=ax, color="#0284c7", linewidth=3.0, alpha=0.95
    )
    ax.scatter(case["origin"][1], case["origin"][0], s=42, color="#16a34a", edgecolor="white", zorder=5)
    ax.scatter(case["destination"][1], case["destination"][0], s=42, color="#dc2626", edgecolor="white", zorder=5)
    ax.set_xlim(min_lng, max_lng)
    ax.set_ylim(min_lat, max_lat)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(case["title"], fontsize=10.5, weight="bold", pad=7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")
        spine.set_linewidth(0.65)
    distance_increase = (safe["distance_km"] / fast["distance_km"] - 1) * 100
    ax.text(
        0.02,
        0.02,
        f"Riesgo: -{response['risk_reduction']:.2f}% | Distancia: +{distance_increase:.2f}%",
        transform=ax.transAxes,
        fontsize=8,
        color="#0f172a",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.94, "pad": 3},
    )


def plot_results(results: list[dict]) -> None:
    background = load_background(results)
    fig = plt.figure(figsize=(11.5, 8.5), facecolor="white")
    grid = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.12], hspace=0.22, wspace=0.12)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    for ax, item in zip(axes, results):
        plot_case(ax, item, background)
    legend = [
        Line2D([0], [0], color="#334155", lw=2.8, linestyle="--", label="Ruta corta"),
        Line2D([0], [0], color="#0284c7", lw=3.0, label="Ruta segura"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#16a34a", markeredgecolor="white", markersize=8, label="Origen"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#dc2626", markeredgecolor="white", markersize=8, label="Destino"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Comparación experimental entre ruta corta y ruta segura", fontsize=14, weight="bold", y=0.99)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.03, right=0.97)
    fig.savefig(FIGURE, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    results = [request_route(case) for case in CASES]
    save_results(results)
    plot_results(results)
    print(OUTPUT)
    print(FIGURE)


if __name__ == "__main__":
    main()
