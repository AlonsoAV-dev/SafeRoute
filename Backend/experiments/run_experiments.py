from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from sklearn.preprocessing import LabelEncoder

BACKEND_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(BACKEND_DIR / "cache" / "matplotlib"))
(BACKEND_DIR / "cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - clear runtime message
    raise SystemExit(
        "Falta matplotlib. Instala dependencias con: python -m pip install -r Backend/requirements.txt"
    ) from exc
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.preprocessing import load_crime_records, normalize_turno  # noqa: E402
from app.services.risk_model import TURNO_RISK, haversine_m, level_from_score  # noqa: E402


RANDOM_STATE = 42
RISK_LEVELS = ["bajo", "medio", "alto"]
FEATURE_COLUMNS = [
    "lat",
    "lng",
    "turno_code",
    "tipo_code",
    "subtipo_code",
    "distrito_code",
    "cluster",
    "cluster_density",
    "crimes_500m",
    "crimes_1000m",
]

LIMA_METROPOLITANA_DISTRICTS = {
    "ANCON",
    "ATE",
    "BARRANCO",
    "BRENA",
    "CARABAYLLO",
    "CHACLACAYO",
    "CHORRILLOS",
    "CIENEGUILLA",
    "COMAS",
    "EL AGUSTINO",
    "INDEPENDENCIA",
    "JESUS MARIA",
    "LA MOLINA",
    "LA VICTORIA",
    "LIMA",
    "LINCE",
    "LOS OLIVOS",
    "LURIGANCHO",
    "LURIN",
    "MAGDALENA DEL MAR",
    "MIRAFLORES",
    "PACHACAMAC",
    "PUCUSANA",
    "PUEBLO LIBRE",
    "PUENTE PIEDRA",
    "PUNTA HERMOSA",
    "PUNTA NEGRA",
    "RIMAC",
    "SAN BARTOLO",
    "SAN BORJA",
    "SAN ISIDRO",
    "SAN JUAN DE LURIGANCHO",
    "SAN JUAN DE MIRAFLORES",
    "SAN LUIS",
    "SAN MARTIN DE PORRES",
    "SAN MIGUEL",
    "SANTA ANITA",
    "SANTA MARIA DEL MAR",
    "SANTA ROSA",
    "SANTIAGO DE SURCO",
    "SURQUILLO",
    "VILLA EL SALVADOR",
    "VILLA MARIA DEL TRIUNFO",
}


@dataclass
class RouteResult:
    scenario: str
    algorithm: str
    objective: str
    distance_km: float
    avg_risk: float
    risk_level: str
    weighted_cost: float
    elapsed_ms: float
    visited_nodes: int
    path_points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta experimentos reales de clustering, prediccion de riesgo y rutas seguras."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=BACKEND_DIR / "data" / "AAV-DATASET.csv",
        help="CSV delictivo georreferenciado.",
    )
    parser.add_argument("--clusters", type=int, default=4, help="K final para el modelo experimental.")
    parser.add_argument("--min-k", type=int, default=2, help="K minimo para evaluar K-Means.")
    parser.add_argument("--max-k", type=int, default=8, help="K maximo para evaluar K-Means.")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Muestra aleatoria opcional para pruebas rapidas. 0 usa todos los registros.",
    )
    parser.add_argument(
        "--safety-weight",
        type=float,
        default=4.0,
        help="Peso de seguridad aplicado al costo de ruta.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BACKEND_DIR / "experiments" / "outputs",
        help="Carpeta raiz donde se guardan las corridas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset if args.dataset.exists() else BACKEND_DIR / "data" / "sample_crimes.csv"
    output_dir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = load_crime_records(dataset_path)
    if len(records) < 10:
        raise SystemExit("Se necesitan al menos 10 registros validos para una corrida experimental.")

    df = records_to_dataframe(records)
    total_valid_records = len(df)
    df = filter_lima_metropolitana(df)
    if len(df) < 10:
        raise SystemExit("No hay suficientes registros de Lima Metropolitana para experimentar.")
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=RANDOM_STATE).reset_index(drop=True)

    kmeans_metrics, selected_k = evaluate_kmeans(df, args.min_k, args.max_k, args.clusters)
    kmeans_metrics.to_csv(output_dir / "kmeans_metrics.csv", index=False)

    df, final_kmeans = assign_clusters(df, selected_k)
    df, encoders = engineer_features_and_labels(df)
    df.to_csv(output_dir / "crime_experiment_dataset.csv", index=False)

    rf_metrics, class_report, feature_importances, y_test, y_pred = run_random_forest(df)
    pd.DataFrame([rf_metrics]).to_csv(output_dir / "random_forest_metrics.csv", index=False)
    pd.DataFrame(class_report).transpose().to_csv(output_dir / "random_forest_classification_report.csv")
    feature_importances.to_csv(output_dir / "random_forest_feature_importance.csv", index=False)

    route_metrics, route_paths = run_routing_experiments(df, final_kmeans, args.safety_weight)
    pd.DataFrame([asdict(result) for result in route_metrics]).to_csv(
        output_dir / "route_metrics.csv", index=False
    )
    with (output_dir / "route_paths.json").open("w", encoding="utf-8") as file:
        json.dump(route_paths, file, ensure_ascii=False, indent=2)

    plot_kmeans_metrics(kmeans_metrics, selected_k, plots_dir / "kmeans_elbow_silhouette.png")
    plot_cluster_map(df, final_kmeans, plots_dir / "kmeans_risk_clusters.png")
    plot_risk_distribution(df, plots_dir / "risk_distribution.png")
    plot_confusion(y_test, y_pred, plots_dir / "random_forest_confusion_matrix.png")
    plot_feature_importance(feature_importances, plots_dir / "random_forest_feature_importance.png")
    plot_routes(df, route_paths, plots_dir / "route_comparison.png")
    plot_route_metrics(route_metrics, plots_dir / "route_metrics.png")

    summary = {
        "dataset": str(dataset_path),
        "scope": "Lima Metropolitana",
        "valid_records_before_scope_filter": int(total_valid_records),
        "records_used": int(len(df)),
        "districts_used": sorted(df["distrito"].unique()),
        "selected_k": int(selected_k),
        "rf_metrics": rf_metrics,
        "route_scenarios": len(route_metrics),
        "safety_weight": args.safety_weight,
        "outputs": {
            "kmeans_metrics": "kmeans_metrics.csv",
            "random_forest_metrics": "random_forest_metrics.csv",
            "route_metrics": "route_metrics.csv",
            "plots": sorted(path.name for path in plots_dir.glob("*.png")),
        },
        "labeling_note": (
            "risk_label es una etiqueta experimental construida con densidad de cluster, "
            "delitos cercanos y turno; no reemplaza una etiqueta policial oficial de riesgo."
        ),
        "encoders": {name: list(encoder.classes_) for name, encoder in encoders.items()},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"Experimento completado: {output_dir}")
    print(f"Graficos: {plots_dir}")


def records_to_dataframe(records) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat": record.lat,
            "lng": record.lng,
            "turno": normalize_turno(record.turno),
            "tipo": record.tipo,
            "subtipo": record.subtipo,
            "distrito": normalize_place_name(record.distrito),
            "fecha": record.fecha,
        }
        for record in records
    )


def normalize_place_name(value: str) -> str:
    text = (value or "NO ESPECIFICADO").strip().upper()
    try:
        text = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        pass
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.split())


def filter_lima_metropolitana(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df["distrito"].isin(LIMA_METROPOLITANA_DISTRICTS)].copy()
    return filtered.reset_index(drop=True)


def evaluate_kmeans(df: pd.DataFrame, min_k: int, max_k: int, requested_k: int) -> tuple[pd.DataFrame, int]:
    coordinates = df[["lat", "lng"]].to_numpy()
    max_possible_k = min(max_k, len(df) - 1)
    rows = []

    for k in range(max(2, min_k), max_possible_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(coordinates)
        rows.append(
            {
                "k": k,
                "inertia": round(float(model.inertia_), 6),
                "silhouette": round(float(silhouette_score(coordinates, labels)), 6),
                "davies_bouldin": round(float(davies_bouldin_score(coordinates, labels)), 6),
                "calinski_harabasz": round(float(calinski_harabasz_score(coordinates, labels)), 3),
            }
        )

    metrics = pd.DataFrame(rows)
    if requested_k in set(metrics["k"]):
        selected_k = requested_k
    else:
        selected_k = int(metrics.sort_values("silhouette", ascending=False).iloc[0]["k"])
    return metrics, selected_k


def assign_clusters(df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, KMeans]:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(df[["lat", "lng"]].to_numpy())
    cluster_counts = df["cluster"].value_counts()
    df["cluster_density"] = df["cluster"].map(cluster_counts) / cluster_counts.max()
    return df, kmeans


def engineer_features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    df = df.copy()
    encoders = {}
    for column in ["turno", "tipo", "subtipo", "distrito"]:
        encoder = LabelEncoder()
        df[f"{column}_code"] = encoder.fit_transform(df[column].fillna("NO ESPECIFICADO"))
        encoders[column] = encoder

    points_rad = np.radians(df[["lat", "lng"]].to_numpy())
    tree = BallTree(points_rad, metric="haversine")
    earth_radius_m = 6_371_000
    df["crimes_500m"] = tree.query_radius(points_rad, r=500 / earth_radius_m, count_only=True) - 1
    df["crimes_1000m"] = tree.query_radius(points_rad, r=1000 / earth_radius_m, count_only=True) - 1

    density_score = normalize_series(df["cluster_density"])
    local_score = normalize_series(df["crimes_500m"])
    turno_score = df["turno"].map(TURNO_RISK).fillna(TURNO_RISK["noche"])
    df["risk_score_experimental"] = np.clip(
        0.55 * density_score + 0.25 * local_score + 0.20 * turno_score / max(TURNO_RISK.values()),
        0,
        1,
    )
    low_cut, high_cut = df["risk_score_experimental"].quantile([0.34, 0.67]).to_list()
    df["risk_label"] = pd.cut(
        df["risk_score_experimental"],
        bins=[-0.001, low_cut, high_cut, 1.001],
        labels=RISK_LEVELS,
        include_lowest=True,
    ).astype(str)
    return df, encoders


def normalize_series(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()
    if math.isclose(float(maximum), float(minimum)):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - minimum) / (maximum - minimum)


def run_random_forest(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df["risk_label"]
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=stratify
    )
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=9,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
    }
    report = classification_report(y_test, y_pred, labels=RISK_LEVELS, output_dict=True, zero_division=0)
    feature_importances = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return metrics, report, feature_importances, y_test, y_pred


def run_routing_experiments(
    df: pd.DataFrame,
    kmeans: KMeans,
    safety_weight: float,
) -> tuple[list[RouteResult], dict]:
    route_df = df.reset_index(drop=True)
    route_pairs = choose_route_pairs(route_df)
    graph, node_positions = build_experimental_graph(route_df)

    node_risk = {
        node: predict_grid_risk(node[0], node[1], df, kmeans)
        for node in graph.nodes
    }
    for u, v, data in graph.edges(data=True):
        distance = haversine_m(u[0], u[1], v[0], v[1])
        risk = (node_risk[u] + node_risk[v]) / 2
        data["length"] = distance
        data["safe_cost"] = distance * (1 + safety_weight * risk)

    results = []
    paths = {}
    for index, (origin, destination) in enumerate(route_pairs, start=1):
        start = nearest_node(graph, origin)
        end = nearest_node(graph, destination)
        scenario = f"ruta_{index}"

        for algorithm, objective, weight in [
            ("dijkstra", "distancia_minima", "length"),
            ("dijkstra", "riesgo_ponderado", "safe_cost"),
            ("astar", "riesgo_ponderado", "safe_cost"),
        ]:
            result, path = compute_route(graph, start, end, algorithm, objective, weight, node_risk)
            results.append(result)
            result.scenario = scenario
            path_key = f"{scenario}_{algorithm}_{objective}"
            paths[path_key] = [{"lat": lat, "lng": lng} for lat, lng in path]

    return results, paths

def choose_route_pairs(df: pd.DataFrame) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    safe = df.sort_values("risk_score_experimental").reset_index(drop=True)
    risky = df.sort_values("risk_score_experimental", ascending=False).reset_index(drop=True)
    mid = df.iloc[len(df) // 2]
    return [
        (
            (float(safe.iloc[min(10, len(safe) - 1)].lat), float(safe.iloc[min(10, len(safe) - 1)].lng)),
            (float(risky.iloc[min(10, len(risky) - 1)].lat), float(risky.iloc[min(10, len(risky) - 1)].lng)),
        ),
        (
            (float(safe.iloc[len(safe) // 4].lat), float(safe.iloc[len(safe) // 4].lng)),
            (float(risky.iloc[len(risky) // 4].lat), float(risky.iloc[len(risky) // 4].lng)),
        ),
        (
            (float(mid.lat), float(mid.lng)),
            (float(risky.iloc[min(25, len(risky) - 1)].lat), float(risky.iloc[min(25, len(risky) - 1)].lng)),
        ),
    ]


def build_experimental_graph(df: pd.DataFrame) -> tuple[nx.Graph, dict]:
    min_lat, max_lat = df["lat"].quantile([0.03, 0.97]).to_list()
    min_lng, max_lng = df["lng"].quantile([0.03, 0.97]).to_list()
    lat_padding = max((max_lat - min_lat) * 0.08, 0.008)
    lng_padding = max((max_lng - min_lng) * 0.08, 0.008)
    min_lat -= lat_padding
    max_lat += lat_padding
    min_lng -= lng_padding
    max_lng += lng_padding

    rows = 28
    cols = 28
    graph = nx.Graph()
    lat_values = [round(float(lat), 6) for lat in np.linspace(min_lat, max_lat, rows)]
    lng_values = [round(float(lng), 6) for lng in np.linspace(min_lng, max_lng, cols)]
    nodes = [(lat_values[row], lng_values[col]) for row in range(rows) for col in range(cols)]
    graph.add_nodes_from(nodes)
    for row in range(rows):
        for col in range(cols):
            node = (lat_values[row], lng_values[col])
            for next_row, next_col in [
                (row + 1, col),
                (row, col + 1),
                (row + 1, col + 1),
                (row + 1, col - 1),
            ]:
                if 0 <= next_row < rows and 0 <= next_col < cols:
                    graph.add_edge(node, (lat_values[next_row], lng_values[next_col]))
    return graph, {node: node for node in nodes}


def predict_grid_risk(lat: float, lng: float, df: pd.DataFrame, kmeans: KMeans) -> float:
    cluster = int(kmeans.predict([[lat, lng]])[0])
    cluster_density = float(df.loc[df["cluster"] == cluster, "cluster_density"].mean())
    nearest_distances = df.apply(lambda row: haversine_m(lat, lng, row["lat"], row["lng"]), axis=1).nsmallest(8)
    proximity = max(0.0, 1.0 - float(nearest_distances.mean()) / 2500)
    return round(float(np.clip(0.7 * cluster_density + 0.3 * proximity, 0, 1)), 4)


def nearest_node(graph: nx.Graph, point: tuple[float, float]) -> tuple[float, float]:
    return min(graph.nodes, key=lambda node: haversine_m(point[0], point[1], node[0], node[1]))


def compute_route(
    graph: nx.Graph,
    start,
    end,
    algorithm: str,
    objective: str,
    weight: str,
    node_risk: dict,
) -> tuple[RouteResult, list[tuple[float, float]]]:
    visited = 0

    def edge_weight(u, v, data):
        nonlocal visited
        visited += 1
        return data[weight]

    started = time.perf_counter()
    if algorithm == "astar":
        path = nx.astar_path(
            graph,
            start,
            end,
            heuristic=lambda a, b: haversine_m(a[0], a[1], b[0], b[1]),
            weight=edge_weight,
        )
    else:
        path = nx.dijkstra_path(graph, start, end, weight=edge_weight)
    elapsed_ms = (time.perf_counter() - started) * 1000

    distance = sum(haversine_m(a[0], a[1], b[0], b[1]) for a, b in zip(path, path[1:]))
    avg_risk = float(np.mean([node_risk[node] for node in path]))
    weighted_cost = sum(graph.edges[a, b]["safe_cost"] for a, b in zip(path, path[1:]))
    result = RouteResult(
        scenario="",
        algorithm=algorithm,
        objective=objective,
        distance_km=round(distance / 1000, 3),
        avg_risk=round(avg_risk, 4),
        risk_level=level_from_score(avg_risk),
        weighted_cost=round(float(weighted_cost), 3),
        elapsed_ms=round(elapsed_ms, 3),
        visited_nodes=visited,
        path_points=len(path),
    )
    return result, path


def plot_kmeans_metrics(metrics: pd.DataFrame, selected_k: int, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(metrics["k"], metrics["inertia"], marker="o", color="#1f77b4")
    axes[0].axvline(selected_k, color="#d62728", linestyle="--", linewidth=1)
    axes[0].set_title("Metodo del codo")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inercia")
    axes[1].plot(metrics["k"], metrics["silhouette"], marker="o", color="#2ca02c")
    axes[1].axvline(selected_k, color="#d62728", linestyle="--", linewidth=1)
    axes[1].set_title("Silhouette")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_cluster_map(df: pd.DataFrame, kmeans: KMeans, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        df["lng"],
        df["lat"],
        c=df["cluster"],
        s=12,
        cmap="tab10",
        alpha=0.72,
        linewidths=0,
    )
    ax.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], marker="X", s=130, c="black")
    ax.set_title("Clusters K-Means de zonas delictivas")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_risk_distribution(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df["risk_score_experimental"], bins=24, color="#4c78a8", edgecolor="white")
    axes[0].set_title("Distribucion de riesgo experimental")
    axes[0].set_xlabel("Riesgo")
    axes[0].set_ylabel("Registros")
    counts = df["risk_label"].value_counts().reindex(RISK_LEVELS)
    axes[1].bar(counts.index, counts.values, color=["#59a14f", "#f28e2b", "#e15759"])
    axes[1].set_title("Clases usadas por Random Forest")
    axes[1].set_xlabel("Nivel")
    axes[1].set_ylabel("Registros")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion(y_test: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    matrix = confusion_matrix(y_test, y_pred, labels=RISK_LEVELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(RISK_LEVELS)), RISK_LEVELS)
    ax.set_yticks(range(len(RISK_LEVELS)), RISK_LEVELS)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion Random Forest")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="#111")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(feature_importances: pd.DataFrame, output_path: Path) -> None:
    data = feature_importances.sort_values("importance").tail(10)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(data["feature"], data["importance"], color="#8c6d31")
    ax.set_title("Importancia de variables")
    ax.set_xlabel("Importancia")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_routes(df: pd.DataFrame, paths: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    sample = df.sample(min(len(df), 2500), random_state=RANDOM_STATE)
    ax.scatter(sample["lng"], sample["lat"], c=sample["risk_score_experimental"], cmap="YlOrRd", s=8, alpha=0.35)
    colors = {
        "dijkstra_distancia_minima": "#1f77b4",
        "dijkstra_riesgo_ponderado": "#2ca02c",
        "astar_riesgo_ponderado": "#9467bd",
    }
    for key, coords in paths.items():
        if not key.startswith("ruta_1"):
            continue
        suffix = key.replace("ruta_1_", "")
        lngs = [point["lng"] for point in coords]
        lats = [point["lat"] for point in coords]
        ax.plot(lngs, lats, linewidth=2.2, label=suffix, color=colors.get(suffix, None))
    ax.set_title("Comparacion de rutas sobre mapa de riesgo")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_route_metrics(route_metrics: list[RouteResult], output_path: Path) -> None:
    df = pd.DataFrame([asdict(result) for result in route_metrics])
    df = df[df["scenario"] == "ruta_1"].copy()
    df["label"] = df["algorithm"] + "\n" + df["objective"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(df["label"], df["distance_km"], color="#4c78a8")
    axes[0].set_title("Distancia")
    axes[0].set_ylabel("km")
    axes[1].bar(df["label"], df["avg_risk"], color="#e15759")
    axes[1].set_title("Riesgo promedio")
    axes[2].bar(df["label"], df["elapsed_ms"], color="#59a14f")
    axes[2].set_title("Tiempo")
    axes[2].set_ylabel("ms")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
