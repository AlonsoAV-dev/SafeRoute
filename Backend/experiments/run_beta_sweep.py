from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.preprocessing import load_crime_records  # noqa: E402
from app.services.risk_model import RiskModel  # noqa: E402
from app.services.routing import generate_route_comparison  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara beta, buffer y modelo para una pareja origen-destino."
    )
    parser.add_argument("--origin-lat", type=float, required=True)
    parser.add_argument("--origin-lng", type=float, required=True)
    parser.add_argument("--destination-lat", type=float, required=True)
    parser.add_argument("--destination-lng", type=float, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=BACKEND_DIR
        / "experiments"
        / "outputs"
        / f"beta_sweep_{datetime.now():%Y%m%d_%H%M%S}.json",
    )
    args = parser.parse_args()

    records = load_crime_records(BACKEND_DIR / "data" / "AAV-DATASET.csv")
    risk_model = RiskModel(
        records,
        model_dir=BACKEND_DIR / "data" / "procesados",
    )
    results = []
    for model in ("auto", "random_forest", "xgboost"):
        for buffer_m in (50, 100, 150):
            comparison = generate_route_comparison(
                origin=(args.origin_lat, args.origin_lng),
                destination=(args.destination_lat, args.destination_lng),
                risk_model=risk_model,
                modelo_riesgo=model,
                beta=10,
                buffer_m=buffer_m,
            )
            results.append(
                {
                    "modelo_solicitado": model,
                    "modelo_usado": comparison["modelo_usado"],
                    "buffer_m": buffer_m,
                    "ruta_rapida": comparison["ruta_rapida"],
                    "ruta_segura": comparison["ruta_segura"],
                    "reduccion_riesgo": comparison["reduccion_riesgo"],
                    "diagnostico_beta": comparison["diagnostico_beta"],
                    "diagnostico_grafo": comparison["diagnostico_grafo"],
                    "mensaje": comparison["mensaje"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
