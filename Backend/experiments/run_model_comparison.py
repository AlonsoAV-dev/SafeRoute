from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.flujo_entrenamiento.ejecutar import ejecutar_flujo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara Random Forest y XGBoost en una corrida experimental fechada."
    )
    parser.add_argument(
        "--fuente",
        type=Path,
        default=BACKEND_DIR / "data" / "AAV-DATASET.csv",
    )
    parser.add_argument("--muestra", type=int, default=0)
    parser.add_argument("--tamano-segmento-m", type=int, default=200)
    parser.add_argument(
        "--salida-raiz",
        type=Path,
        default=BACKEND_DIR / "experiments" / "outputs",
    )
    args = parser.parse_args()
    output = args.salida_raiz / datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = ejecutar_flujo(
        fuente=args.fuente,
        salida=output,
        muestra=args.muestra,
        tamano_segmento_m=args.tamano_segmento_m,
    )
    print(json.dumps({"salida": str(output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

