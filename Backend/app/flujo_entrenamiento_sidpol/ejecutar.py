"""Ejecute: python -m app.flujo_entrenamiento_sidpol.ejecutar --config ..."""

from __future__ import annotations

import argparse
import json

from app.flujo_entrenamiento_sidpol.config import DEFAULT_CONFIG, TrainingConfig
from app.flujo_entrenamiento_sidpol.fuente import preparar_fuente
from app.flujo_entrenamiento_sidpol.modelado import entrenar_y_exportar
from app.flujo_entrenamiento_sidpol.segmentacion import asociar_delitos, cargar_matrices
from app.flujo_entrenamiento_sidpol.variables import FeatureBuilder, eligible_months


# Orquesta la lectura de la fuente, la asociación vial y la exportación de los modelos.
def ejecutar(config: TrainingConfig) -> dict:
    config.output.mkdir(parents=True, exist_ok=True)
    complete = config.output / "entrenamiento_completo.json"
    complete.unlink(missing_ok=True)
    normalized = config.output / "delitos_geolocalizados.csv"
    source_audit_path = config.output / "auditoria_fuente.json"
    reuse_source = normalized.exists() and source_audit_path.exists()
    if reuse_source:
        audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
        reuse_source = (
            audit.get("source") == str(config.source)
            and audit.get("weights_file") == str(config.weights_file)
            and normalized.stat().st_mtime_ns >= config.source.stat().st_mtime_ns
            and normalized.stat().st_mtime_ns >= config.weights_file.stat().st_mtime_ns
        )
    if reuse_source:
        source_audit = audit
        print("Fuente normalizada reutilizada desde caché", flush=True)
    else:
        source_audit = preparar_fuente(config.source, normalized, source_audit_path, config.weights_file)
    splits = eligible_months(config, source_audit)
    print("Meses aptos por corte:", {name: len(months) for name, months in splits.items()}, flush=True)
    tramos, spatial_audit = asociar_delitos(config, normalized)
    builder = FeatureBuilder(tramos, cargar_matrices(config.output), config)
    return entrenar_y_exportar(config, builder, splits, source_audit, spatial_audit)


# Recibe la configuración desde la línea de comandos e inicia el pipeline SIDPOL.
def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena riesgo por tramo y turno con SIDPOL 2018–2026")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    arguments = parser.parse_args()
    from pathlib import Path

    config = TrainingConfig.load(Path(arguments.config))
    summary = ejecutar(config)
    print(json.dumps({
        "seleccionado": summary["selected_for_routing"],
        "prediccion": summary["forecast_period"],
        "prueba": summary["assessments"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
