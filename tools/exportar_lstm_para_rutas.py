"""Exporta inferencia del checkpoint LSTM mensual existente para el servicio de rutas.

No reentrena ni calcula métricas nuevas. Conserva la procedencia y el mes de
entrenamiento del checkpoint; usa el historial mensual observado hasta agosto.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Backend/data"


class MonthlyLSTM(nn.Module):
    def __init__(self, channels: int, hidden: int, context: int):
        super().__init__()
        self.sequence = nn.LSTM(channels, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + context, 32), nn.ReLU(),
                                  nn.Dropout(.1), nn.Linear(32, 3))

    def forward(self, seq, context):
        _, (hidden, _) = self.sequence(seq)
        return self.head(torch.cat([hidden[-1], context], dim=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DATA / "experimentos_sidpol/buffer_300_mensual_tres_modelos_v1/modelo_lstm_2026-05.pt")
    parser.add_argument("--output", type=Path, default=DATA / "procesados_sidpol_v2")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = MonthlyLSTM(architecture["channels"], architecture["hidden"], architecture["context_features"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    torch.set_num_threads(8)
    sequence_dir = DATA / "experimentos_sidpol/alumbrado_fechas_originales_v1"
    sequence_meta = json.loads((sequence_dir / "secuencias_complete.json").read_text(encoding="utf-8"))
    periods = sequence_meta["fingerprint"]["periods"]
    temporal = np.load(sequence_dir / "secuencias_delitos_originales.npy", mmap_mode="r")
    roads_path = DATA / "procesados/evaluacion_2026/tramos_osm.csv"
    roads = pd.read_csv(roads_path, usecols=["tramo_id", "latitud", "longitud", "longitud_m"])
    if temporal.shape != (len(periods), len(roads), architecture["channels"]):
        raise ValueError("El historial y la red vial no corresponden a la arquitectura guardada")
    target = str(pd.Period(periods[-1], freq="M") + 1)
    scales = checkpoint["scales"]
    window = architecture["window"]
    probability = np.empty((len(roads), 3), dtype=np.float32)
    season = [np.sin(2*np.pi*(len(periods) % 12)/12), np.cos(2*np.pi*(len(periods) % 12)/12)]
    with torch.inference_mode():
        for left in range(0, len(roads), 8192):
            right = min(left + 8192, len(roads))
            seq = np.asarray(temporal[-window:, left:right], dtype=np.float32).transpose(1, 0, 2)
            seq = (seq - scales["mean"]) / scales["std"]
            static = roads.iloc[left:right]
            context = np.column_stack([static.latitud, static.longitud, np.log1p(static.longitud_m),
                                       np.broadcast_to(season, (right-left, 2))]).astype(np.float32)
            context = (context - scales["context_mean"]) / scales["context_std"]
            probability[left:right] = torch.softmax(model(torch.from_numpy(seq), torch.from_numpy(context)), dim=1).numpy()
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1, atol=1e-5):
        raise ValueError("La inferencia produjo probabilidades inválidas")
    result = roads[["tramo_id", "latitud", "longitud"]].copy()
    result["periodo_objetivo"] = target
    result["riesgo_score"] = .5 * probability[:, 1] + probability[:, 2]
    result["nivel_riesgo"] = np.asarray(["bajo", "medio", "alto"])[probability.argmax(axis=1)]
    result["modelo_usado"] = "LSTM"
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "predicciones_tramos_lstm.csv"
    temporary = path.with_suffix(".tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(path)
    metadata = {
        "modelo": "LSTM", "version_variables": "lstm_mensual_300m_originales_checkpoint_v1",
        "periodo_prediccion": target, "tramo_turno": False, "unidad": "segmento × mes",
        "entrenamiento_hasta": checkpoint["training"]["last_month"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "historial_hasta": periods[-1], "arquitectura": architecture,
        "origen_historial": str(sequence_dir.resolve()), "red_vial": str(roads_path.resolve()),
        "variables": [f"canal_{i}" for i in range(architecture["channels"])],
        "reentrenado": False, "evaluacion_nueva": False, "buffer_m": checkpoint["buffer_m"],
        "lighting": checkpoint["lighting"],
    }
    meta_path = args.output / "metadata_modelo_lstm.json"
    meta_tmp = meta_path.with_suffix(".tmp")
    meta_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_tmp.replace(meta_path)
    print(f"LSTM: {len(result):,} segmentos para {target}; checkpoint entrenado hasta {metadata['entrenamiento_hasta']}")
    print(path)


if __name__ == "__main__":
    main()
