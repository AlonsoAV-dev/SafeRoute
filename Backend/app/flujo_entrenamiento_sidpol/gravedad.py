from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from app.services.pesos_delito import obtener_peso_desde_campos


def _ascii(value: str) -> str:
    text = unicodedata.normalize("NFKD", value.upper())
    return "".join(char for char in text if not unicodedata.combining(char))


# Aplica la configuración de gravedad por identificadores y palabras de la modalidad.
class CrimeWeights:
    def __init__(self, path: Path):
        self.path = path
        self.rules = json.loads(path.read_text(encoding="utf-8"))

    # Combina las reglas configuradas con la ponderación textual y conserva el mayor peso aplicable.
    def get(self, subtype_id: str, modality_id: str, subtype: str, modality: str) -> int:
        rules = self.rules
        weight = max(
            int(rules["peso_por_defecto"]),
            int(rules["por_subtipo"].get(subtype_id, 0)),
            int(rules["por_modalidad"].get(modality_id, 0)),
            obtener_peso_desde_campos(modality, subtype),
        )
        name = _ascii(modality)
        for phrase, severity in rules["palabras_modalidad"].items():
            if phrase in name:
                weight = max(weight, int(severity))
        return weight
