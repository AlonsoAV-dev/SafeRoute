from __future__ import annotations

import unicodedata


PESO_DELITO_DEFAULT = 3

PESOS_DELITO = {
    "SICARIATO": 5,
    "FEMINICIDIO": 5,
    "HOMICIDIO CALIFICADO - ASESINATO": 5,
    "PARRICIDIO": 5,
    "HOMICIDIO SIMPLE": 5,
    "HOMICIDIO POR PAF": 5,
    "HOMICIDIO POR EMOCION VIOLENTA": 5,
    "ROBO AGRAVADO A MANO ARMADA": 5,
    "ROBO AGRAVADO EN BANDA": 5,
    "ROBO AGRAVADO DURANTE LA NOCHE O EN LUGAR DESOLADO": 5,
    "ROBO AGRAVADO": 5,
    "ROBO DE CELULAR": 3,
    "ROBO DE PASAPORTE": 3,
    "ROBO": 3,
    "ROBO FRUSTRADO": 2,
    "HURTO AGRAVADO EN CASA HABITADA": 3,
    "HURTO AGRAVADO DURANTE LA NOCHE": 3,
    "HURTO AGRAVADO": 3,
    "HURTO DE VEHICULO": 2,
    "HURTO DE CELULAR": 1,
    "HURTO DE ACCESORIOS Y AUTOPARTES DE VEHICULOS": 1,
    "HURTO": 1,
    "HURTO DE USO": 1,
    "HURTO FRUSTRADO": 1,
}


def normalizar_modalidad(valor: str | None) -> str:
    texto = str(valor or "").strip().upper()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return " ".join(texto.split())


def obtener_peso_delito(modalidad: str | None) -> int:
    return PESOS_DELITO.get(normalizar_modalidad(modalidad), PESO_DELITO_DEFAULT)
