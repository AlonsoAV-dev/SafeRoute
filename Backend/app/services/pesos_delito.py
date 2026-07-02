from __future__ import annotations

import unicodedata


PESO_DELITO_DEFAULT = 1

PESOS_DELITO = {
    "SICARIATO": 5,
    "FEMINICIDIO": 5,
    "HOMICIDIO CALIFICADO - ASESINATO": 5,
    "PARRICIDIO": 5,
    "HOMICIDIO SIMPLE": 5,
    "HOMICIDIO POR PAF": 5,
    "HOMICIDIO POR EMOCION VIOLENTA": 5,
    "SECUESTRO": 5,
    "ROBO AGRAVADO A MANO ARMADA": 4,
    "ROBO AGRAVADO EN BANDA": 4,
    "ROBO AGRAVADO DURANTE LA NOCHE O EN LUGAR DESOLADO": 4,
    "ROBO AGRAVADO": 4,
    "EXTORSION AGRAVADA": 4,
    "EXTORSION": 4,
    "ROBO DE CELULAR": 3,
    "ROBO DE PASAPORTE": 3,
    "ROBO": 3,
    "ROBO FRUSTRADO": 3,
    "ASALTO Y ROBO DE VEHICULOS": 3,
    "HURTO AGRAVADO EN CASA HABITADA": 2,
    "HURTO AGRAVADO DURANTE LA NOCHE": 2,
    "HURTO AGRAVADO": 2,
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
    normalizada = normalizar_modalidad(modalidad)
    if normalizada in PESOS_DELITO:
        return PESOS_DELITO[normalizada]
    if "HOMICIDIO" in normalizada or "ASESINATO" in normalizada:
        return 5
    if "SECUESTRO" in normalizada:
        return 5
    if "EXTORSION" in normalizada or "ROBO AGRAVADO" in normalizada:
        return 4
    if "ROBO" in normalizada:
        return 3
    if "HURTO AGRAVADO" in normalizada:
        return 2
    if "HURTO" in normalizada:
        return 1
    return PESO_DELITO_DEFAULT


def obtener_peso_desde_campos(modalidad: str | None, subtipo: str | None) -> int:
    """Prioriza la modalidad y usa el subtipo cuando aquella no es reconocible."""
    modalidad_normalizada = normalizar_modalidad(modalidad)
    if modalidad_normalizada in PESOS_DELITO or any(
        token in modalidad_normalizada
        for token in ("HURTO", "ROBO", "EXTORSION", "HOMICIDIO", "SECUESTRO")
    ):
        return obtener_peso_delito(modalidad_normalizada)
    return obtener_peso_delito(subtipo)
