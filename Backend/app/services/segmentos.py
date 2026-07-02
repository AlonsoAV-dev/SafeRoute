from __future__ import annotations


def construir_id_segmento(u: object, v: object, key: object, data: dict) -> str:
    """Crea un identificador estable y compartido por ambos sentidos de una vía."""
    extremos = sorted((str(u), str(v)))
    osmid = data.get("osmid", key)
    if isinstance(osmid, (list, tuple, set)):
        osmid = "-".join(sorted(str(value) for value in osmid))
    return f"OSM-{extremos[0]}-{extremos[1]}-{osmid}"
