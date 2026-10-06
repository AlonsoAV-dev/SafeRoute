"""Conserva las referencias históricas tras los renombres documentados del código."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def _code_digest(path: Path) -> str:
    # Git puede convertir CRLF/LF; esa diferencia de plataforma no cambia el código.
    source = path.read_text(encoding='utf-8-sig')
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def huellas_compatibles(root: Path, paths) -> dict[str, str]:
    """Reconoce solo las versiones exactas autorizadas en el manifiesto de renombres.

    Los registros de experimentos siguen conservando sus rutas y hashes originales.
    Una edición posterior del código no obtiene esta equivalencia automáticamente.
    """
    migration_path = Path(__file__).with_name('renombres_experimentos.json')
    migration = json.loads(migration_path.read_text(encoding='utf-8'))
    if _code_digest(Path(__file__)) != migration['compatibilidad_sha256']:
        raise ValueError('Cambió el adaptador de compatibilidad; revise la versión del protocolo')
    approved = migration['archivos']
    result = {}
    for path in paths:
        path = Path(path).resolve()
        relative = path.relative_to(root.resolve())
        entry = approved.get(relative.as_posix())
        if entry is None:
            result[str(relative)] = _digest(path)
            continue
        if _code_digest(path) != entry['codigo_actual_sha256']:
            raise ValueError(f'Cambió {relative} después del renombre; use otra versión de protocolo')
        result[str(Path(entry['ruta_original']))] = entry['codigo_original_sha256']
    return result
