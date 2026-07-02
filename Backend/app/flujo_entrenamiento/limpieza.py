from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


LIMITES_LIMA = {
    "lat_min": -12.60,
    "lat_max": -11.45,
    "lng_min": -77.30,
    "lng_max": -76.60,
}

MESES_ES = {
    "enero": "01",
    "febrero": "02",
    "marzo": "03",
    "abril": "04",
    "mayo": "05",
    "junio": "06",
    "julio": "07",
    "agosto": "08",
    "septiembre": "09",
    "setiembre": "09",
    "octubre": "10",
    "noviembre": "11",
    "diciembre": "12",
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}

NOMBRES_MESES_ES = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}


def cargar_fuente(ruta: Path) -> pd.DataFrame:
    """Carga CSV o Excel sin modificar el archivo original."""
    extension = ruta.suffix.lower()
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(ruta)
    if extension == ".csv":
        with ruta.open("r", encoding="utf-8-sig") as archivo:
            primera_linea = archivo.readline()
        separador = ";" if primera_linea.count(";") > primera_linea.count(",") else ","
        return pd.read_csv(ruta, sep=separador, encoding="utf-8-sig", low_memory=False)
    raise ValueError(f"Formato no soportado: {extension}. Use CSV, XLSX o XLS.")


def limpiar_delitos(datos: pd.DataFrame) -> pd.DataFrame:
    """Normaliza DELITOS TOTAL y descarta registros no utilizables."""
    limpios = pd.DataFrame(index=datos.index)
    limpios["id_hecho"] = _identificador(datos)
    limpios["latitud"] = _numero(
        _primera_columna(datos, ["lat_hecho", "y", "latitud", "lat"])
    )
    limpios["longitud"] = _numero(
        _primera_columna(datos, ["long_hecho", "x", "longitud", "lng", "lon"])
    )
    limpios["distrito"] = _texto(
        _primera_columna(datos, ["distrito_hecho", "distrito"])
    )
    limpios["provincia"] = _texto(
        _primera_columna(datos, ["provincia_hecho", "provincia"])
    )
    limpios["departamento"] = _texto(
        _primera_columna(datos, ["departamento_hecho", "departamento"])
    )
    limpios["tipo_delito"] = _texto(
        _primera_columna(datos, ["tipo_hecho", "tipo_delito", "tipo"])
    )
    limpios["subtipo_delito"] = _texto(
        _primera_columna(datos, ["subtipo_hecho", "subtipo_delito", "subtipo"])
    )
    limpios["modalidad"] = _texto(
        _primera_columna(datos, ["modalidad_hecho", "modalidad"])
    )
    limpios["turno"] = _turno(_primera_columna(datos, ["turno_hecho", "turno"]))
    limpios["fecha"] = _fecha(datos)
    limpios["hora"] = _hora(datos, limpios["turno"])
    limpios["anio"] = limpios["fecha"].dt.year
    limpios["mes"] = limpios["fecha"].dt.month
    limpios["mes_nombre"] = limpios["mes"].map(NOMBRES_MESES_ES)
    limpios["dia"] = limpios["fecha"].dt.day
    limpios["periodo"] = limpios["fecha"].dt.to_period("M").astype(str)
    limpios["dia_semana"] = limpios["fecha"].dt.day_name().map(
        {
            "Monday": "lunes",
            "Tuesday": "martes",
            "Wednesday": "miercoles",
            "Thursday": "jueves",
            "Friday": "viernes",
            "Saturday": "sabado",
            "Sunday": "domingo",
        }
    )

    limpios = limpios.dropna(subset=["latitud", "longitud", "fecha"])
    dentro_area = (
        limpios["latitud"].between(LIMITES_LIMA["lat_min"], LIMITES_LIMA["lat_max"])
        & limpios["longitud"].between(LIMITES_LIMA["lng_min"], LIMITES_LIMA["lng_max"])
    )
    limpios = limpios.loc[dentro_area].copy()
    limpios = limpios.drop_duplicates(subset=["id_hecho"], keep="first")
    return limpios.sort_values("fecha").reset_index(drop=True)


def _identificador(datos: pd.DataFrame) -> pd.Series:
    candidatos = ["GlobalID", "globalid", "id_dgc", "ID_DGC_03", "OBJECTID"]
    resultado = pd.Series("", index=datos.index, dtype="object")
    for columna in candidatos:
        if columna not in datos.columns:
            continue
        valores = datos[columna].fillna("").astype(str).str.strip()
        resultado = resultado.where(resultado.ne(""), valores)
    faltantes = resultado.eq("")
    resultado.loc[faltantes] = "FILA-" + resultado.index[faltantes].astype(str)
    return resultado


def _primera_columna(datos: pd.DataFrame, nombres: list[str]) -> pd.Series:
    for nombre in nombres:
        if nombre in datos.columns:
            return datos[nombre].fillna("")
    return pd.Series("", index=datos.index, dtype="object")


def _numero(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(
        serie.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def _normalizar_texto(valor: object) -> str:
    texto = str(valor or "").strip().upper()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", texto) or "NO ESPECIFICADO"


def _texto(serie: pd.Series) -> pd.Series:
    return serie.map(_normalizar_texto)


def _turno(serie: pd.Series) -> pd.Series:
    normalizada = _texto(serie).str.lower()
    return normalizada.where(
        normalizada.isin({"manana", "tarde", "noche", "madrugada"}),
        "noche",
    )


def _fecha(datos: pd.DataFrame) -> pd.Series:
    year_column = next(
        (column for column in ("año_hecho", "anio_hecho", "aÃ±o_hecho") if column in datos.columns),
        None,
    )
    if year_column and {"mes_hecho", "dia_hecho"}.issubset(datos.columns):
        componentes = pd.DataFrame(
            {
                "year": pd.to_numeric(
                    datos[year_column].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                ),
                "month": pd.to_numeric(datos["mes_hecho"], errors="coerce"),
                "day": pd.to_numeric(datos["dia_hecho"], errors="coerce"),
            }
        )
        fecha = pd.to_datetime(componentes, errors="coerce")
    else:
        fecha = pd.Series(pd.NaT, index=datos.index, dtype="datetime64[ns]")

    fuente = _primera_columna(
        datos, ["fecha_hora_hecho", "fecha_hora_registro_hecho", "fecha"]
    ).astype(str)
    for nombre, numero in MESES_ES.items():
        fuente = fuente.str.replace(nombre, numero, case=False, regex=False)
    alternativa = pd.to_datetime(fuente, errors="coerce", dayfirst=False)
    return fecha.fillna(alternativa)


def _hora(datos: pd.DataFrame, turnos: pd.Series) -> pd.Series:
    fuente = _primera_columna(datos, ["hora_hecho", "hora"]).astype(str)
    hora = pd.to_numeric(fuente.str.extract(r"(\d{1,2})", expand=False), errors="coerce")
    hora_turno = turnos.map({"madrugada": 2, "manana": 9, "tarde": 15, "noche": 21})
    return hora.fillna(hora_turno).clip(0, 23).astype(int)
