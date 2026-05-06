# SafeRoute Backend

Backend FastAPI para recomendar rutas seguras en Lima con datos delictivos y OpenStreetMap.

## Stack

- FastAPI
- scikit-learn (K-Means + Random Forest)
- osmnx + networkx (grafo OSM + A*)

## Ejecución local

```zsh
python3 -m pip install -r /Users/alonso/Tesis-System/SafeRoute/Backend/requirements.txt
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir /Users/alonso/Tesis-System/SafeRoute/Backend
```

## Endpoints

### `GET /health`

Respuesta:

```json
{
  "status": "ok",
  "records": 12345,
  "dataset": "DB-1JAN-28MARCH.csv"
}
```

### `GET /risk-zones?turno=noche`

Retorna clusters K-Means y su nivel de riesgo para el turno.

### `GET /crime-points?turno=noche`

Retorna puntos delictivos por turno (opcional).

### `POST /route`

Solicitud:

```json
{
  "origin": { "lat": -12.0464, "lng": -77.0428 },
  "destination": { "lat": -12.0905, "lng": -77.0068 },
  "turno": "noche",
  "safety_weight": 4
}
```

Respuesta:

```json
{
  "route": [{"lat": -12.0464, "lng": -77.0428}],
  "distance_km": 4.2,
  "risk_score": 0.41,
  "risk_level": "medio",
  "turno": "noche",
  "zones_considered": []
}
```

## Flujo de recomendación

1. Recibe origen, destino y turno del usuario.
2. Descarga un grafo vial de OpenStreetMap con `osmnx` (drive).
3. Calcula riesgo por nodo con Random Forest + heurística del turno.
4. Asigna costo por tramo: $costo = distancia + (\alpha \cdot riesgo)$.
5. Ejecuta A* con `networkx`.
6. Retorna la ruta con riesgo y distancia.

> Nota: si la descarga OSM falla, el backend usa una grilla local como fallback.

## Comunicación con el frontend

- El frontend envía `origin`, `destination` y `turno`.
- El turno recomendado se puede calcular en el frontend según la hora local del usuario.
- La búsqueda por dirección se resuelve en el frontend usando Nominatim y luego se envían coordenadas al backend.
