# Backend SafeRoute

API FastAPI para calcular rutas en Lima considerando distancia y riesgo delictivo.

## Estructura

- `app/main.py`: endpoints existentes y carga del modelo.
- `app/services/preprocessing.py`: carga ligera usada por la API.
- `app/services/risk_model.py`: predicción de riesgo en línea.
- `app/services/routing.py`: grafo OSM, fallback local y A*.
- `app/flujo_entrenamiento/limpieza.py`: carga, normalización y validación.
- `app/flujo_entrenamiento/riesgo.py`: `riesgo_base`, recencia y riesgo espacial.
- `app/flujo_entrenamiento/modelos.py`: K-Means, Random Forest y XGBoost.
- `app/flujo_entrenamiento/ejecutar.py`: orquestación y persistencia.

## Instalar

```powershell
python -m pip install -r Backend/requirements.txt
```

## Entrenar o actualizar modelos

Ejecutar desde `Backend`:

```powershell
python -m app.flujo_entrenamiento.ejecutar
```

Para una prueba rápida:

```powershell
python -m app.flujo_entrenamiento.ejecutar --muestra 5000 --salida data/procesados_prueba
```

Cuando se agreguen registros al CSV, basta con ejecutar nuevamente el comando completo.
La fuente original nunca se modifica.

La guía detallada para repetir Random Forest, XGBoost y las corridas
experimentales está en `GUIA_ENTRENAMIENTO_Y_EXPERIMENTACION.md`.

## Artefactos generados

La carpeta `data/procesados` contiene:

- `delitos_original.csv`
- `delitos_limpios.csv`
- `delitos_con_riesgo.csv`
- `delitos_asignados_a_tramo.csv`
- `riesgo_por_tramo.csv`
- `clusters_riesgo.csv`
- `metricas_modelos.csv`
- `modelo_ganador.json`
- `modelo_ganador.joblib`
- `predicciones_riesgo.csv`
- `rutas_calculadas.csv`
- `resumen_flujo.json`

Los archivos son generados y están ignorados por Git.

## Selección del modelo

Random Forest y XGBoost usan la misma partición train/test y las mismas variables.
El ganador se elige, en este orden, por:

1. Recall de la clase `alto`.
2. F1 macro.
3. F1 macro promedio en validación cruzada.
4. Menor brecha entre train y test.

El `riesgo_score` no se usa como variable predictora porque define la clase objetivo y
su inclusión produciría fuga de información.

## Ejecutar API

Desde la raíz del proyecto:

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --app-dir Backend
```

Endpoints conservados:

- `GET /health`
- `GET /risk-zones`
- `GET /crime-points`
- `POST /route`
- `POST /api/route/calculate`
- `GET /api/heatmap`
- `GET /api/risk-zones`
- `GET /api/stats`
- `GET /api/crime-points`
- `GET /api/crime-filters`

Los filtros por día, turno, tipo o modalidad modifican únicamente las capas
visuales. El riesgo de la ruta siempre usa todo el historial disponible.

El costo de A* es:

```text
costo_tramo = distancia_metros * (1 + alpha * riesgo_normalizado)
```

Con `alpha = 0` se obtiene la ruta más rápida. Con valores mayores se penalizan
los tramos de mayor riesgo.
