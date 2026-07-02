# Backend SafeRoute

API FastAPI que predice el riesgo delictivo futuro por tramo vial y calcula rutas
con OpenStreetMap y A*.

## Metodología activa

La unidad de análisis es `tramo OSM + ventana temporal`. El flujo:

1. Lee `data/DELITOS TOTAL.csv`.
2. Limpia coordenadas, fechas, categorías y duplicados por `GlobalID`.
3. Asigna pesos de gravedad entre 1 y 5.
4. Agrega a cada tramo los delitos cercanos mediante buffers de 100, 150 y 200 metros.
5. Excluye meses incompletos y crea ventanas históricas de tres meses con 16 variables.
6. Entrena un Random Forest por radio y selecciona el de mayor F1 macro temporal.
7. Convierte las probabilidades en `riesgo_score = 0.5 * P(medio) + P(alto)`.
8. A* usa `distancia_m * (1 + beta * riesgo_score)`.

La exposición acumulada de una ruta se calcula como la suma de
`distancia_m * riesgo_score`; así no depende de cuántas aristas pequeñas contiene.

Por defecto, el riesgo usado por A* es únicamente la predicción futura de Random Forest.
También se puede elegir riesgo histórico o una combinación de 70% histórico y 30% RF
para comparar resultados. Ambos componentes se devuelven por tramo para auditoría.

KMeans no interviene en el entrenamiento, la predicción ni el ruteo.

## Entrenar

Desde `Backend`:

```powershell
python -m pip install -r requirements.txt
python -m app.flujo_entrenamiento.ejecutar
```

La primera ejecución descarga y guarda `data/red_vial_lima.graphml`. Las siguientes
reutilizan esa red. Cuando se agreguen meses a `DELITOS TOTAL.csv`, basta con ejecutar
el mismo comando.

Opciones principales:

```powershell
python -m app.flujo_entrenamiento.ejecutar --ventana-meses 3 --radios 100 150 200
```

## Artefactos

`data/procesados` contiene:

- `delitos_limpios.csv`
- `tramos_osm.csv`
- `datos_random_forest.csv` (entrenamiento y prueba temporal)
- `datos_prediccion_futura.csv` (entrada para el siguiente mes)
- `comparacion_radios.csv`
- `modelo_random_forest.joblib`
- `predicciones_tramos.csv`
- `metricas_random_forest.csv`
- `classification_report_random_forest.csv`
- `matriz_confusion_random_forest.csv`
- `metadata_modelo.json`
- `resumen_flujo.json`
- `graficos/01_metricas_random_forest.png`
- `graficos/02_matriz_confusion.png`
- `graficos/03_metricas_por_clase.png`
- `graficos/04_distribucion_predicciones.png`
- `graficos/05_importancia_variables.png`
- `graficos/06_comparacion_radios.png`

Los gráficos se regeneran automáticamente al entrenar. También pueden crearse de nuevo
sin reentrenar:

```powershell
python -m app.flujo_entrenamiento.graficos
```

## Ejecutar API

Desde la raíz del proyecto:

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --app-dir Backend
```

Endpoints principales:

- `GET /health`
- `POST /api/route/calculate`
- `GET /api/heatmap`
- `GET /api/prediction-heatmap`
- `GET /api/prediction-points`
- `GET /api/stats`
- `GET /api/crime-points`
- `GET /api/crime-filters`

Los filtros del mapa solo afectan las capas históricas. El criterio de ruteo se selecciona
con `risk_mode`: `predicted` (predeterminado), `historical` o `hybrid`. Las capas RF
muestran las predicciones por tramo del periodo indicado en `metadata_modelo.json`.

## Pruebas

```powershell
python -m unittest discover -s tests -v
```
