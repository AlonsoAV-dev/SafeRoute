# Entrenamiento y experimentación

## 1. Preparar el entorno

Desde la raíz del proyecto:

```powershell
python -m pip install -r Backend/requirements.txt
```

El archivo fuente esperado por defecto es:

```text
Backend/data/AAV-DATASET.csv
```

El archivo original se lee, pero nunca se sobrescribe.

## 2. Entrenar modelos para la API

Ejecutar desde `Backend`:

```powershell
python -m app.flujo_entrenamiento.ejecutar
```

Este comando:

1. Limpia y valida coordenadas, fechas y campos categóricos.
2. Asigna `peso_delito` desde `modalidad`.
3. Agrupa el historial en unidades espaciales de 200 metros.
4. Calcula frecuencia y gravedad promedio.
5. Calcula `riesgo_score`.
6. Ejecuta K-Means.
7. Divide una sola vez los mismos datos en train/test para ambos modelos.
8. Entrena Random Forest y XGBoost con las mismas variables.
9. Evalúa ambos modelos y selecciona el ganador.
10. Guarda el modelo y las predicciones en `data/procesados`.

Prueba rápida:

```powershell
python -m app.flujo_entrenamiento.ejecutar --muestra 6000 --salida data/procesados_prueba
```

Cambiar el tamaño espacial:

```powershell
python -m app.flujo_entrenamiento.ejecutar --tamano-segmento-m 100
```

## 3. Ejecutar una experimentación sin reemplazar el modelo activo

Desde `Backend`:

```powershell
python experiments/run_model_comparison.py
```

Cada corrida crea una carpeta fechada en:

```text
Backend/experiments/outputs/AAAAMMDD_HHMMSS
```

Ejemplo con muestra:

```powershell
python experiments/run_model_comparison.py --muestra 10000
```

Ejemplo para comparar otra resolución espacial:

```powershell
python experiments/run_model_comparison.py --tamano-segmento-m 100
python experiments/run_model_comparison.py --tamano-segmento-m 300
```

## 4. Variables usadas por los modelos

Variables numéricas:

- latitud y longitud
- frecuencia de delitos
- suma y promedio de `peso_delito`
- peso promedio y peso máximo
- cantidad de delitos graves
- cantidades históricas por turno
- cluster K-Means

Variable categórica:

- distrito, codificado con One-Hot Encoding

El día de la semana no se incluye en el score ni en las variables predictoras.
Solo se conserva para filtros, gráficos y análisis exploratorio.

## 5. Fórmula de riesgo

```text
riesgo_score =
    0.30 * frecuencia_normalizada
    + 0.70 * (peso_delito_promedio / 10)
```

La gravedad tiene mayor peso para permitir que una zona con pocos homicidios pueda
ser más riesgosa que una zona con numerosos hurtos menores.

La escala vigente de `peso_delito` es de 1 a 5.

## 6. División y evaluación

- Test: 25%
- Train: 75%
- Semilla: 42
- División estratificada
- Validación cruzada: 5 folds estratificados

Métricas guardadas:

- accuracy
- precision macro
- recall macro
- F1 macro
- recall de riesgo alto
- matriz de confusión
- F1 promedio y desviación en validación cruzada
- accuracy train
- brecha train-test

Orden de selección:

1. Mayor recall de riesgo alto.
2. Mayor F1 macro.
3. Mayor F1 macro de validación cruzada.
4. Menor brecha train-test.

## 7. Archivos clave para revisar

- `metricas_modelos.csv`: comparación RF/XGBoost.
- `modelo_ganador.json`: ganador y justificación.
- `riesgo_por_tramo.csv`: frecuencia, gravedad y score por unidad espacial.
- `clusters_riesgo.csv`: interpretación de clusters.
- `predicciones_riesgo.csv`: riesgo predicho usado por la API.
- `resumen_flujo.json`: resumen de la corrida.

## 8. Activar el nuevo modelo

Una experimentación fechada no reemplaza el modelo activo. Para activar un
entrenamiento nuevo se debe ejecutar:

```powershell
python -m app.flujo_entrenamiento.ejecutar
```

Después se reinicia FastAPI para que cargue `data/procesados`.

## 9. Experimentar con A*, beta y buffers

```powershell
python experiments/run_beta_sweep.py `
  --origin-lat -12.071 `
  --origin-lng -77.068 `
  --destination-lat -12.085 `
  --destination-lng -77.035
```

El experimento compara:

- `beta`: 1, 3, 5, 10, 15 y 20.
- buffer: 50, 100 y 150 metros.
- modelo: automático, Random Forest y XGBoost.

El costo rápido es:

```text
costo_rapido = distancia_m
```

El costo seguro es:

```text
costo_seguro = distancia_m * (1 + beta * riesgo_segmento_normalizado)
```

Cada arista recibe cantidad de delitos cercanos, peso acumulado, riesgo bruto,
riesgo normalizado, tiempo y nivel de riesgo. La normalización aplica clipping
con percentil 95 para evitar que pocos valores extremos dominen todos los costos.

## 10. Advertencia metodológica

La clase `nivel_riesgo` se construye a partir del `riesgo_score` calculado con
frecuencia y gravedad. Por eso, una métrica muy alta significa que el modelo
aprende correctamente ese criterio diseñado.

No significa por sí sola que el modelo haya sido validado contra una etiqueta
externa u oficial de peligrosidad. Para una validación más fuerte se necesitaría
una variable objetivo independiente, por ejemplo incidentes futuros por tramo,
evaluación policial o una ventana temporal posterior que no participe en el
entrenamiento.
