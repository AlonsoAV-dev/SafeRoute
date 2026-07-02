# Guía de entrenamiento temporal

## Fuente

El archivo esperado es `Backend/data/DELITOS TOTAL.csv`, con una fila por delito.
Debe conservar el separador `;`, las columnas originales y `GlobalID` para eliminar
duplicados.

## Unidad de análisis

Random Forest no recibe delitos individuales. Recibe un panel donde:

```text
1 fila = 1 tramo vial OSM + 1 ventana histórica
```

Con una ventana de tres meses:

| Variables históricas | Objetivo |
|---|---|
| enero-marzo | riesgo observado en abril |
| febrero-abril | riesgo observado en mayo |
| marzo-mayo | riesgo observado en junio |

El último periodo se reserva para prueba. No se realiza una división aleatoria porque
permitiría que el entrenamiento utilizara información temporal posterior.

Los nombres de meses en español o inglés se normalizan a un número y a un nombre en
español. La fecha procesada se conserva en formato ISO `AAAA-MM-DD`.

## Agregación espacial

Cada tramo recibe la señal de los delitos ubicados dentro de un buffer alrededor de su
geometría. Se comparan radios de 100, 150 y 200 metros. La gravedad disminuye mediante
decaimiento gaussiano conforme aumenta la distancia entre el delito y el tramo.

Un delito puede influir en varios tramos vecinos porque todos ellos están expuestos al
entorno del hecho, pero solo se cuenta una vez dentro de cada tramo. Todos los tramos
permanecen en el panel, incluidos los que no presentan delitos cercanos.

El radio se selecciona únicamente con el periodo de prueba: mayor F1 macro y, en caso
de empate, mayor recall y PR-AUC para riesgo Alto.

## Variables

Las 16 entradas se agrupan en ubicación y longitud del tramo; frecuencia, gravedad y
delitos graves de la ventana histórica; conteos por tipo de delito; conteos por turno;
y densidades de frecuencia y gravedad por 100 metros. No se usan identificadores,
geometría, mes objetivo ni la variable objetivo como predictores.

La variable continua futura es:

```text
riesgo_bruto_futuro = suma_pesos_mes_siguiente / max(longitud_m / 100, 1)
```

Los ceros forman la clase `bajo`. Los valores positivos se dividen entre `medio` y
`alto` usando el percentil 75 calculado solo con el entrenamiento.

## Score para A*

Random Forest devuelve probabilidades de las tres clases:

```text
riesgo_score = 0.0 * P(bajo) + 0.5 * P(medio) + 1.0 * P(alto)
```

El costo de la ruta segura es:

```text
costo_tramo = distancia_m * (1 + beta * riesgo_score)
```

En producción, el modo predeterminado usa directamente el `riesgo_score` futuro de
Random Forest. El sistema permite dos comparaciones opcionales: riesgo histórico del
buffer de 200 metros y modo combinado (70% histórico + 30% Random Forest). El modo usado
queda registrado en la respuesta de la ruta y ambos componentes se devuelven por tramo.

## Ejecución

Desde `Backend`:

```powershell
python -m app.flujo_entrenamiento.ejecutar
```

Los meses que no llegan a su último día calendario se excluyen automáticamente. Con la
fuente actual se usan los doce meses completos de 2025, diciembre se reserva para prueba
y se generan predicciones para enero de 2026.

## Interpretación preliminar

La accuracy puede ser alta porque la mayoría de tramos no registra delitos. Para evaluar
el modelo deben priorizarse `balanced_accuracy`, `f1_macro`, `recall_riesgo_alto` y
`pr_auc_riesgo_alto`. Los resultados actuales validan el funcionamiento técnico; deben
recalcularse cuando la serie incluya más meses.

Para evitar que millones de ceros dominen el ajuste, Random Forest conserva todos los
ejemplos `medio/alto` y usa una muestra reproducible de la clase `bajo`. El mes de prueba
no se submuestrea: las métricas se calculan sobre todos sus tramos.
