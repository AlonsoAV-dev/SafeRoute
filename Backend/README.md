# Backend SafeRoute

API FastAPI para consultar el riesgo delictivo por segmento vial y calcular
rutas con A* sobre la red de OpenStreetMap.

## Flujo activo

1. Limpia los delitos georreferenciados de 2025 y 2026.
2. Asocia los eventos con segmentos viales mediante un buffer metrico.
3. Construye ventanas historicas de tres meses con 16 variables.
4. Entrena y compara Random Forest y XGBoost.
5. Genera probabilidades para las clases `bajo`, `medio` y `alto`.
6. Calcula `riesgo_score = 0.5 * P(medio) + P(alto)`.
7. A* aplica `distancia_m * (1 + beta * riesgo_score)`.

K-Means no interviene en el flujo activo.

## Instalar y ejecutar

Desde la raiz del repositorio:

```powershell
python -m pip install -r Backend\requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --app-dir Backend
```

## Entrenar

```powershell
cd Backend
python -m app.flujo_entrenamiento.ejecutar --ventana-meses 3 --radios 100 150 200
```

Los datos requeridos y los artefactos generados se describen en
`Backend/data/README.md`.

## Pruebas

Desde la carpeta `Backend`:

```powershell
python -m unittest discover -s tests -v
```
