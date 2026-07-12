# SafeRoute

Sistema de recomendacion de rutas vehiculares con menor exposicion estimada al
riesgo delictivo en Lima Metropolitana.

El backend asocia delitos georreferenciados con segmentos de OpenStreetMap,
construye un panel tramo-temporal y compara Random Forest con XGBoost. Las
probabilidades del modelo se convierten en un `riesgo_score` por segmento, que
se incorpora al costo de A* para comparar una ruta corta con una ruta segura.

K-Means no participa en el entrenamiento, la prediccion ni el ruteo.

## Estructura

```text
Backend/                  API FastAPI, entrenamiento y pruebas
Frontend/SafeRoute-app/   Interfaz React, Leaflet y mapas de riesgo
tools/                    Evaluacion temporal y graficos reproducibles
```

Los datasets, modelos entrenados y resultados pesados no se almacenan en Git.
Consulta `Backend/data/README.md` para conocer los archivos requeridos.

## Requisitos

- Python 3.11 o superior
- Node.js 20 o superior
- Acceso a internet en la primera carga de OpenStreetMap

## Backend

Desde la raiz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r Backend\requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --app-dir Backend
```

La documentacion interactiva queda disponible en `http://127.0.0.1:8000/docs`.

Endpoints principales:

- `GET /health`
- `GET /api/heatmap`
- `GET /api/prediction-heatmap`
- `GET /api/prediction-points`
- `POST /api/route/calculate`

## Frontend

```powershell
cd Frontend\SafeRoute-app
npm install
npm run dev
```

La interfaz se abre normalmente en `http://127.0.0.1:5173`.

## Entrenamiento y evaluacion

Entrenamiento base con ventana historica de tres meses:

```powershell
cd Backend
python -m app.flujo_entrenamiento.ejecutar --ventana-meses 3 --radios 100 150 200
```

Evaluacion temporal con 2025 como entrenamiento y enero-mayo de 2026 como
periodo externo:

```powershell
python tools\evaluar_modelos_2026.py
python tools\ajustar_modelos_2026.py
```

Los modelos utilizan 16 variables historicas y espaciales. La salida contiene
tres clases: `bajo`, `medio` y `alto`.

## Pruebas

```powershell
cd Backend
python -m unittest discover -s tests -v
cd ..
cd Frontend\SafeRoute-app
npm run lint
npm run build
```

## Datos y artefactos

Los archivos pesados se distribuyen fuera de GitHub mediante el repositorio de
artefactos de la investigacion. No se deben confirmar datasets originales,
paneles generados, modelos `.joblib`, redes `.graphml`, documentos Word ni
capturas de experimentacion.
