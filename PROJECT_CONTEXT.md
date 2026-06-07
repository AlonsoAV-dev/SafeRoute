# Proyecto de Tesis - Sistema de Recomendacion de Rutas Seguras

#Correr el proyecto :
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --app-dir .\Backend

## Objetivo del Proyecto

Desarrollar un sistema de recomendacion de rutas seguras utilizando datos delictivos georreferenciados.

El sistema debe recomendar rutas entre un punto de origen y destino priorizando la seguridad sobre la distancia.

El proyecto esta orientado a una tesis universitaria, por lo que el enfoque debe ser:

- claro
- modular
- defendible academicamente
- sin sobreingenieria

﻿# Proyecto de Tesis - SafeRoute

## Ejecutar el proyecto

```powershell
\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --app-dir .\Backend

o

python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --app-dir .\Backend


```


## Objetivo

Desarrollar un sistema de recomendación de rutas seguras en Lima usando datos delictivos georreferenciados, ML (K-Means + Random Forest) y algoritmos de grafos (A* / Dijkstra).

Principios del proyecto:

- claro
- modular
- defendible académicamente
- sin sobreingeniería

## Stack tecnológico

- Frontend: React (Vite), Leaflet, OpenStreetMap
- Backend: FastAPI (Python)
- ML: scikit-learn (K-Means, Random Forest)
- Datos: CSV

## Dataset activo

Archivo principal:

- `Backend/data/AAV-DATASET.csv`

Características del CSV:

- Separador `;`
- Decimales con coma (por ejemplo: `-12,071`)
- Columnas de coordenadas: `lat_hecho`, `long_hecho` y/o `x`, `y`
- Turno: `turno_hecho`
- Tipo: `tipo_hecho`
- Subtipo: `subtipo_hecho`
- Distrito: `distrito_hecho`
- Fecha: `fecha_hora_hecho` y/o `fecha_hora_registro_hecho`

## Enfoque del sistema

1. Detectar zonas con alta concentración de delitos (K-Means).
2. Estimar riesgo por punto según turno y densidad (Random Forest + heurística).
3. Generar rutas que minimicen costo: distancia + riesgo ponderado.

## Backend (estado actual)

Ubicación: `Backend/app`

Módulos:

- `main.py`: endpoints FastAPI
- `schemas.py`: contratos de entrada y salida
- `services/preprocessing.py`: limpieza y validación del CSV
- `services/risk_model.py`: K-Means, Random Forest y scoring
- `services/routing.py`: rutas seguras con A* y grilla fallback

Endpoints actuales:

- `GET /health`
- `GET /risk-zones?turno=noche`
- `GET /crime-points?turno=noche`
- `POST /route`

## Frontend (estado actual)

Ubicación: `Frontend/SafeRoute-app`

La interfaz permite:

- ingresar origen y destino (búsqueda + selección en mapa)
- seleccionar turno
- ajustar peso de seguridad
- visualizar clusters de riesgo
- visualizar rutas segura y tradicional

## Sistema de diseño (objetivo)

### Paleta de colores (CSS variables obligatorias)

```css
:root {
  --bg-primary: #0f1117;
  --bg-secondary: #1a1d27;
  --bg-card: #1e2130;
  --bg-input: #252836;
  --accent-green: #22c55e;
  --accent-green-dark: #16a34a;
  --accent-green-glow: rgba(34, 197, 94, 0.15);
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --border-color: #2d3148;
  --risk-high: #ef4444;
  --risk-medium: #f59e0b;
  --risk-low: #22c55e;
  --sidebar-width: 220px;
  --panel-width: 300px;
}
```

### Tipografía

```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body {
  font-family: 'Inter', sans-serif;
  font-size: 14px;
  color: var(--text-primary);
  background: var(--bg-primary);
}
```

### Layout principal (4 zonas)

- Sidebar: 220px
- Panel izquierdo: 300px
- Mapa central: flex: 1
- Panel derecho: 280px

## Reglas de consistencia visual

- Botón primario: `--accent-green` con hover `--accent-green-dark`
- Inputs: `--bg-input`, borde `--border-color`
- Cards: `--bg-card`, borde sutil y glow verde opcional
- Riesgo alto: `--risk-high`
- Riesgo bajo: `--risk-low`
- Fuente Inter en todo el sistema
│   SIDEBAR   │   PANEL IZQUIERDO│           MAPA CENTRAL            │ PANEL DERECHO │
│  (220px)    │    (300px)       │         (flex: 1)                 │   (280px)     │
│  nav links  │  form de ruta    │      Leaflet Y OPEN STREET MAP             │  resultados   │
└─────────────┴──────────────────┴──────────────────────────────────┴───────────────┘
Todo el layout en display: flex; height: 100vh; overflow: hidden.

🗂️ COMPONENTES — ESPECIFICACIONES DETALLADAS

1. SIDEBAR (izquierda, 220px)

Fondo: var(--bg-secondary)
Borde derecho: 1px solid var(--border-color)
Logo SafeRoute:

Ícono: escudo con letra "K" en verde (var(--accent-green))
Título: "SafeRoute" en blanco, 16px, bold
Subtítulo: "Rutas óptimas y seguras" en var(--text-muted), 11px


Menú de navegación con íconos Lucide React:

Home → Inicio (activo)
MapPin → Nueva ruta
Clock → Historial
AlertTriangle → Zonas de riesgo
BarChart2 → Estadísticas
Settings → Configuración
Info → Acerca de


Item activo: fondo var(--accent-green), texto blanco, border-radius 8px, padding 10px 16px
Items inactivos: texto var(--text-secondary), hover suave con fondo rgba(255,255,255,0.05)
Card inferior (esquina abajo):

Texto: "Tu seguridad es nuestra prioridad"
Ícono grande de escudo verde luminoso
Fondo degradado sutil verde oscuro




2. PANEL IZQUIERDO — Formulario de ruta (300px)

Fondo: var(--bg-secondary)
Título: "Nueva ruta", 18px, 600 weight
Campos del formulario:

jsx// Punto de origen
<div className="input-field">
  <span className="dot green" /> {/* círculo verde */}
  <input placeholder="Av. Arequipa 1234, Lima" />
  <X size={14} /> {/* botón limpiar */}
</div>

// Punto de destino
<div className="input-field">
  <span className="dot red" /> {/* círculo rojo */}
  <input placeholder="Universidad de Lima" />
  <X size={14} />
</div>

Estilo inputs: background: var(--bg-input), borde 1px solid var(--border-color), border-radius 8px, padding 10px 14px
Fecha y hora: 2 inputs en row (calendario + reloj), mismo estilo
Preferencia de ruta: 2 botones toggle:

"Ruta más segura" (activo): borde verde, fondo rgba(34,197,94,0.1), ícono Shield
"Ruta más rápida": fondo var(--bg-input), ícono Clock


Slider — Peso de seguridad (α):

Range input 0.0–1.0
Track: degradado de gris a verde
Thumb: verde var(--accent-green), circular
Labels: "Priorizar rapidez" ←→ "Priorizar seguridad"
Valor actual mostrado en badge verde (ej. "0.7")


Botón "Buscar ruta segura":

Ancho 100%, height 44px
Background: var(--accent-green)
Hover: var(--accent-green-dark) con sombra 0 0 20px rgba(34,197,94,0.3)
Border-radius: 8px, font-weight: 600
Ícono Shield a la izquierda del texto


Niveles de riesgo (leyenda inferior):

Título "Niveles de riesgo"
3 filas: ● Alto riesgo 0.7–1.0 (rojo) / ● Medio riesgo 0.3–0.7 (amarillo) / ● Bajo riesgo 0.0–0.3 (verde)




3. MAPA CENTRAL
Librería: Leaflet Y OPEN STREET MAP
Configuración del mapa:

Tiles: CartoDB Dark Matter (mapa oscuro sin texto excesivo)

  https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png

Centro inicial: Lima, Perú [-12.0464, -77.0428], zoom 13
Sin controles de zoom visibles (o en esquina discreta)

Heatmap de riesgo:

Usar leaflet.heat o @asymmetrik/ngx-leaflet-markercluster
Los puntos del heatmap deben tener:

Gradiente: { 0.0: '#22c55e', 0.3: '#84cc16', 0.5: '#f59e0b', 0.7: '#f97316', 1.0: '#ef4444' }
Radio: 35px, blur: 25, maxZoom: 16
Opacidad: 0.65 (semitransparente para ver el mapa debajo)


Las zonas rojas deben verse claramente en Jesús María, Lince, La Victoria
Las zonas verdes en los bordes (Miraflores, San Borja, Surco)

Rutas trazadas:
js// Ruta segura (verde)
L.polyline(safeRouteCoords, {
  color: '#22c55e',
  weight: 5,
  opacity: 0.9,
  dashArray: null,
  lineCap: 'round',
  lineJoin: 'round'
})

// Ruta tradicional (gris)
L.polyline(traditionalRouteCoords, {
  color: '#64748b',
  weight: 4,
  opacity: 0.7,
  dashArray: '8, 6'
})
Marcadores:

Origen: marcador pin verde personalizado (SVG)
Destino: marcador pin rojo personalizado (SVG)

Barra de búsqueda superior del mapa:

Input flotante sobre el mapa, centrado arriba
Fondo blanco/claro, sombra, border-radius 24px
Ícono lupa + placeholder "Buscar lugar..."
Botón "X" para limpiar

Popup de leyenda del mapa (flotante inferior izquierdo):
┌─────────────────────────────────┐
│  Mapa de riesgo (predicción)    │
│  [████ degradado verde→rojo ███]│
│  Bajo                      Alto │
│  Predicción: Random Forest      │
└─────────────────────────────────┘

Fondo var(--bg-card), border-radius 12px, padding 16px, sombra suave

Toggle dark/light mode (esquina superior derecha del mapa):

Botón con íconos Sol/Luna, fondo oscuro redondeado


4. PANEL DERECHO — Resultados (280px)

Fondo: var(--bg-secondary), borde izquierdo 1px solid var(--border-color)
Card "Ruta recomendada":

Header con ícono Shield verde y título
Ruta segura (recomendada):

Indicador: línea verde gruesa ━━━
Métricas en grid 2x2:

Distancia: 8.42 km
Tiempo estimado: 23 min
Riesgo promedio: 0.28 en verde + badge "Bajo"




Ruta tradicional (más corta):

Indicador: línea gris punteada ╌╌╌
Mismas métricas: 6.71 km / 18 min / 0.72 (Alto) en rojo


Card de reducción de riesgo:

Fondo con borde verde sutil + glow verde
Ícono Shield
Porcentaje grande: 61% en verde, 32px, bold
Texto: "comparado con la ruta tradicional"


Botón "Ver indicaciones paso a paso":

Fondo transparente, borde 1px solid var(--border-color)
Texto blanco, ícono ChevronRight
Hover: borde verde






5. BARRA INFERIOR — Métricas del sistema (3 cards)
Fila horizontal debajo del mapa, fondo var(--bg-secondary):
┌──────────────────┬──────────────────┬──────────────────┐
│ 🧠 Predicción    │ 🔵 Zonas         │ ✦  Algoritmo     │
│    de riesgo     │    detectadas    │    de ruta       │
│ Modelo Random    │ Clustering K-    │ A* (costo =      │
│ Forest           │ Means            │ distancia +      │
│ Precisión: 87.3% │ Actualizado:     │ α*riesgo)        │
│                  │ 24/05/2025       │ Tiempo: 320 ms   │
└──────────────────┴──────────────────┴──────────────────┘

Cada card: var(--bg-card), borde var(--border-color), border-radius 12px, padding 16px
Ícono en color acento, texto principal blanco, subtexto gris


⚙️ BACKEND — FastAPI
Endpoints principales:
pythonPOST /api/route/calculate
# Body: { origin: [lat, lng], destination: [lat, lng], alpha: float, datetime: str }
# Response: { safe_route: [...coords], traditional_route: [...coords], metrics: {...} }

GET /api/heatmap
# Response: { points: [[lat, lng, intensity], ...] }

GET /api/risk-zones
# Response: { zones: [{ center: [lat, lng], radius: float, risk_level: str }] }

GET /api/stats
# Response: { model_accuracy: float, zones_count: int, calc_time_ms: int }
Función de costo del algoritmo A*:
pythondef cost(distance_km: float, risk_score: float, alpha: float) -> float:
    return (1 - alpha) * distance_km + alpha * risk_score * distance_km
Modelo ML:
python# Pipeline de predicción de riesgo
features = ['lat', 'lng', 'hour', 'day_of_week', 'month', 'crime_type_encoded']
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Clustering de zonas
kmeans = KMeans(n_clusters=8, random_state=42)
zones = kmeans.fit_predict(crime_data[['lat', 'lng']])

🎯 REGLAS ABSOLUTAS DE CONSISTENCIA VISUAL

Todo botón primario → background: var(--accent-green), mismo border-radius (8px), mismo padding (10px 20px)
Todo input/select → background: var(--bg-input), border: 1px solid var(--border-color), color: var(--text-primary), border-radius 8px
Todo modal/dropdown → background: var(--bg-card), mismo borde y sombra box-shadow: 0 8px 32px rgba(0,0,0,0.4)
Texto de riesgo ALTO siempre en var(--risk-high) (#ef4444)
Texto de riesgo BAJO siempre en var(--risk-low) (#22c55e)
Fuente Inter en absolutamente todo — sin excepciones
Iconos: exclusivamente Lucide React, tamaño base 16px
Scrollbars personalizados: thin, color verde, fondo oscuro
Transiciones: transition: all 0.2s ease en todos los elementos interactivos
Sin bordes redondeados excesivos — máximo 12px en cards, 8px en botones/inputs, 50% solo en avatares/dots


📁 ESTRUCTURA DE ARCHIVOS SUGERIDA
saferoute/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Sidebar.jsx
│   │   │   ├── RouteForm.jsx 
│   │   │   ├── MapView.jsx          ← Leaflet + heatmap + rutas + OPEN STREET MAP
│   │   │   ├── ResultsPanel.jsx
│   │   │   ├── BottomMetrics.jsx
│   │   │   └── RiskLegend.jsx
│   │   ├── styles/
│   │   │   └── global.css           ← CSS variables + reset
│   │   └── App.jsx
├── backend/
│   ├── main.py                      ← FastAPI app
│   ├── routes/
│   │   ├── route_calculator.py      ← A* algorithm
│   │   └── risk_predictor.py        ← RF model
│   └── models/
│       ├── kmeans_model.pkl
│       └── random_forest_model.pkl