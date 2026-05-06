# Proyecto de Tesis - Sistema de Recomendacion de Rutas Seguras

## Objetivo del Proyecto

Desarrollar un sistema de recomendacion de rutas seguras utilizando datos delictivos georreferenciados.

El sistema debe recomendar rutas entre un punto de origen y destino priorizando la seguridad sobre la distancia.

El proyecto esta orientado a una tesis universitaria, por lo que el enfoque debe ser:

- claro
- modular
- defendible academicamente
- sin sobreingenieria

## Stack Tecnologico

### Frontend

- React
- Leaflet

### Backend

- Python
- FastAPI

### Machine Learning

- scikit-learn

### Mapas y rutas

- OpenStreetMap
- osmnx
- networkx

## Contexto del Dataset

El dataset contiene registros de delitos georreferenciados ocurridos en Lima.

Cada fila representa un hecho delictivo.

Columnas importantes:

- `lat_hecho`: latitud del delito
- `long_hecho`: longitud del delito
- `turno_hecho`: manana, tarde, noche o madrugada
- `tipo_hecho`: categoria general del delito
- `subtipo_hecho`: detalle del delito
- `distrito_hecho`: distrito donde ocurrio
- `fecha_hora_hecho`: fecha del delito
- `ESTADO_COORD`: estado de la coordenada
- `OBSERVACION`: observaciones sobre georreferenciacion

Algunos registros contienen:

- `ESTADO_COORD = SIN COORDENADA`
- `OBSERVACION = GEO FORZADA AL CENTROIDE DE UBIC COMISARIA`

Estos datos deben filtrarse o tratarse como coordenadas imprecisas, y mencionarse como limitacion del proyecto.

## Enfoque del Proyecto

El proyecto no busca hacer analisis criminologico complejo.

La idea principal es:

1. Detectar zonas con alta concentracion de delitos.
2. Identificar patrones simples segun horario.
3. Generar rutas evitando zonas peligrosas.

## Metodologia General

### 1. Preprocesamiento

- limpiar dataset
- eliminar datos sin coordenadas validas
- normalizar variables
- trabajar principalmente con coordenadas, turno y tipo de delito

### 2. K-Means

Usar `lat_hecho` y `long_hecho` para detectar clusters de concentracion delictiva.

Cada cluster puede clasificarse como:

- riesgo bajo
- riesgo medio
- riesgo alto

### 3. Random Forest Simple

El modelo no debe ser complejo. Se utiliza para aprender una clasificacion basica de riesgo segun:

- ubicacion
- cluster
- turno

### 4. Generacion de Rutas

Usar `networkx` y A* para calcular rutas con un costo ponderado:

- distancia
- riesgo del punto o tramo
- turno seleccionado

La prioridad del sistema es seguridad sobre distancia. `osmnx` queda como componente para integrar la red vial real de OpenStreetMap cuando se trabaje con el dataset final y el area exacta de estudio.

## MVP Implementado

### Backend

Ubicacion: `Backend/app`

Modulos:

- `main.py`: endpoints FastAPI
- `schemas.py`: contratos de entrada y salida
- `services/preprocessing.py`: limpieza y validacion de registros
- `services/risk_model.py`: K-Means, Random Forest simple y puntaje de riesgo
- `services/routing.py`: ruta segura con A* y peso de seguridad

Endpoints:

- `GET /health`
- `GET /risk-zones?turno=noche`
- `POST /route`

Dataset principal:

- `Backend/data/DB-1JAN-28MARCH.csv`

Dataset de respaldo para pruebas pequenas:

- `Backend/data/sample_crimes.csv`

### Frontend

Ubicacion: `Frontend/SafeRoute-app`

La interfaz permite:

- ingresar origen y destino
- seleccionar turno
- ajustar peso de seguridad
- visualizar clusters de riesgo
- visualizar la ruta recomendada en Leaflet

## Prioridades

1. Que funcione correctamente.
2. Que sea entendible.
3. Que sea defendible en tesis.
4. Luego optimizar detalles.

## Restricciones

- Mantener arquitectura simple.
- Evitar sobreingenieria.
- Explicar el codigo claramente.
- Mantener modularidad.
- Priorizar claridad sobre complejidad.
