# Datos requeridos

Esta carpeta se mantiene fuera de Git porque contiene datasets y artefactos que
superan el tamano apropiado para un repositorio de codigo.

Antes de ejecutar el entrenamiento o iniciar la API, colocar aqui:

```text
Backend/data/
  DELITOS TOTAL.csv
  DELITOS TOTAL-2026.csv
  red_vial_lima.graphml          # opcional; se genera o descarga si no existe
  procesados/                    # se genera durante el entrenamiento
```

Archivos de entrada:

- `DELITOS TOTAL.csv`: registros completos de 2025.
- `DELITOS TOTAL-2026.csv`: registros de enero a mayo de 2026 usados para la
  evaluacion temporal externa.

La carpeta `procesados/` contiene paneles tramo-temporales, predicciones,
metricas, modelos y graficos. Todos esos archivos son regenerables mediante los
scripts del sistema y deben compartirse en el repositorio externo de artefactos,
no en GitHub.

