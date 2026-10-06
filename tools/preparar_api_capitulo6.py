from pathlib import Path
import json
import csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/resultados_capitulo6_snapping_nkde_2026_10_02'
DEPLOY = ROOT / 'Backend/data/experimentos_sidpol/snapping_nkde_mensual_2026_10_02/predicciones_web'
result = json.loads((OUT / 'resultados_finales.json').read_text(encoding='utf-8'))
(DEPLOY / 'entrenamiento_completo.json').write_text(json.dumps({
    'selected_for_routing': result['selected_primary_B'],
    'forecast_period': '2026-09',
    'experiment': 'snapping150_nkde250_original_monthly',
}, indent=2), encoding='utf-8')
for family, data in result['results'].items():
    m = data['metrics']
    row = dict(modelo=family, periodo_prueba='2025-01 a 2026-08',
               accuracy=m['accuracy'], balanced_accuracy=m['balanced_accuracy'],
               f1_macro=m['f1_macro'], recall_riesgo_alto=m['recall'][2])
    with (DEPLOY / f'metricas_{family}.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(family, m['accuracy'], m['balanced_accuracy'], m['recall'][2], m['precision'][2])
