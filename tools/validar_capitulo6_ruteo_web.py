"""Pruebas reproducibles de API de ruteo con el modelo mensual snapping/NKDE."""
from pathlib import Path
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
import hashlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/resultados_capitulo6_snapping_nkde_2026_10_02/web'
BASE='http://127.0.0.1:8001'
CASES=[
 {'id':'OD1','name':'Avenida Tomás Marsano','origin':[-12.1082345,-77.0152248],'destination':[-12.117657,-77.0086386]},
 {'id':'OD2','name':'San Isidro a Chorrillos','origin':[-12.0977,-77.0355],'destination':[-12.1655,-77.0262]},
 {'id':'OD3','name':'Centro de Lima a San Borja','origin':[-12.0510,-77.0340],'destination':[-12.0840,-76.9890]},
]
def request(name,path,payload=None):
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();raw=json.dumps(payload).encode() if payload is not None else None
    req=urllib.request.Request(BASE+path,data=raw,headers={'Content-Type':'application/json'} if raw else {})
    try:
        with urllib.request.urlopen(req,timeout=180) as r:status,content=r.status,r.read()
    except urllib.error.HTTPError as e:status,content=e.code,e.read()
    value={'name':name,'recorded_at':datetime.now(timezone.utc).isoformat(),'endpoint':path,'request':payload,
           'status':status,'elapsed_seconds':time.perf_counter()-start,'response':json.loads(content)}
    (OUT/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(name,status,round(value['elapsed_seconds'],3),flush=True);return value
def save(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    request('salud','/health');models=request('modelos','/api/models');checks=[];route_rows=[];betas=[];surfaces={}
    for family in ('random_forest','xgboost','lstm'):
        value=request('superficie_'+family,'/api/risk-surface?bbox=-12.19,-77.06,-12.035,-76.965&zoom=14&modelo_riesgo='+family)
        assert value['status']==200;surfaces[family]=value['response']
        for case in CASES:
            payload={'origin':case['origin'],'destination':case['destination'],'routePreference':'safe','modelo_riesgo':family,
                     'beta':10,'buffer_m':200,'risk_mode':'predicted','turno':None}
            first=request(case['id']+'_'+family,'/api/route/calculate',payload)
            assert first['status']==200,first['response']
            repeated=request(case['id']+'_'+family+'_repetido','/api/route/calculate',payload)
            assert repeated['status']==200
            r=first['response'];a=r['safe_route'];b=r['traditional_route'];surface=surfaces[family]
            flags={'same_model_field':r['modelo_usado']==surface['model'],
                   'same_period':r['periodo_prediccion']==surface['prediction_period']=='2026-09',
                   'same_field_version':r['risk_field']['version']==surface['risk_field']['version'],
                   'within_detour_limit':a['distance_km']<=1.5*b['distance_km']+.002,
                   'segment_continuity':all(x['nodo_destino']==y['nodo_origen'] for x,y in zip(a['segments'],a['segments'][1:])),
                   'exposure_recomputed':abs(a['risk_total']-sum(x['distancia_metros']*x['riesgo_segmento_normalizado'] for x in a['segments'])/1000)<1e-4,
                   'repeat_same_route':a['route']==repeated['response']['safe_route']['route'],
                   'cache_hit_on_repeat':bool(repeated['response']['metrics']['cache_hit']),
                   'bounded_score':0<=a['risk_score']<=1 and 0<=b['risk_score']<=1}
            checks.append({'case':case['id'],'model':family,**flags,'all_passed':all(flags.values())})
            delta=(a['distance_km']/b['distance_km']-1)*100
            exposure_delta=(1-a['risk_total']/b['risk_total'])*100 if b['risk_total'] else None
            route_rows.append({'case':case['id'],'case_name':case['name'],'model':family,'period':r['periodo_prediccion'],
                   'reference':{k:b[k] for k in ['distance_km','time_min','risk_total','risk_score','high_risk_segments','red_distance_m','orange_distance_m']},
                   'safe':{k:a[k] for k in ['distance_km','time_min','risk_total','risk_score','high_risk_segments','red_distance_m','orange_distance_m']},
                   'delta_distance_pct':delta,'delta_time_pct':(a['time_min']/b['time_min']-1)*100,
                   'exposure_reduction_pct_unclipped':exposure_delta,'reported_reduction_pct':r['risk_reduction'],
                   'same_route':r['misma_ruta'],'selected_beta':r['parametros_a_star']['beta_ruta_segura'],
                   'strategy':r['estrategia_seleccionada'],'seconds_first':first['elapsed_seconds'],'seconds_repeat':repeated['elapsed_seconds'],
                   'first_cache_hit':r['metrics']['cache_hit'],'hotspot_policy':r['hotspot_policy'],
                   'diagnostics_graph':r['diagnostico_grafo']})
            for diag in r['diagnostico_beta']:
                betas.append({'case':case['id'],'model':family,**diag})
    base={'origin':CASES[0]['origin'],'destination':CASES[0]['destination'],'routePreference':'safe','modelo_riesgo':'xgboost',
          'beta':10,'buffer_m':200,'risk_mode':'predicted','turno':None}
    negatives=[
       ('sin_destino','/api/route/calculate',{'origin':base['origin']},422),
       ('coordenada_incompleta','/api/route/calculate',dict(base,origin=[-12.1]),422),
       ('modelo_invalido','/api/route/calculate',dict(base,modelo_riesgo='svm'),422),
       ('beta_fuera_rango','/api/route/calculate',dict(base,beta=21),422),
       ('beta_negativo','/api/route/calculate',dict(base,beta=-1),422),
       ('buffer_no_admitido','/api/route/calculate',dict(base,buffer_m=300),422),
       ('fuera_red_local','/api/route/calculate',dict(base,origin=[-16.3988,-71.5369]),400),
       ('latitud_fuera_rango','/api/route/calculate',dict(base,origin=[91,-77]),400),
       ('bbox_invalido','/api/risk-surface?bbox=1,2,3&zoom=14',None,400),
       ('zoom_fuera_rango','/api/risk-surface?bbox=-12.18,-77.04,-12.1,-77.01&zoom=21',None,422),
    ]
    errors=[]
    for name,path,payload,status in negatives:
        value=request(name,path,payload);errors.append({'case':name,'expected_status':status,'observed_status':value['status'],'passed':value['status']==status})
    for beta in (0,20):
        value=request('beta_valido_'+str(beta),'/api/route/calculate',dict(base,beta=beta))
        errors.append({'case':'beta_valido_'+str(beta),'expected_status':200,'observed_status':value['status'],'passed':value['status']==200})
    source=['Backend/app/main.py','Backend/app/services/routing.py','Backend/app/services/alternative_routes.py',
            'Backend/app/services/hotspot_routing.py','Backend/app/services/risk_surface.py','Backend/app/services/risk_model.py',
            'Frontend/SafeRoute-app/src/App.jsx','Frontend/SafeRoute-app/src/components/MapView.jsx']
    save('resumen_ruteo.json',{'cases':CASES,'routes':route_rows,'beta_diagnostics':betas})
    save('validacion_api.json',{'checks':checks,'error_tests':errors,'all_route_checks_passed':all(x['all_passed'] for x in checks),
              'all_validation_tests_passed':all(x['passed'] for x in errors),
              'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source}})
    print('TERMINADO',len(route_rows),'recorridos comparados',len(errors),'validaciones',flush=True)

if __name__=='__main__':main()
