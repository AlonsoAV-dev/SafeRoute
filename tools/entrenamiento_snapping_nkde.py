"""Comparación real de tres modelos: snapping 150 m, NKDE 250 m, panel mensual.

Ejecutar .venv/Scripts/python tools/entrenamiento_snapping_nkde.py
Se conserva el protocolo A/B/C y el criterio ya definido, con salidas independientes.
"""
from pathlib import Path
import sys
import json
import gc
import hashlib
import time
from datetime import datetime, timezone
import random
import numpy as np
import pandas as pd
from scipy import sparse
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from buffer300.modelos import fit_model, predictions, matrix_metrics, model_parameters
from modelos_mensuales import RF_PARAMETERS, MonthlyLSTM
from probar_alumbrado_fecha_xgb_lstm import normalize_training
from compatibilidad_experimentos import huellas_compatibles

OUT=ROOT/'outputs/resultados_capitulo6_snapping_nkde_2026_10_02'
DATA=ROOT/'Backend/data/experimentos_sidpol/snapping_nkde_mensual_2026_10_02'
SOURCE=ROOT/'Backend/data/procesados_sidpol_v2'
KERNEL=ROOT/'Backend/data/experimentos_sidpol/nkde_v1/features/kernel_250m.npz'
PERIODS=tuple(f'{y}-{m:02}' for y in range(2018,2027) for m in range(1,13))[:104]
B=tuple(f'{y}-{m:02}' for y in range(2022,2025) for m in (3,6,9,12))
C=tuple(p for p in PERIODS if p>='2025-01')
FAMILIES=('xgboost','random_forest','lstm')
WINDOWS=('3m','12m','desde2018')
THRESHOLD=3.
SPEC={'features':'contexto','balance':1.25}
CAP=90000
EPOCHS=4

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(6)
def score(m):return .4*m['accuracy']+.6*m['f1_medio_alto']
# OBJETIVO: Bajo si R=0, Medio si 0<R<3 y Alto si R>=3; NKDE entra como predictor.
def labels(r):return np.where(r>=THRESHOLD,2,np.where(r>0,1,0)).astype(np.int8)

# MÉTRICAS: compara las tres clases y calcula PR-AUC a partir de sus probabilidades.
def metrics(y,p):
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
    cm=np.bincount(y.astype(np.int64)*3+p.argmax(1),minlength=9).reshape(3,3)
    m=matrix_metrics(cm);m['balanced_accuracy']=float(np.mean(m['recall']))
    m['pr_auc_ap']=[float(average_precision_score(y==i,p[:,i])) for i in range(3)]
    m['pr_auc_macro']=float(np.mean(m['pr_auc_ap']))
    m['majority_baseline_accuracy']=float(np.bincount(y,minlength=3).max()/len(y))
    return m

# PANEL: utiliza delitos ya asignados a un solo tramo por snapping con límite de 150 m.
# Agrega los turnos por mes y calcula densidades sobre la red vial.
def prepare():
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    paths=[SOURCE/'delitos_geolocalizados.csv',SOURCE/'asignacion_espacial.json',SOURCE/'auditoria_fuente.json',KERNEL,Path(__file__)]
    # Conserva la identificación del panel existente tras el renombre documentado.
    fingerprint=huellas_compatibles(ROOT,paths)
    marker=DATA/'panel_complete.json'
    if marker.exists():
        assert read(marker)['sha256']==fingerprint,'Cambió el código o la fuente; use una versión nueva'
        return
    tramos=pd.read_csv(SOURCE/'tramos_osm.csv')
    expected=pd.read_csv(KERNEL.parent.parent.parent/'escenario_distrito/procesados/tramos_osm.csv',usecols=['tramo_id'])
    assert tramos.tramo_id.equals(expected.tramo_id)
    tramos.to_csv(DATA/'tramos_osm.csv',index=False)
    n=len(tramos)
    panel=np.lib.format.open_memmap(DATA/'panel_mensual_snapping.npy',mode='w+',dtype='float32',shape=(104,n,11))
    mapping={'count':0,'weight':1,'grave':2,'hurto':3,'robo':4,'extorsion':5,'homicidio':6}
    for name,col in mapping.items():
        # Suma los cuatro turnos: una observación corresponde a segmento vial-mes.
        m=sparse.load_npz(SOURCE/f'matriz_{name}.npz')
        for month in range(104):panel[month,:,col]=np.asarray(m[month*4:month*4+4].sum(0)).ravel()
        if name=='count':
            for month in range(104):
                for turn,colturn in enumerate((10,7,8,9)):panel[month,:,colturn]=m.getrow(month*4+turn).toarray().ravel()
        del m
    panel.flush()
    assert int(panel[:,:,0].sum())==read(SOURCE/'asignacion_espacial.json')['total_asignados']
    kernel=sparse.load_npz(KERNEL)
    density=np.lib.format.open_memmap(DATA/'nkde_mensual.npy',mode='w+',dtype='float32',shape=(104,n,2))
    temporal=np.lib.format.open_memmap(DATA/'secuencias.npy',mode='w+',dtype='float16',shape=(104,n,14))
    factor=np.maximum(tramos.longitud_m.to_numpy(dtype=np.float32)/100,1)
    dist=[]
    for month,period in enumerate(PERIODS):
        # NKDE de 250 m: propaga frecuencia y gravedad según distancias de la red.
        density[month,:,0]=kernel@panel[month,:,0]
        density[month,:,1]=kernel@panel[month,:,1]
        # LSTM recibe 11 canales delictivos, 2 canales NKDE y 1 indicador de historial.
        temporal[month,:,:11]=np.log1p(panel[month]/factor[:,None]).astype(np.float16)
        temporal[month,:,11:13]=np.log1p(density[month]).astype(np.float16)
        temporal[month,:,13]=1
        yy=labels(panel[month,:,1]/factor)
        dist.append({'periodo':period,'clases':np.bincount(yy,minlength=3).tolist(),
                     'delitos_asignados':int(panel[month,:,0].sum())})
        if month%12==11:print('PANEL',period,flush=True)
    density.flush();temporal.flush()
    save(OUT/'procesamiento_panel.json',{'source_audit':read(SOURCE/'auditoria_fuente.json'),
          'snapping_audit':read(SOURCE/'asignacion_espacial.json'),'filter_audit':read(ROOT/'outputs/delitos-filtrados/auditoria_filtrado.json'),
          'months':104,'segments':n,'panel_rows':104*n,'channels':11,'monthly':dist,
          'class_counts':np.sum([r['clases'] for r in dist],axis=0).tolist(),
          'labels':{'low':'R=0','medium':'0<R<3','high':'R>=3','R':'suma de pesos del mes / max(longitud_m/100,1)'},
          'nkde_input':'frecuencia y suma de pesos originales; kernel triangular de 250 m',
          'observed_counts_note':'Las etiquetas representan delitos registrados y georreferenciados. No representan riesgo real exhaustivo.'})
    save(marker,{'sha256':fingerprint,'complete':True,'created_at':now()})
    del panel,density,temporal,kernel;gc.collect()

class Data:
    def __init__(self):
        self.panel=np.load(DATA/'panel_mensual_snapping.npy',mmap_mode='r')
        self.temporal=np.load(DATA/'secuencias.npy',mmap_mode='r')
        self.density=np.load(DATA/'nkde_mensual.npy',mmap_mode='r')
        roads=pd.read_csv(DATA/'tramos_osm.csv');self.static=roads[['latitud','longitud','longitud_m']].to_numpy(dtype=np.float32)
        self.n=len(roads);self.factor=np.maximum(self.static[:,2]/100,1)
        self.risk=self.panel[:,:,1]/self.factor
    def target(self,month):return labels(self.risk[month])
    # RF Y XGBOOST: 74 variables tabulares, incluidos seis resúmenes históricos NKDE.
    def features(self,month,ids):
        assert month>=3
        f=self.factor[ids];cols=[self.static[ids,0],self.static[ids,1],np.log1p(self.static[ids,2]),np.full(len(ids),min(month,12))]
        for lag in (1,2,3):
            values=self.panel[month-lag,ids]/f[:,None];cols.extend(values[:,i] for i in range(11))
        for w in (3,6,12):
            r=self.risk[max(0,month-w):month,ids];cnt=self.panel[max(0,month-w):month,ids,0]
            cols.extend([cnt.mean(0)/f,r.mean(0),r.std(0),r.max(0),(r>0).mean(0),(r>=THRESHOLD).mean(0)])
        cols.extend([self.risk[month-6,ids] if month>=6 else np.zeros(len(ids)),
                     self.risk[month-12,ids] if month>=12 else np.zeros(len(ids)),
                     self.risk[month-1,ids]-self.risk[month-2,ids],
                     self.risk[month-3:month,ids].mean(0)-self.risk[max(0,month-6):month-3,ids].mean(0) if month>3 else np.zeros(len(ids)),
                     (self.risk[month-1,ids]+.05)/(self.risk[max(0,month-6):month,ids].mean(0)+.05),
                     np.full(len(ids),np.sin(2*np.pi*(month%12)/12)),np.full(len(ids),np.cos(2*np.pi*(month%12)/12))])
        for channel in (0,1):
            # Usa NKDE del mes anterior y sus promedios de 3 y 6 meses; nunca el mes objetivo.
            cols.extend([self.density[month-1,ids,channel],self.density[max(0,month-3):month,ids,channel].mean(0),
                         self.density[max(0,month-6):month,ids,channel].mean(0)])
        for lag in (1,2,3):
            r=self.risk[month-lag];cols.extend([np.full(len(ids),(r>0).mean()),np.full(len(ids),(r>=THRESHOLD).mean())])
        result=np.column_stack(cols).astype(np.float32);assert result.shape[1]==74 and np.isfinite(result).all()
        return result
    # LSTM: secuencia de 12 meses anteriores, con 14 variables por mes, incluida NKDE.
    def sequence(self,month,ids):
        start=max(0,month-12);s=np.zeros((len(ids),12,14),dtype=np.float32)
        known=np.asarray(self.temporal[start:month,ids],dtype=np.float32).transpose(1,0,2);s[:,12-known.shape[1]:]=known;return s
    # CONTEXTO LSTM: latitud, longitud, log(longitud vial+1) y seno/coseno del mes.
    def context(self,month,ids):
        return np.column_stack([self.static[ids,:2],np.log1p(self.static[ids,2]),
                               np.full(len(ids),np.sin(2*np.pi*(month%12)/12)),np.full(len(ids),np.cos(2*np.pi*(month%12)/12))]).astype(np.float32)
    # MUESTREO: toma solo meses anteriores y limita a 90.000 ejemplos, con pesos por clase.
    def select(self,month,window):
        first=3 if window=='desde2018' else max(3,month-int(window[:-1]))
        periods=list(range(first,month));limit=min(18000,CAP//(3*len(periods)));rng=np.random.default_rng(42)
        entries=[];ys=[];exp=[];counts=np.zeros(3,dtype=np.int64);h=hashlib.sha256()
        for m in periods:
            yy=self.target(m);counts+=np.bincount(yy,minlength=3);ids=[];rat=[]
            for c in range(3):
                eligible=np.flatnonzero(yy==c);keep=min(limit,len(eligible))
                if keep:ids.append(rng.choice(eligible,keep,replace=False));rat.append(np.full(keep,len(eligible)/keep,dtype=np.float32))
            ids=np.concatenate(ids);order=np.argsort(ids);ids=ids[order];entries.append((m,ids));ys.append(yy[ids]);exp.append(np.concatenate(rat)[order])
            h.update(np.asarray(m,dtype=np.int32).tobytes());h.update(ids.astype(np.int32).tobytes())
        y=np.concatenate(ys).astype(np.int64);w=np.concatenate(exp)*(counts.sum()/(3*counts[y]))**1.25;w=(w/w.mean()).astype(np.float32)
        return entries,y,w,{'first_target':PERIODS[first],'last_target':PERIODS[month-1],
                            'target_months':len(periods),'train_units':len(periods)*self.n,'train_samples':len(y),
                            'class_counts':counts.tolist(),'sample_sha256':h.hexdigest(),'per_class_month_cap':limit}
    def tabular(self,entries):return np.concatenate([self.features(m,ids) for m,ids in entries])

# SALIDA LSTM: softmax convierte los tres puntajes en P(Bajo), P(Medio) y P(Alto).
def predict_lstm(model,data,month,scales):
    p=np.empty((data.n,3),dtype=np.float32);model.eval()
    with torch.inference_mode():
        for left in range(0,data.n,8192):
            ids=np.arange(left,min(left+8192,data.n));seq=(data.sequence(month,ids)-scales['mean'])/scales['std']
            ctx=(data.context(month,ids)-scales['context_mean'])/scales['context_std']
            p[ids]=torch.softmax(model(torch.from_numpy(seq),torch.from_numpy(ctx)),1).numpy()
    return p

# COMPARACIÓN: los tres modelos comparten segmentos, etiquetas y meses de evaluación.
def run_origin(data,phase,period,window,families,epochs=4):
    month=PERIODS.index(period);missing=[f for f in families if not (OUT/phase/f'{f}_{window}_{period}.json').exists()]
    if not missing:return
    entries,y,w,training=data.select(month,window)
    x=None;current=None
    for family in missing:
        seed();start=time.perf_counter();history=[]
        print('AJUSTE',phase,period,window,family,'filas',len(y),flush=True)
        if family=='lstm':
            # LSTM aprende la evolución mensual de los delitos y la densidad NKDE.
            seq=np.concatenate([data.sequence(m,ids) for m,ids in entries]);ctx=np.concatenate([data.context(m,ids) for m,ids in entries])
            # Ajusta la normalización solo con los ejemplos de entrenamiento.
            scales=normalize_training(seq,ctx)
            loader=DataLoader(TensorDataset(torch.from_numpy(seq),torch.from_numpy(ctx),torch.from_numpy(y),torch.from_numpy(w)),batch_size=2048,shuffle=True,num_workers=0)
            # PyTorch: LSTM de 32 unidades; AdamW con tasa 0,001 y lotes de 2.048 ejemplos.
            model=MonthlyLSTM();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
            for epoch in range(1,epochs+1):
                model.train();losses=[]
                for a,b,t,ww in loader:
                    # Minimiza entropía cruzada ponderada y limita el gradiente a 1.
                    optimizer.zero_grad(set_to_none=True);loss=(F.cross_entropy(model(a,b),t,reduction='none')*ww).sum()/ww.sum()
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
                row={'epoch':epoch,'loss':float(np.mean(losses))}
                if phase=='B':row['metrics']=metrics(data.target(month),predict_lstm(model,data,month,scales))
                history.append(row);print('EPOCA',period,window,epoch,'loss',round(row['loss'],4),flush=True)
            p=predict_lstm(model,data,month,scales)
            if phase=='C':torch.save({'state_dict':model.state_dict(),'scales':scales,'training':training,'epochs':epochs},DATA/f'modelo_lstm_{period}.pt')
            del seq,ctx,loader,optimizer
        else:
            if x is None:x=data.tabular(entries);current=data.features(month,np.arange(data.n))
            if family=='xgboost':
                # XGBOOST: 320 árboles sucesivos, profundidad 5 y tasa 0,055; entrada tabular con NKDE.
                model=fit_model(SPEC,x,y,w)
            else:
                # RANDOM FOREST: combina 200 árboles de profundidad máxima 16; misma entrada con NKDE.
                model=RandomForestClassifier(**RF_PARAMETERS).fit(x,y,sample_weight=w)
            # Ambos clasificadores producen probabilidades para las tres clases de riesgo.
            p=predictions(model,current)
            if phase=='C':joblib.dump({'model':model,'training':training},DATA/f'modelo_{family}_{period}.joblib',compress=3)
        # EVALUACIÓN: compara con el mes observado y guarda métricas, matriz y probabilidades.
        m=metrics(data.target(month),p)
        save(OUT/phase/f'{family}_{window}_{period}.json',{'phase':phase,'period':period,'family':family,'window':window,
               'training':training,'metrics':m,'history':history,'seconds':time.perf_counter()-start})
        np.savez_compressed(DATA/f'{phase}_{family}_{window}_{period}.npz',truth=data.target(month),probability=p)
        print('RESULTADO',phase,family,period,window,'accuracy',round(m['accuracy'],5),'recall_alto',round(m['recall'][2],5),flush=True)
        del model,p;gc.collect()
    del entries,y,w,x,current;gc.collect()

def aggregate_records(rows):
    m=matrix_metrics(sum(np.asarray(r['metrics']['matriz_confusion']) for r in rows));m['balanced_accuracy']=float(np.mean(m['recall']));return m

# PRONÓSTICO: entrena con información hasta agosto de 2026 y exporta septiembre sin evaluarlo.
def export_forecast(data,selected):
    period='2026-09';month=104;roads=pd.read_csv(DATA/'tramos_osm.csv');deploy=DATA/'predicciones_web';deploy.mkdir(exist_ok=True)
    for family,c in selected.items():
        entries,y,w,training=data.select(month,c['window'])
        # Último periodo está fuera de la lista de etiquetas, pero sus predictores solo consultan 0..103.
        seed()
        if family=='lstm':
            seq=np.concatenate([data.sequence(m,ids) for m,ids in entries]);ctx=np.concatenate([data.context(m,ids) for m,ids in entries])
            scales=normalize_training(seq,ctx);loader=DataLoader(TensorDataset(torch.from_numpy(seq),torch.from_numpy(ctx),torch.from_numpy(y),torch.from_numpy(w)),batch_size=2048,shuffle=True,num_workers=0)
            model=MonthlyLSTM();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
            for epoch in range(c['epochs']):
                model.train()
                for a,b,t,ww in loader:
                    opt.zero_grad(set_to_none=True);loss=(F.cross_entropy(model(a,b),t,reduction='none')*ww).sum()/ww.sum();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            p=predict_lstm(model,data,month,scales);torch.save({'state_dict':model.state_dict(),'scales':scales,'training':training},DATA/'modelo_final_lstm.pt')
            del seq,ctx,loader,opt
        else:
            x=data.tabular(entries)
            if family=='xgboost':
                model=fit_model(SPEC,x,y,w)
            else:
                model=RandomForestClassifier(**RF_PARAMETERS).fit(x,y,sample_weight=w)
            p=predictions(model,data.features(month,np.arange(data.n)));joblib.dump({'model':model,'training':training},DATA/f'modelo_final_{family}.joblib',compress=3);del x
        # RISK SCORE: combina P(Medio) y P(Alto) en un valor continuo entre 0 y 1.
        score_values=.5*p[:,1]+p[:,2]
        frame=roads[['tramo_id','latitud','longitud']].copy();frame['periodo_objetivo']=period;frame['riesgo_score']=score_values
        frame['nivel_riesgo']=np.array(['bajo','medio','alto'])[p.argmax(1)];frame['modelo_usado']={'random_forest':'Random Forest','xgboost':'XGBoost','lstm':'LSTM'}[family]
        for k,name in enumerate(['prob_bajo','prob_medio','prob_alto']):frame[name]=p[:,k]
        frame.to_csv(deploy/f'predicciones_tramos_{family}.csv',index=False)
        metadata={'periodo_prediccion':period,'entrenamiento_hasta':'2026-08','tramo_turno':False,'unidad':'segmento × mes',
                  'modelo':family,'version':'snapping150_nkde250_mensual_2026_10_02','n_features':74 if family!='lstm' else 14,'training':training,
                  'metodo_espacial':'snapping 150 m + NKDE 250 m','coordinates':'originales','probability_class_order':['Bajo','Medio','Alto']}
        save(deploy/('metadata_modelo.json' if family=='random_forest' else f'metadata_modelo_{family}.json'),metadata)
        save(OUT/f'pronostico_{family}.json',{'period':period,'evaluated':False,'training':training,'counts_argmax':np.bincount(p.argmax(1),minlength=3).tolist(),
              'counts_score_bands':np.bincount(np.where(score_values>=.66,2,np.where(score_values>=.34,1,0)),minlength=3).tolist(),
              'score_quantiles':np.quantile(score_values,[0,.25,.5,.75,.95,1]).tolist()})
        del entries,y,w,model,p;gc.collect();print('PRONOSTICO',family,period,flush=True)

def main():
    seed();prepare();data=Data()
    registry={'registered_at_utc':now(),'spatial_method':'snapping único 150 m + NKDE 250 m',
              'source':'coordenadas originales de la base filtrada','periods':list(PERIODS),'phase_B':list(B),'phase_C':list(C),
              'windows':list(WINDOWS),'epochs_compared':[1,2,3,4],'sequence_months':12,'sequence_channels':14,'tabular_features':74,
              'nkde_features':'reemplazan vecindad euclidiana por densidad de frecuencia y gravedad sobre red',
              'rf':RF_PARAMETERS,'xgb':model_parameters(SPEC),'sampling_cap':CAP,'class_balance_power':1.25,'threshold':THRESHOLD,
              'selection':'0.4 Accuracy + 0.6 media(F1 Medio,F1 Alto), con guardia del 90% de Precision y Recall Alto del control 3m',
              'phase_C_status':'validación retrospectiva; etiquetas basadas en registros georreferenciados observados, con cobertura incompleta',
              'code_sha256':digest(__file__)}
    if not (OUT/'registro_inicial.json').exists():save(OUT/'registro_inicial.json',registry)
    # Una prueba de causalidad altera meses objetivo/futuros para verificar ausencia de lectura.
    m=PERIODS.index('2025-08');ids=np.arange(100);before=data.features(m,ids);seq=data.sequence(m,ids)
    oldp,oldd,oldt,oldr=data.panel,data.density,data.temporal,data.risk
    class PastOnly:
        def __init__(self,a):self.a=a
        def __getitem__(self,key):
            first=key[0] if isinstance(key,tuple) else key
            if isinstance(first,slice):assert first.stop<=m
            else:assert first<m
            return self.a[key]
    data.panel,data.density,data.temporal,data.risk=map(PastOnly,[oldp,oldd,oldt,oldr])
    assert np.array_equal(before,data.features(m,ids)) and np.array_equal(seq,data.sequence(m,ids))
    data.panel,data.density,data.temporal,data.risk=oldp,oldd,oldt,oldr
    save(OUT/'causalidad.json',{'passed':True,'max_predictor_month':'t-1','same_features':True,'same_sequence':True,'future_labels_excluded':True})
    # FASE B: 12 meses de evaluación entre 2022 y 2024; compara ventanas de 3, 12 y expansiva.
    for period in B:
        for window in WINDOWS:run_origin(data,'B',period,window,FAMILIES)
    # SELECCIÓN: 0,4 Accuracy + 0,6 promedio F1(Medio, Alto), con control de Precision/Recall Alto.
    candidates={};selected={}
    for family in FAMILIES:
        cc=[]
        for window in WINDOWS:
            rr=[read(OUT/'B'/f'{family}_{window}_{p}.json') for p in B]
            for epoch in (range(1,5) if family=='lstm' else [None]):
                rows=[{'metrics':r['history'][epoch-1]['metrics']} for r in rr] if epoch else rr
                metric=aggregate_records(rows);cc.append({'family':family,'window':window,'epochs':epoch,'metrics':metric,'score':score(metric)})
        control=next(c for c in cc if c['window']=='3m' and c['epochs'] in (None,4))
        for c in cc:c['eligible_high_guard']=(c['metrics']['precision'][2]>=.9*control['metrics']['precision'][2] and c['metrics']['recall'][2]>=.9*control['metrics']['recall'][2])
        candidates[family]=cc;selected[family]=max([c for c in cc if c['eligible_high_guard']],key=lambda c:c['score'])
    winner=max(selected.values(),key=lambda c:c['score'])['family']
    freeze={'frozen_at_utc':now(),'selected':selected,'candidates':candidates,'primary_model':winner,'selection_only_B':True,'confirmation_consulted':False}
    if not (OUT/'seleccion_congelada_antes_C.json').exists():save(OUT/'seleccion_congelada_antes_C.json',freeze)
    else:assert read(OUT/'seleccion_congelada_antes_C.json')['selected']==selected
    print('SELECCION',[(f,s['window'],s['epochs']) for f,s in selected.items()],'modelo',winner,flush=True)
    # FASE C: 20 meses entre enero de 2025 y agosto de 2026, con la configuración elegida en B.
    for period in C:
        for family,c in selected.items():run_origin(data,'C',period,c['window'],[family],epochs=c['epochs'] or 4)
    result={}
    for family,c in selected.items():
        rr=[read(OUT/'C'/f'{family}_{c["window"]}_{p}.json') for p in C]
        probs=[];truth=[]
        for p in C:
            a=np.load(DATA/f'C_{family}_{c["window"]}_{p}.npz');probs.append(a['probability']);truth.append(a['truth'])
        y=np.concatenate(truth);p=np.concatenate(probs);aggregate=metrics(y,p)
        result[family]={'selected':c,'monthly':rr,'metrics':aggregate}
        for periodset,name in [(tuple(s for s in C if s.startswith('2026')),'2026'),(tuple(s for s in C if s.startswith('2025')),'2025')]:
            indices=[C.index(s) for s in periodset];result[family][name]=metrics(np.concatenate([truth[i] for i in indices]),np.concatenate([probs[i] for i in indices]))
        pd.DataFrame(aggregate['matriz_confusion'],index=['Bajo','Medio','Alto'],columns=['Bajo','Medio','Alto']).to_csv(OUT/f'matriz_confusion_{family}.csv')
        del probs,truth,y,p;gc.collect()
    save(OUT/'resultados_finales.json',{'completed_at_utc':now(),'selected_primary_B':winner,'results':result,'new_fits_B':108,'new_fits_C':60,'evaluated_months_C':20,'prospective_evaluation':False})
    export_forecast(data,selected)
    print('COMPLETADO',OUT,flush=True)

if __name__=='__main__':main()
