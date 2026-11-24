"""Export the four frozen pipelines for browser inference. Never trains models."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,joblib

SOURCE=Path(sys.argv[1]) if len(sys.argv)>1 else Path(r'C:\Users\User\Desktop\laptoppredictions\milestone2(Model training)')
OUT=Path(__file__).resolve().parents[1]/'dist/data'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(SOURCE/'models'))
from features import engineer
def save(name,data):
 (OUT/name).write_text(json.dumps(data,separators=(',',':'),allow_nan=False),encoding='utf8')
train=pd.read_csv(SOURCE/'inputs/train.csv');test=pd.read_csv(SOURCE/'inputs/test.csv')
options={c:sorted(train[c].dropna().astype(str).unique().tolist()) for c in ['status','brand','model','cpu','cpu_model','storage_type','gpu','touch']}
cpu_pairs=train[['cpu_model','cpu']].drop_duplicates().to_dict('records')
brand_models=train[['brand','model']].drop_duplicates().to_dict('records')
save('options.json',dict(options=options,cpu_pairs=cpu_pairs,brand_models=brand_models))
checks=[];metadata={}
for name in ['linear','ridge','forest','xgboost']:
 path=SOURCE/'models'/(name+'.joblib');pipe=joblib.load(path);prep=pipe.regressor_.named_steps['prepare'];est=pipe.regressor_.named_steps['model']
 numeric=prep.transformers_[0];cats=prep.transformers_[1];scale=numeric[1].named_steps['scale'];imp=numeric[1].named_steps['impute'];enc=cats[1].named_steps['encode']
 fields=[];offset=len(numeric[2])
 for i,col in enumerate(cats[2]):
  transformed=enc._compute_transformed_categories(i);infrequent=set(enc.infrequent_categories_[i]) if enc.infrequent_categories_[i] is not None else set()
  indices={str(v):offset+j for j,v in enumerate(transformed)}
  mapping={str(v):indices['infrequent_sklearn' if v in infrequent else str(v)] for v in enc.categories_[i]}
  fields.append(dict(name=col,mapping=mapping,fallback=str(cats[1].named_steps['impute'].statistics_[i])))
  offset+=len(transformed)
 data=dict(name=name,dimension=offset,numeric=list(numeric[2]),median=imp.statistics_.tolist(),mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),categories=fields)
 if name in ['linear','ridge']:data.update(kind='linear',coef=est.coef_.tolist(),intercept=float(est.intercept_))
 elif name=='forest':
  data.update(kind='forest',trees=[dict(left=e.tree_.children_left.tolist(),right=e.tree_.children_right.tolist(),feature=e.tree_.feature.tolist(),threshold=e.tree_.threshold.tolist(),value=e.tree_.value[:,0,0].tolist()) for e in est.estimators_])
 else:
  raw=json.loads(bytes(est.get_booster().save_raw(raw_format='json')));learner=raw['learner'];base=learner['learner_model_param']['base_score'];base=json.loads(base) if base.startswith('[') else float(base)
  data.update(kind='xgboost',base=float(base[0] if isinstance(base,list) else base),trees=[dict(left=t['left_children'],right=t['right_children'],feature=t['split_indices'],threshold=t['split_conditions'],default=t['default_left']) for t in learner['gradient_booster']['model']['trees']])
 save(name+'.json',data)
 pred=pipe.predict(engineer(test));metadata[name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),test_mae=float(np.abs(pred-test.final_price).mean()),test_rmse=float(np.sqrt(np.mean((pred-test.final_price)**2))))
 for i,row in enumerate(test.to_dict('records')):
  if name=='linear':checks.append(dict(input=row,expected={}))
  checks[i]['expected'][name]=float(pred[i])
save('provenance.json',dict(source='Frozen after-feature pipelines; no retraining',models=metadata,test_rows=len(test)))
save('examples.json',[dict(id=int(r['id']),input={k:v for k,v in r.items() if k not in ['id','final_price']},actual=float(r['final_price'])) for r in test.to_dict('records')])
# Full parity cases stay local to repository verification, not required by the website.
(OUT.parents[1]/'scripts/parity_cases.json').write_text(json.dumps(checks,separators=(',',':')),encoding='utf8')
print('Exported four frozen models, training vocabularies and 432 real test examples.')
