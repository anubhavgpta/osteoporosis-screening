"""Reproduce author-verification checks without fitting or altering saved models."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from osteoporosis_screening.evaluation import classification_metrics

OUT = ROOT / 'manuscript' / 'verification_outputs'
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
meta = json.loads((ROOT/'artifacts/classification_model_metadata.json').read_text())
manifest = json.loads((ROOT/'reports/tables/external_knhanes_2008_2011_manifest.json').read_text())
checks = {'artifact_hashes': {}}
for name, path in [('model','artifacts/classification_model.joblib'),('model_metadata','artifacts/classification_model_metadata.json'),('input','data/processed/knhanes_2008_2011_harmonized.csv.gz')]:
    actual = sha(ROOT/path)
    checks['artifact_hashes'][name] = {'actual':actual,'matches_external_manifest':actual==manifest[name+'_sha256']}
cohort = pd.read_csv(ROOT/'data/processed/classification_cohort.csv.gz')
external = pd.read_csv(ROOT/'data/processed/knhanes_2008_2011_harmonized.csv.gz')
external = external.loc[external.age.ge(50) & external.femoral_neck_bmd.notna()].copy()
external['osteoporosis'] = ((external.femoral_neck_bmd-.858)/.120 <= -2.5).astype(int)
checks['cohort'] = {'nhanes_n':len(cohort),'nhanes_cases':int(cohort.osteoporosis.sum()),'knhanes_n':len(external),'knhanes_cases':int(external.osteoporosis.sum()),'nhanes_duplicate_ids':int(cohort.participant_id.duplicated().sum()),'knhanes_duplicate_ids':int(external.participant_id.duplicated().sum())}
demo=[]
for name,frame,cycle in [('NHANES',cohort,'survey_cycle'),('KNHANES',external,'survey_year')]:
    for year,g in frame.groupby(cycle):
        demo.append({'survey':name,'cycle':year,'n':len(g),'cases':int(g.osteoporosis.sum()),'male_n':int(g.sex.eq(1).sum()),'female_n':int(g.sex.eq(2).sum()),'age_mean_unweighted':g.age.mean(),'age_min':g.age.min(),'age_max':g.age.max()})
pd.DataFrame(demo).to_csv(OUT/'cohort_by_cycle.csv',index=False)
rows=[]
predictions={}
for name,file,threshold in [('internal','classification_test_predictions.csv',meta['sensitivity_oriented_threshold']),('temporal','classification_temporal_validation_test_predictions.csv',.029248871071765024),('external','external_knhanes_2008_2011_predictions.csv',meta['sensitivity_oriented_threshold'])]:
    f=pd.read_csv(ROOT/'reports/tables'/file)
    predictions[name]=f
    metrics=classification_metrics(f.osteoporosis,f.probability,threshold,f.analysis_weight)
    rows.append({'evaluation':name,'n':len(f),'cases':int(f.osteoporosis.sum()),'weighted_prevalence':np.average(f.osteoporosis,weights=f.analysis_weight),'mean_predicted_probability':np.average(f.probability,weights=f.analysis_weight),**metrics})
pd.DataFrame(rows).to_csv(OUT/'recomputed_metrics.csv',index=False)
# Inspect subgroup denominators at the single published global threshold.
# Kish effective sample size is descriptive, not a complex-survey variance estimate.
sg=[]
for name in ['internal','temporal']:
    f=predictions[name].copy()
    f['age_group']=pd.cut(f.age,[49,64,74,np.inf],labels=['50–64','65–74','75+'])
    threshold=meta['sensitivity_oriented_threshold'] if name=='internal' else .029248871071765024
    for dim in ['sex','age_group','race_ethnicity']:
        for group,g in f.groupby(dim,observed=True):
            y=g.osteoporosis.to_numpy();p=g.probability.to_numpy();w=g.analysis_weight.to_numpy();pos=p>=threshold
            sg.append({'evaluation':name,'dimension':dim,'group':str(group),'n':len(g),'cases':int(y.sum()),'noncases':int((y==0).sum()),'positive_screens':int(pos.sum()),'false_negatives':int(((y==1)&~pos).sum()),'kish_effective_n':w.sum()**2/(w*w).sum(),'sensitivity':w[(y==1)&pos].sum()/w[y==1].sum() if y.sum() else np.nan,'specificity':w[(y==0)&~pos].sum()/w[y==0].sum() if (y==0).sum() else np.nan})
pd.DataFrame(sg).to_csv(OUT/'subgroup_denominators.csv',index=False)
# Verify source-file hashes and the actual 2009 DXA-file weight, not only basic data.
hm=json.loads((ROOT/'reports/tables/knhanes_2008_2011_harmonization_manifest.json').read_text())
checks['source_hashes']={p:sha(ROOT/Path(p.replace('\\','/')))==h for p,h in hm['source_files'].items()}
dxa=pd.read_sas(ROOT/'data/raw_knhanes/KNHANES_2009_DXA_SAS.sas7bdat')
ids=external.loc[external.survey_year.eq(2009),'participant_id']
checks['external_id_example']=str(ids.iloc[0])
# Independently match the identifier suffix used by the harmonizer.
dxa['raw_id']=dxa.ID.map(lambda x:x.decode() if isinstance(x,bytes) else str(x))
g=external.loc[external.survey_year.eq(2009)].copy()
g['raw_id']=g.participant_id.astype(str).str.split('_').str[-1]
m=g.merge(dxa[['raw_id','wt_itvex','wt_ex']],on='raw_id',validate='one_to_one')
assert len(m)==len(g)
checks['weight_2009']={'n':len(m),'stored_weight_matches_joint_weight':bool(np.allclose(m.external_weight,m.wt_itvex*200/580)),'joint_and_exam_weights_differ_n':int((~np.isclose(m.wt_itvex,m.wt_ex)).sum())}
checks['software']={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}
(OUT/'verification_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
print(pd.DataFrame(rows)[['evaluation','roc_auc','pr_auc','sensitivity','specificity','ppv','npv','calibration_intercept','calibration_slope']].to_string(index=False))

