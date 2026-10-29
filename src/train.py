"""End-to-end experiment; all selection uses development data, never the test set.
Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python src/train.py
"""
from pathlib import Path
import hashlib, json, time, platform
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve, roc_curve)
from sklearn.inspection import permutation_importance
import joblib
from download import download, URL
from schema import FEATURES, SHORT

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'dist/assets'; OUT.mkdir(parents=True,exist_ok=True)
ART=ROOT/'artifacts'; ART.mkdir(exist_ok=True)
SEED=42

def write(path,obj):
    Path(path).write_text(json.dumps(obj,separators=(',',':'),allow_nan=False))

def metrics(y,p,t):
    pred=p>=t
    return dict(accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),
        recall=recall_score(y,pred),f1=f1_score(y,pred),ap=average_precision_score(y,p),
        auc=roc_auc_score(y,p),brier=brier_score_loss(y,p),threshold=float(t),
        confusion=confusion_matrix(y,pred,labels=[0,1]).tolist())

def optimal_threshold(y,p):
    prec,rec,ts=precision_recall_curve(y,p)
    fs=2*prec[:-1]*rec[:-1]/np.maximum(prec[:-1]+rec[:-1],1e-12)
    return float(ts[np.argmax(fs)])

def tree_export(tree, classifier=True):
    t=tree.tree_
    v=t.value[:,0,:]
    val=v[:,1]/np.maximum(v.sum(axis=1),1e-15) if classifier else v[:,0]
    return dict(left=t.children_left.tolist(),right=t.children_right.tolist(),
        feature=t.feature.tolist(),threshold=t.threshold.tolist(),value=val.tolist())

def export_model(name,base,cal,columns,threshold):
    o=dict(name=name,features=columns,threshold=threshold,
        calibration=dict(a=float(cal.calibrated_classifiers_[0].calibrators[0].a_),
                         b=float(cal.calibrated_classifiers_[0].calibrators[0].b_)))
    if hasattr(base,'steps'):
        sc,lr=base.steps[0][1],base.steps[1][1]
        o.update(kind='logistic',mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),
                 coef=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]))
    elif isinstance(base,DecisionTreeClassifier):
        o.update(kind='tree',trees=[tree_export(base)])
    elif isinstance(base,RandomForestClassifier):
        o.update(kind='forest',trees=[tree_export(t) for t in base.estimators_])
    else:
        prior=base.init_.class_prior_[1]
        o.update(kind='boosting',initial=float(np.log(prior/(1-prior))),rate=base.learning_rate,
                 trees=[tree_export(t[0],False) for t in base.estimators_])
    return o

def curve_sample(x,y,n=101):
    idx=np.unique(np.linspace(0,len(x)-1,min(n,len(x))).astype(int))
    return [[float(x[i]),float(y[i])] for i in idx]

def capacity(y,p,k):
    count=max(1,int(np.ceil(len(y)*k)))
    idx=np.argsort(-p,kind='stable')[:count]
    hits=int(np.asarray(y)[idx].sum())
    return dict(capacity=k,selected=count,captured=hits,recall=hits/int(np.sum(y)),precision=hits/count)

def main():
    start=time.time(); path=download(); df=pd.read_csv(path)
    columns=[f[0] for f in FEATURES]; X=df[columns]; y=df.Diabetes_binary
    assert df.shape==(253680,23), 'Unexpected source version: inspect before proceeding'
    assert not df.isna().any().any()
    assert set(y.unique())=={0,1}
    for c,_,lo,hi,_ in FEATURES: assert X[c].between(lo,hi).all(),c
    # Hash feature profiles only (not IDs or outcomes). Identical inputs never cross roles.
    groups=pd.util.hash_pandas_object(X,index=False).to_numpy()
    fold=np.empty(len(df),dtype=int)
    for i,(_,ix) in enumerate(StratifiedGroupKFold(10,shuffle=True,random_state=SEED).split(X,y,groups)):
        fold[ix]=i
    splits=dict(train=np.flatnonzero(fold>=4),calibration=np.flatnonzero(fold==3),
                validation=np.flatnonzero(fold==2),test=np.flatnonzero(fold<2))
    for a in splits:
        for b in splits:
            if a!=b: assert not set(groups[splits[a]])&set(groups[splits[b]])
    pd.DataFrame({'ID':df.ID,'role':np.select([fold<2,fold==2,fold==3],['test','validation','calibration'],default='train')}).to_csv(ART/'split_assignments.csv',index=False)
    tr,ca,va,te=[splits[k] for k in ['train','calibration','validation','test']]
    audit=dict(rows=len(df),features=len(columns),positive=int(y.sum()),negative=int((1-y).sum()),
        prevalence=float(y.mean()),missing=int(df.isna().sum().sum()),duplicate_feature_rows=int(X.duplicated().sum()),
        duplicate_feature_target_rows=int(df.drop(columns='ID').duplicated().sum()),
        conflicting_profiles=int(pd.DataFrame({'g':groups,'y':y}).groupby('g').y.nunique().gt(1).sum()),
        source=URL,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        target='UCI Diabetes_binary: 0 = no diabetes; 1 = prediabetes or diabetes',
        splits={k:dict(n=len(ix),positive=int(y.iloc[ix].sum()),prevalence=float(y.iloc[ix].mean())) for k,ix in splits.items()},
        data_dictionary=[dict(key=c,label=l,min=lo,max=hi,description=d) for c,l,lo,hi,d in FEATURES],
        distributions={c:[dict(value=int(v),n=int(n),positive_rate=float(y[X[c]==v].mean())) for v,n in X[c].value_counts().sort_index().items()] for c in columns},
        versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__),seed=SEED)
    write(OUT/'audit.json',audit)
    print('AUDIT',audit['splits'],flush=True)
    specs=[
      ('Logistic regression',make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=SEED)),{'logisticregression__C':[0.1,1.0]},columns),
      ('Decision tree',DecisionTreeClassifier(min_samples_leaf=100,random_state=SEED),{'max_depth':[4,8]},columns),
      ('Random forest',RandomForestClassifier(n_estimators=80,min_samples_leaf=30,n_jobs=2,random_state=SEED),{'max_depth':[8,12]},columns),
      ('Gradient boosting',GradientBoostingClassifier(n_estimators=70,learning_rate=0.07,min_samples_leaf=60,subsample=0.8,random_state=SEED),{'max_depth':[2,3]},columns),
      ('Short-form logistic',make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=SEED)),{'logisticregression__C':[0.1,1.0]},SHORT),
    ]
    trained=[]; exported=[]
    cv=StratifiedGroupKFold(3,shuffle=True,random_state=SEED)
    for name,base,grid,cols in specs:
        t0=time.time(); print('TRAINING',name,flush=True)
        search=GridSearchCV(base,grid,scoring='average_precision',cv=cv,n_jobs=1,refit=True)
        search.fit(X.iloc[tr][cols],y.iloc[tr],groups=groups[tr])
        base=search.best_estimator_
        cal=CalibratedClassifierCV(FrozenEstimator(base),method='sigmoid')
        cal.fit(X.iloc[ca][cols],y.iloc[ca])
        pv=cal.predict_proba(X.iloc[va][cols])[:,1]
        threshold=optimal_threshold(y.iloc[va],pv)
        val=metrics(y.iloc[va],pv,threshold)
        trained.append((name,base,cal,cols,threshold,dict(name=name,params=search.best_params_,cv_ap=float(search.best_score_),validation=val,training_seconds=time.time()-t0)))
        exported.append(export_model(name,base,cal,cols,threshold))
        joblib.dump(cal,ART/(name.lower().replace(' ','_')+'.joblib'))
        print('VALIDATION',name,round(val['ap'],4),flush=True)
    winner=max(trained[:4],key=lambda a:a[5]['validation']['ap'])[0]
    write(ART/'frozen_selection.json',dict(winner=winner,criterion='Validation average precision among four full models',thresholds={a[0]:a[4] for a in trained}))
    print('FROZEN WINNER',winner,'Now evaluating untouched test set.',flush=True)
    results=[]; predictions={}; probes=[]
    yp=y.iloc[te].to_numpy()
    for name,base,cal,cols,threshold,res in trained:
        xt=X.iloc[te][cols]; pt=cal.predict_proba(xt)[:,1]; predictions[name]=pt
        res['test']=metrics(yp,pt,threshold)
        res['uncalibrated_brier']=float(brier_score_loss(yp,base.predict_proba(xt)[:,1]))
        pr,re,_=precision_recall_curve(yp,pt); fpr,tpr,_=roc_curve(yp,pt)
        res['pr_curve']=curve_sample(re,pr); res['roc_curve']=curve_sample(fpr,tpr)
        fraction,mean=calibration_curve(yp,pt,n_bins=10,strategy='quantile')
        res['calibration_curve']=curve_sample(mean,fraction)
        res['capacity']=[capacity(yp,pt,k/100) for k in range(5,101,5)]
        rng=np.random.default_rng(SEED); ids=rng.choice(len(te),min(1200,len(te)),replace=False)
        imp=permutation_importance(cal,xt.iloc[ids],yp[ids],scoring='average_precision',n_repeats=3,random_state=SEED,n_jobs=1)
        res['importance']=[dict(feature=c,mean=float(m),std=float(s)) for c,m,s in zip(cols,imp.importances_mean,imp.importances_std)]
        res['subgroups']=[]
        for c,masks in [('Sex',[(str(v),X.iloc[te].Sex.to_numpy()==v) for v in [0,1]]),('Age',[('18–44',X.iloc[te].Age.to_numpy()<=5),('45–64',(X.iloc[te].Age.to_numpy()>=6)&(X.iloc[te].Age.to_numpy()<=9)),('65+',X.iloc[te].Age.to_numpy()>=10)])]:
            for label,mask in masks:
                met=metrics(yp[mask],pt[mask],threshold); tn,fp=met['confusion'][0]
                res['subgroups'].append(dict(group=c,label=label,n=int(mask.sum()),positive=int(yp[mask].sum()),fpr=fp/(tn+fp),**met))
        lat=[]
        for _ in range(30):
            t=time.perf_counter(); cal.predict_proba(xt.iloc[:1]); lat.append((time.perf_counter()-t)*1000)
        res['python_latency_ms']=dict(median=float(np.median(lat)),p95=float(np.percentile(lat,95)))
        results.append(res)
        # Artificial profiles exercise every supported input range without publishing survey rows.
        rng_probe=np.random.default_rng(SEED)
        artificial=pd.DataFrame({c:rng_probe.integers(lo,hi+1,100) for c,_,lo,hi,_ in FEATURES})
        probes.append(dict(name=name,inputs=artificial.to_dict('records'),expected=cal.predict_proba(artificial[cols])[:,1].tolist()))
        print('TEST',name,res['test'],flush=True)
    dummy=np.full(len(te),float(y.iloc[tr].mean()))
    summary=dict(winner=winner,selection='Highest validation average precision among four full-feature models',
        models=results,dummy=metrics(yp,dummy,0.5),test_n=len(te),test_positive=int(yp.sum()),
        short_features=SHORT,question_reduction=1-len(SHORT)/len(columns))
    # Cluster bootstrap resamples feature-profile groups, preserving dependence within a profile.
    _, inv=np.unique(groups[te],return_inverse=True); ng=inv.max()+1
    pwin=predictions[winner]; plog=predictions['Logistic regression']
    winthreshold=next(a[4] for a in trained if a[0]==winner)
    recall_ci=[]; ap_delta=[]; rng=np.random.default_rng(SEED)
    for _ in range(200):
        counts=np.bincount(rng.integers(0,ng,ng),minlength=ng); weights=counts[inv]
        recall_ci.append(recall_score(yp,pwin>=winthreshold,sample_weight=weights))
        ap_delta.append(average_precision_score(yp,pwin,sample_weight=weights)-average_precision_score(yp,plog,sample_weight=weights))
    summary['uncertainty']=dict(method='200 feature-profile cluster bootstrap replicates; percentile 95% intervals; fixed fitted models',
        winner_recall_ci=np.percentile(recall_ci,[2.5,97.5]).tolist(),ap_difference_vs_logistic_ci=np.percentile(ap_delta,[2.5,97.5]).tolist())
    summary['runtime_seconds']=time.time()-start
    write(OUT/'results.json',summary); write(OUT/'models.json',exported); write(ART/'parity_probes.json',probes)
    # Static simulator uses aggregate results only, never publishes individual survey records.
    # Store test predictions locally for reproduction; omit raw data from git/site.
    np.savez_compressed(ROOT/'data/test_predictions.npz',y=yp,**{str(i):predictions[n] for i,n in enumerate(predictions)})
    pd.DataFrame([dict(model=r['name'],**{k:v for k,v in r['test'].items() if k!='confusion'}) for r in results]).to_csv(OUT/'metrics.csv',index=False)
    print('DONE seconds',round(time.time()-start),flush=True)

if __name__=='__main__': main()
