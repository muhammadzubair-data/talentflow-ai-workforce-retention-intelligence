import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

FEATURES=['department','tenure_years','weekly_hours','engagement','training_hours','promotion_wait_years']
TARGET='left'
def generate(n=1800,seed=42):
    r=np.random.default_rng(seed)
    d=pd.DataFrame({'department':r.choice(['Operations','Engineering','Sales','Support','Finance'],n),'tenure_years':r.uniform(.2,15,n),'weekly_hours':r.uniform(30,60,n),'engagement':r.uniform(1,5,n),'training_hours':r.uniform(0,80,n),'promotion_wait_years':r.uniform(0,8,n)})
    z=-1.4+.08*(d.weekly_hours-40)-.65*(d.engagement-3)+.18*d.promotion_wait_years-.025*d.training_hours-.06*d.tenure_years
    d[TARGET]=r.binomial(1,1/(1+np.exp(-z)))
    return d.round(2)
def validate(d):
    if not set(FEATURES+[TARGET]).issubset(d.columns): raise ValueError('Missing required columns')
    d=d[FEATURES+[TARGET]].copy()
    if d.isna().any().any(): raise ValueError('Missing values are not supported')
    for k in FEATURES[1:]+[TARGET]: d[k]=pd.to_numeric(d[k],errors='raise')
    if not np.isfinite(d[FEATURES[1:]+[TARGET]].to_numpy()).all(): raise ValueError('Nonfinite values')
    if not d[TARGET].isin([0,1]).all() or d[TARGET].value_counts().min()<10 or d[TARGET].nunique()!=2 or len(d)<100: raise ValueError('Need at least 100 records and 10 per outcome')
    if (d[FEATURES[1:]]<0).any().any() or not d.engagement.between(1,5).all() or not d.weekly_hours.between(0,168).all(): raise ValueError('Invalid numeric ranges')
    d.department=d.department.astype(str).str.strip()
    if (d.department=='').any() or d.department.nunique()>50: raise ValueError('Invalid department categories')
    return d

def train(d):
    d=validate(d); xtr,xte,ytr,yte=train_test_split(d[FEATURES],d[TARGET],test_size=.25,random_state=42,stratify=d[TARGET]); pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),['department'])],remainder='passthrough'); model=make_pipeline(pre,LogisticRegression(max_iter=2000)); model.fit(xtr,ytr); p=model.predict_proba(xte)[:,1]; return model,{'ROC AUC':roc_auc_score(yte,p),'Average precision':average_precision_score(yte,p),'Holdout rows':len(yte)},confusion_matrix(yte,p>=.5)
def aggregate(d):
    a=d.groupby('department').agg(records=('left','size'),turnover=('left','mean'),engagement=('engagement','mean'),weekly_hours=('weekly_hours','mean')).reset_index(); return a[a.records>=10]
def scenario(model,d,hours=0,engagement=0,training=0):
    x=d[FEATURES].copy(); base=model.predict_proba(x)[:,1].mean(); x.weekly_hours=(x.weekly_hours+hours).clip(0,168); x.engagement=(x.engagement+engagement).clip(1,5); x.training_hours=(x.training_hours+training).clip(lower=0); return float(base),float(model.predict_proba(x)[:,1].mean())
