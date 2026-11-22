"""TalentFlow AI analytics core: synthetic workforce data, validation, aggregate analysis and model evaluation.

Everything here works on organisation-level aggregates. The model is used to report evaluation metrics and the
average change in model score under hypothetical scenarios; no function returns a ranked list of individuals.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ['department', 'tenure_years', 'weekly_hours', 'engagement', 'training_hours', 'promotion_wait_years']
NUMERIC = FEATURES[1:]
TARGET = 'left'
YEAR = 'year'
DEPARTMENTS = ['Operations', 'Engineering', 'Sales', 'Support', 'Finance']
MIN_GROUP = 10          # smallest group whose statistics are ever shown
MAX_DEPARTMENTS = 50
MAX_ROWS = 200_000
SEED = 42

LABELS = {'department': 'Department', 'tenure_years': 'Tenure (years)', 'weekly_hours': 'Weekly hours',
          'engagement': 'Engagement (1–5)', 'training_hours': 'Training hours',
          'promotion_wait_years': 'Years since promotion', 'year': 'Year', 'left': 'Left (0/1)'}

# Annual turnover profile used only to spread synthetic records across snapshot years.
YEAR_PROFILE = {2021: 0.17, 2022: 0.27, 2023: 0.23, 2024: 0.20, 2025: 0.18}

ALIASES = {
    'dept': 'department', 'department_name': 'department', 'team': 'department', 'business_unit': 'department',
    'tenure': 'tenure_years', 'years_at_company': 'tenure_years', 'yearsatcompany': 'tenure_years',
    'service_years': 'tenure_years', 'tenure_yrs': 'tenure_years',
    'hours_per_week': 'weekly_hours', 'avg_weekly_hours': 'weekly_hours', 'weekly_hrs': 'weekly_hours',
    'engagement_score': 'engagement', 'engagement_1_5': 'engagement',
    'training': 'training_hours', 'training_hrs': 'training_hours',
    'years_since_last_promotion': 'promotion_wait_years', 'yearssincelastpromotion': 'promotion_wait_years',
    'promotion_wait': 'promotion_wait_years', 'years_since_promotion': 'promotion_wait_years',
    'attrition': 'left', 'turnover': 'left', 'exited': 'left', 'churn': 'left', 'terminated': 'left',
    'snapshot_year': 'year', 'fiscal_year': 'year', 'record_year': 'year',
}
# Columns that must never become model inputs or be analysed (protected or proxy characteristics).
PROTECTED_HINTS = ('age', 'birth', 'dob', 'gender', 'sex', 'race', 'ethnic', 'religio', 'nationality', 'citizen',
                   'marital', 'married', 'pregnan', 'disab', 'veteran', 'orientation', 'caste', 'color', 'colour',
                   'national_origin', 'immigration', 'visa', 'union', 'health', 'medical')
IDENTIFIER_HINTS = ('name', 'email', 'phone', 'address', 'employee_id', 'emp_id', 'employeenumber', 'staff_id',
                    'ssn', 'social_security', 'passport', 'national_id', 'cnic', 'postcode', 'zip', 'manager_id',
                    'id', 'employee_number', 'employeenumber', 'salary', 'pay', 'wage', 'bonus', 'compensation')
ID_PREFIXES = {'employee', 'emp', 'staff', 'person', 'worker', 'user', 'member', 'badge', 'record', 'payroll'}
YES = {'1', '1.0', 'yes', 'y', 'true', 't', 'left', 'leaver', 'exited', 'terminated', 'attrited'}
NO = {'0', '0.0', 'no', 'n', 'false', 'f', 'stayed', 'stay', 'active', 'retained', 'current'}

TENURE_BANDS = ([0, 1, 2, 4, 7, 10, np.inf], ['<1 yr', '1–2 yrs', '2–4 yrs', '4–7 yrs', '7–10 yrs', '10+ yrs'])
HOURS_BANDS = ([0, 35, 40, 45, 50, 55, 169], ['<35 h', '35–40 h', '40–45 h', '45–50 h', '50–55 h', '55+ h'])
ENGAGE_BANDS = ([1, 2, 3, 4, 5.0001], ['1–2 low', '2–3', '3–4', '4–5 high'])
TRAIN_BANDS = ([0, 10, 25, 40, 60, np.inf], ['<10 h', '10–25 h', '25–40 h', '40–60 h', '60+ h'])
PROMO_BANDS = ([0, 1, 2, 4, 6, np.inf], ['<1 yr', '1–2 yrs', '2–4 yrs', '4–6 yrs', '6+ yrs'])
BANDS = {'tenure_years': TENURE_BANDS, 'weekly_hours': HOURS_BANDS, 'engagement': ENGAGE_BANDS,
         'training_hours': TRAIN_BANDS, 'promotion_wait_years': PROMO_BANDS}


# ---------------------------------------------------------------------------------------------- data
def generate(n: int = 1800, seed: int = SEED, with_year: bool = True) -> pd.DataFrame:
    """Synthetic employee-year records. The six features and the label are produced exactly as in the original
    project (same seed, same values). The optional ``year`` column comes from a separate random stream, so adding
    it does not change any original column."""
    r = np.random.default_rng(seed)
    d = pd.DataFrame({'department': r.choice(DEPARTMENTS, n), 'tenure_years': r.uniform(.2, 15, n),
                      'weekly_hours': r.uniform(30, 60, n), 'engagement': r.uniform(1, 5, n),
                      'training_hours': r.uniform(0, 80, n), 'promotion_wait_years': r.uniform(0, 8, n)})
    z = (-1.4 + .08 * (d.weekly_hours - 40) - .65 * (d.engagement - 3) + .18 * d.promotion_wait_years
         - .025 * d.training_hours - .06 * d.tenure_years)
    d[TARGET] = r.binomial(1, 1 / (1 + np.exp(-z)))
    d = d.round(2)
    if with_year:
        years = np.array(sorted(YEAR_PROFILE))
        rates = np.array([YEAR_PROFILE[y] for y in years])
        r2 = np.random.default_rng(seed + 7919)
        p_leave, p_stay = rates / rates.sum(), (1 - rates) / (1 - rates).sum()
        left = d[TARGET].to_numpy() == 1
        yr = np.empty(n, dtype=int)
        yr[left] = r2.choice(years, left.sum(), p=p_leave)
        yr[~left] = r2.choice(years, (~left).sum(), p=p_stay)
        d[YEAR] = yr
    return d


def validate(d: pd.DataFrame) -> pd.DataFrame:
    """Strict validation (original behaviour): raises ValueError on any problem. Keeps ``year`` if present."""
    if not set(FEATURES + [TARGET]).issubset(d.columns):
        raise ValueError('Missing required columns')
    keep = FEATURES + [TARGET] + ([YEAR] if YEAR in d.columns else [])
    d = d[keep].copy()
    if d.isna().any().any():
        raise ValueError('Missing values are not supported')
    for k in NUMERIC + [TARGET] + ([YEAR] if YEAR in d.columns else []):
        d[k] = pd.to_numeric(d[k], errors='raise')
    if not np.isfinite(d[NUMERIC + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite values')
    if (not d[TARGET].isin([0, 1]).all() or d[TARGET].nunique() != 2 or d[TARGET].value_counts().min() < 10
            or len(d) < 100):
        raise ValueError('Need at least 100 records and 10 per outcome')
    if (d[NUMERIC] < 0).any().any() or not d.engagement.between(1, 5).all() or not d.weekly_hours.between(0, 168).all():
        raise ValueError('Invalid numeric ranges')
    d['department'] = d['department'].astype(str).str.strip()
    if (d.department == '').any() or d.department.nunique() > MAX_DEPARTMENTS:
        raise ValueError('Invalid department categories')
    d[TARGET] = d[TARGET].astype(int)
    if YEAR in d.columns:
        d[YEAR] = d[YEAR].astype(int)
    return d


@dataclass
class DataReport:
    rows_in: int = 0
    rows_out: int = 0
    renamed: dict = field(default_factory=dict)
    ignored_protected: list = field(default_factory=list)
    ignored_identifier: list = field(default_factory=list)
    ignored_other: list = field(default_factory=list)
    dropped_missing: int = 0
    dropped_range: int = 0
    dropped_label: int = 0
    duplicate_rows: int = 0
    departments: int = 0
    small_departments: int = 0
    has_year: bool = False
    notes: list = field(default_factory=list)

    def as_frame(self) -> pd.DataFrame:
        rows = [('Rows in file', self.rows_in), ('Rows kept for analysis', self.rows_out),
                ('Dropped: missing or non-numeric values', self.dropped_missing),
                ('Dropped: values outside valid ranges', self.dropped_range),
                ('Dropped: unrecognised outcome label', self.dropped_label),
                ('Exact duplicate rows (kept)', self.duplicate_rows),
                ('Departments', self.departments),
                (f'Departments below {MIN_GROUP} records (hidden in summaries)', self.small_departments),
                ('Year column available', 'yes' if self.has_year else 'no'),
                ('Columns renamed', ', '.join(f'{a} → {b}' for a, b in self.renamed.items()) or '—'),
                ('Ignored: protected or proxy characteristics', ', '.join(self.ignored_protected) or '—'),
                ('Ignored: identifiers or pay data', ', '.join(self.ignored_identifier) or '—'),
                ('Ignored: other columns', ', '.join(self.ignored_other) or '—')]
        return pd.DataFrame(rows, columns=['check', 'result']).astype({'result': str})


def _norm(col: str) -> str:
    return re.sub(r'[^0-9a-z]+', '_', str(col).strip().lower()).strip('_')


def _matches(name: str, hints) -> bool:
    parts = set(name.split('_'))
    for h in hints:
        if h == 'id':
            if (name == 'id' or name.endswith('_id') or 'id' in parts
                    or (name.endswith('id') and name[:-2] in ID_PREFIXES)):
                return True
        elif h in ('age', 'sex', 'race', 'pay', 'zip', 'visa', 'union', 'color', 'colour'):
            if h in parts or name == h:
                return True
        elif h in name:
            return True
    return False


def prepare(raw: pd.DataFrame, max_rows: int = MAX_ROWS) -> tuple[pd.DataFrame, DataReport]:
    """Forgiving validation for uploaded files. Maps common header names, ignores protected characteristics and
    identifiers, drops invalid rows (and reports how many), and raises ValueError only when the file cannot be
    analysed at all."""
    rep = DataReport(rows_in=len(raw))
    if len(raw) > max_rows:
        raise ValueError(f'The file has {len(raw):,} rows; the limit for this demo is {max_rows:,}.')
    cols, used = {}, set()
    for c in raw.columns:
        n = _norm(c)
        target = n if n in FEATURES + [TARGET, YEAR] else ALIASES.get(n)
        if target and target not in used:
            cols[c] = target
            used.add(target)
            if _norm(c) != target:
                rep.renamed[str(c)] = target
        elif _matches(n, PROTECTED_HINTS):
            rep.ignored_protected.append(str(c))
        elif _matches(n, IDENTIFIER_HINTS):
            rep.ignored_identifier.append(str(c))
        else:
            rep.ignored_other.append(str(c))
    missing = [c for c in FEATURES + [TARGET] if c not in used]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing) +
                         '. Expected: ' + ', '.join(FEATURES + [TARGET]) + ' (year is optional).')
    d = raw[list(cols)].rename(columns=cols).copy()

    lab = d[TARGET].astype(str).str.strip().str.lower()
    y = pd.Series(np.where(lab.isin(YES), 1, np.where(lab.isin(NO), 0, -1)), index=d.index)
    bad_label = (y == -1) & d[TARGET].notna() & (lab != '') & (lab != 'nan')
    rep.dropped_label = int(bad_label.sum())
    d[TARGET] = y.where(y >= 0)
    d = d[~bad_label]

    d['department'] = d['department'].astype('string').str.strip()
    d.loc[d['department'].isin(['', 'nan', 'None', 'NaN']), 'department'] = pd.NA
    for k in NUMERIC:
        d[k] = pd.to_numeric(d[k], errors='coerce')
    miss = d[FEATURES + [TARGET]].isna().any(axis=1) | ~np.isfinite(d[NUMERIC].to_numpy(dtype=float)).all(axis=1)
    rep.dropped_missing = int(miss.sum())
    d = d[~miss]
    ok = ((d[NUMERIC] >= 0).all(axis=1) & d.engagement.between(1, 5) & d.weekly_hours.between(0, 168)
          & (d.tenure_years <= 70) & (d.promotion_wait_years <= 70) & (d.training_hours <= 2000))
    rep.dropped_range = int((~ok).sum())
    d = d[ok]

    if YEAR in d.columns:
        yr = pd.to_numeric(d[YEAR], errors='coerce')
        if yr.notna().all() and yr.between(1950, 2100).all() and (yr == yr.round()).all():
            d[YEAR] = yr.astype(int)
            rep.has_year = True
        else:
            d = d.drop(columns=YEAR)
            rep.notes.append('The year column had missing or invalid values, so yearly views are turned off.')

    d['department'] = d['department'].astype(str)
    d[TARGET] = d[TARGET].astype(int)
    d = d.reset_index(drop=True)
    rep.rows_out = len(d)
    rep.duplicate_rows = int(d.duplicated().sum())
    counts = d['department'].value_counts()
    rep.departments, rep.small_departments = int(len(counts)), int((counts < MIN_GROUP).sum())
    if rep.departments > MAX_DEPARTMENTS:
        raise ValueError(f'{rep.departments} departments found; the limit is {MAX_DEPARTMENTS}. '
                         'Group small teams into broader departments first.')
    vc = d[TARGET].value_counts()
    if len(d) < 100 or len(vc) < 2 or vc.min() < 10:
        raise ValueError(f'After cleaning, {len(d)} valid rows remain with {int(vc.get(1, 0))} leavers and '
                         f'{int(vc.get(0, 0))} stayers. At least 100 rows and 10 of each outcome are needed.')
    if rep.ignored_protected:
        rep.notes.append('Protected or proxy characteristics were found and excluded from every analysis and model.')
    return d[FEATURES + [TARGET] + ([YEAR] if rep.has_year else [])], rep


# ---------------------------------------------------------------------------------------------- aggregates
def wilson(k, n, z: float = 1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = k / n
        den = 1 + z ** 2 / n
        mid = (p + z ** 2 / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return mid - half, mid + half


def band(d: pd.DataFrame, col: str) -> pd.Series:
    edges, labels = BANDS[col]
    return pd.cut(d[col], edges, labels=labels, right=False, include_lowest=True)


def aggregate(d: pd.DataFrame, min_n: int = MIN_GROUP) -> pd.DataFrame:
    """Department summary (original output columns); groups below ``min_n`` records are suppressed."""
    a = d.groupby('department').agg(records=('left', 'size'), turnover=('left', 'mean'),
                                    engagement=('engagement', 'mean'), weekly_hours=('weekly_hours', 'mean')).reset_index()
    return a[a.records >= min_n].reset_index(drop=True)


def group_rates(d: pd.DataFrame, by, min_n: int = MIN_GROUP) -> tuple[pd.DataFrame, int]:
    """Turnover by group with Wilson 95% intervals. Returns (visible groups, number of suppressed groups)."""
    keys = [by] if isinstance(by, str) else list(by)
    g = d.groupby(keys, observed=True).agg(records=(TARGET, 'size'), leavers=(TARGET, 'sum'),
                                           engagement=('engagement', 'mean'), weekly_hours=('weekly_hours', 'mean'),
                                           tenure_years=('tenure_years', 'mean'),
                                           training_hours=('training_hours', 'mean'),
                                           promotion_wait_years=('promotion_wait_years', 'mean')).reset_index()
    g['turnover'] = g.leavers / g.records
    g['ci_low'], g['ci_high'] = wilson(g.leavers, g.records)
    hidden = int((g.records < min_n).sum())
    return g[g.records >= min_n].reset_index(drop=True), hidden


def band_rates(d: pd.DataFrame, col: str, min_n: int = MIN_GROUP) -> tuple[pd.DataFrame, int]:
    x = d.assign(band=band(d, col))
    return group_rates(x, 'band', min_n)


def yearly(d: pd.DataFrame, min_n: int = MIN_GROUP, by: str | None = None) -> tuple[pd.DataFrame, int]:
    if YEAR not in d.columns:
        return pd.DataFrame(), 0
    return group_rates(d, [YEAR] + ([by] if by else []), min_n)


def heatmap(d: pd.DataFrame, row: str, col: str, min_n: int = MIN_GROUP):
    """Turnover-rate grid of two banded factors; cells under ``min_n`` records are blank (NaN)."""
    x = d.assign(r=band(d, row) if row in BANDS else d[row], c=band(d, col) if col in BANDS else d[col])
    g = x.groupby(['r', 'c'], observed=False)[TARGET].agg(['size', 'mean'])
    rate = g['mean'].unstack('c')
    n = g['size'].unstack('c').fillna(0).astype(int)
    rate = rate.where(n >= min_n)
    return rate, n, int(((n > 0) & (n < min_n)).sum().sum())


def department_profile(d: pd.DataFrame, model=None, min_n: int = MIN_GROUP) -> tuple[pd.DataFrame, int]:
    g, hidden = group_rates(d, 'department', min_n)
    org = d[TARGET].mean()
    g['share'] = g.records / len(d)
    g['vs_org_pts'] = (g.turnover - org) * 100
    if model is not None and len(g):
        scores = pd.Series(model.predict_proba(d[FEATURES])[:, 1], index=d.index)
        g = g.merge(scores.groupby(d['department']).mean().rename('mean_model_score'), left_on='department',
                    right_index=True, how='left')
    return g.sort_values('turnover', ascending=False).reset_index(drop=True), hidden


# ---------------------------------------------------------------------------------------------- model
def make_model(class_weight=None, C: float = 1.0):
    """App model: one-hot department + standardised numeric features → logistic regression."""
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), ['department']),
                             ('num', StandardScaler(), NUMERIC)])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=C, class_weight=class_weight))


def original_model():
    """The original project's pipeline (no scaling), kept for comparison."""
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), ['department'])], remainder='passthrough')
    return make_pipeline(pre, LogisticRegression(max_iter=2000))


def split(d: pd.DataFrame, test_size: float = .25, seed: int = SEED):
    return train_test_split(d[FEATURES], d[TARGET], test_size=test_size, random_state=seed, stratify=d[TARGET])


def train(d: pd.DataFrame):
    """Original API: fit on a stratified 75% split; returns (model, metrics, confusion matrix at 0.5)."""
    d = validate(d)
    xtr, xte, ytr, yte = split(d)
    model = make_model().fit(xtr, ytr)
    p = model.predict_proba(xte)[:, 1]
    return model, {'ROC AUC': roc_auc_score(yte, p), 'Average precision': average_precision_score(yte, p),
                   'Holdout rows': len(yte)}, confusion_matrix(yte, p >= .5)


def best_threshold(y, p) -> float:
    """Threshold that maximises F1 (ties → higher threshold). Used on out-of-fold training predictions only."""
    grid = np.round(np.arange(.05, .951, .01), 2)
    f1 = [f1_score(y, p >= t, zero_division=0) for t in grid]
    best = max(f1)
    return float(max(t for t, f in zip(grid, f1) if f == best))


def classification(y, p, t: float) -> dict:
    pred = p >= t
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'threshold': t, 'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)), 'f1': float(f1_score(y, pred, zero_division=0)),
            'accuracy': float((tp + tn) / len(y)), 'flagged_share': float(pred.mean()), 'tp': int(tp), 'fp': int(fp),
            'fn': int(fn), 'tn': int(tn)}


@dataclass
class Evaluation:
    model: object
    comparison: pd.DataFrame
    metrics: dict
    cm: np.ndarray
    cm_default: np.ndarray
    curves: dict
    thresholds: pd.DataFrame
    calibration: pd.DataFrame
    importance: pd.DataFrame
    coefficients: pd.DataFrame
    cv: dict
    temporal: dict
    imbalance: dict
    meta: dict


def _engagement_score(x: pd.DataFrame) -> np.ndarray:
    return (5 - x['engagement'].to_numpy(dtype=float)) / 4      # lower engagement → higher score, in [0, 1]


def evaluate(d: pd.DataFrame, seed: int = SEED, test_size: float = .25) -> Evaluation:
    """Leakage-safe evaluation. The holdout set is split off first; scaling, encoding, threshold choice and
    cross-validation use training rows only. Baselines are scored on the same holdout."""
    xtr, xte, ytr, yte = split(d, test_size, seed)
    ytr_a, yte_a = ytr.to_numpy(), yte.to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    specs = {
        'Prior baseline (training turnover rate)': DummyClassifier(strategy='prior'),
        'Engagement-only rule (lower engagement → higher score)': None,
        'Logistic regression, original setup (unscaled)': original_model(),
        'Logistic regression, scaled (app model)': make_model(),
        'Logistic regression, scaled + balanced class weights': make_model(class_weight='balanced'),
    }
    rows, fitted, oof_store, test_store = [], {}, {}, {}
    for name, est in specs.items():
        if est is None:
            oof, p = _engagement_score(xtr), _engagement_score(xte)
            cv_auc = [roc_auc_score(ytr_a[v], oof[v]) for _, v in cv.split(xtr, ytr_a)]
        else:
            oof = cross_val_predict(est, xtr, ytr_a, cv=cv, method='predict_proba')[:, 1]
            cv_auc = [roc_auc_score(ytr_a[v], oof[v]) if len(np.unique(oof[v])) > 1 else .5
                      for _, v in cv.split(xtr, ytr_a)]
            est = est.fit(xtr, ytr_a)
            fitted[name] = est
            p = est.predict_proba(xte)[:, 1]
        t = best_threshold(ytr_a, oof) if len(np.unique(oof)) > 1 else .5
        c = classification(yte_a, p, t)
        auc = roc_auc_score(yte_a, p) if len(np.unique(p)) > 1 else .5
        rows.append({'model': name, 'roc_auc': auc, 'average_precision': average_precision_score(yte_a, p),
                     'brier': brier_score_loss(yte_a, np.clip(p, 0, 1)), 'threshold': t, 'precision': c['precision'],
                     'recall': c['recall'], 'f1': c['f1'], 'cv_auc_mean': float(np.mean(cv_auc)),
                     'cv_auc_std': float(np.std(cv_auc))})
        oof_store[name], test_store[name] = oof, p
    comp = pd.DataFrame(rows)

    app = 'Logistic regression, scaled (app model)'
    model, p = fitted[app], test_store[app]
    t = float(comp.loc[comp.model == app, 'threshold'].iloc[0])
    metrics = {'roc_auc': roc_auc_score(yte_a, p), 'average_precision': average_precision_score(yte_a, p),
               'brier': brier_score_loss(yte_a, p), **classification(yte_a, p, t),
               'default_threshold': classification(yte_a, p, .5),
               'test_rows': int(len(yte_a)), 'train_rows': int(len(ytr_a))}

    curves = {}
    for name in [app, 'Engagement-only rule (lower engagement → higher score)', 'Prior baseline (training turnover rate)']:
        fpr, tpr, _ = roc_curve(yte_a, test_store[name])
        pr, rc, _ = precision_recall_curve(yte_a, test_store[name])
        curves[name] = {'fpr': fpr, 'tpr': tpr, 'precision': pr, 'recall': rc}

    grid = np.round(np.arange(.05, .951, .05), 2)
    thr = pd.DataFrame([classification(yte_a, p, float(g)) for g in grid])

    q = pd.qcut(pd.Series(p), 10, duplicates='drop')
    cal = (pd.DataFrame({'p': p, 'y': yte_a}).groupby(q.values, observed=True)
           .agg(mean_score=('p', 'mean'), observed_rate=('y', 'mean'), records=('y', 'size')).reset_index(drop=True))

    pi = permutation_importance(model, xte, yte_a, scoring='roc_auc', n_repeats=20, random_state=seed)
    imp = (pd.DataFrame({'feature': FEATURES, 'auc_drop': pi.importances_mean, 'std': pi.importances_std})
           .assign(label=lambda x: x.feature.map(LABELS)).sort_values('auc_drop', ascending=False).reset_index(drop=True))

    lr = model[-1]
    names = model[0].get_feature_names_out()
    coef = pd.DataFrame({'term': names, 'coef': lr.coef_[0]})
    num = coef[coef.term.str.startswith('num__')].copy()
    num['feature'] = num.term.str.replace('num__', '', regex=False)
    num['label'] = num.feature.map(LABELS)
    num['odds_ratio_per_sd'] = np.exp(num.coef)
    num['sd'] = [float(xtr[f].std()) for f in num.feature]
    coefs = num[['feature', 'label', 'coef', 'odds_ratio_per_sd', 'sd']].sort_values('coef').reset_index(drop=True)

    cvd = {r.model: (r.cv_auc_mean, r.cv_auc_std) for r in comp.itertuples()}
    imbalance = {'train_rate': float(ytr_a.mean()), 'test_rate': float(yte_a.mean()), 'overall_rate': float(d[TARGET].mean()),
                 'leavers': int(d[TARGET].sum()), 'stayers': int((1 - d[TARGET]).sum()),
                 'ratio': float((1 - d[TARGET]).sum() / max(d[TARGET].sum(), 1))}
    meta = {'seed': seed, 'test_size': test_size, 'rows': int(len(d)), 'features': FEATURES,
            'departments': int(d['department'].nunique())}
    return Evaluation(model, comp, metrics, confusion_matrix(yte_a, p >= t, labels=[0, 1]),
                      confusion_matrix(yte_a, p >= .5, labels=[0, 1]), curves, thr, cal, imp, coefs, cvd,
                      temporal_check(d), imbalance, meta)


def temporal_check(d: pd.DataFrame) -> dict:
    """Train on every year before the latest one and test on the latest year (a harder, time-ordered check)."""
    if YEAR not in d.columns or d[YEAR].nunique() < 2:
        return {}
    last = int(d[YEAR].max())
    tr, te = d[d[YEAR] < last], d[d[YEAR] == last]
    if tr[TARGET].value_counts().min() < 10 or te[TARGET].nunique() < 2 or te[TARGET].value_counts().min() < 5:
        return {}
    m = make_model().fit(tr[FEATURES], tr[TARGET])
    p = m.predict_proba(te[FEATURES])[:, 1]
    eng = _engagement_score(te)
    return {'train_years': f'{int(tr[YEAR].min())}–{last - 1}', 'test_year': last, 'train_rows': int(len(tr)),
            'test_rows': int(len(te)), 'roc_auc': roc_auc_score(te[TARGET], p),
            'average_precision': average_precision_score(te[TARGET], p),
            'engagement_rule_auc': roc_auc_score(te[TARGET], eng), 'prior_ap': float(te[TARGET].mean()),
            'mean_score': float(p.mean()), 'observed_rate': float(te[TARGET].mean()),
            'train_rate': float(tr[TARGET].mean())}


# ---------------------------------------------------------------------------------------------- scenarios
def _apply(x: pd.DataFrame, hours=0, engagement=0, training=0, promotion_wait=0) -> pd.DataFrame:
    x = x.copy()
    x['weekly_hours'] = (x.weekly_hours + hours).clip(0, 168)
    x['engagement'] = (x.engagement + engagement).clip(1, 5)
    x['training_hours'] = (x.training_hours + training).clip(lower=0)
    x['promotion_wait_years'] = (x.promotion_wait_years + promotion_wait).clip(lower=0)
    return x


def scenario(model, d, hours=0, engagement=0, training=0):
    """Original API: mean model score before and after a uniform hypothetical change."""
    x = d[FEATURES].copy()
    base = model.predict_proba(x)[:, 1].mean()
    return float(base), float(model.predict_proba(_apply(x, hours, engagement, training))[:, 1].mean())


def scenario_detail(model, d, hours=0, engagement=0, training=0, promotion_wait=0, departments=None,
                    min_n: int = MIN_GROUP) -> dict:
    """Aggregate scenario: the change is applied to the chosen departments (all if None); results are averages per
    department, with groups under ``min_n`` hidden. Associations only — not intervention effects."""
    x = d[FEATURES].copy()
    mask = x['department'].isin(departments) if departments else pd.Series(True, index=x.index)
    base = model.predict_proba(x)[:, 1]
    x2 = x.copy()
    x2.loc[mask] = _apply(x[mask], hours, engagement, training, promotion_wait)
    scen = model.predict_proba(x2)[:, 1]
    t = pd.DataFrame({'department': x['department'], 'base': base, 'scenario': scen, 'changed': mask})
    g = t.groupby('department').agg(records=('base', 'size'), base=('base', 'mean'), scenario=('scenario', 'mean'),
                                    changed=('changed', 'max')).reset_index()
    g['delta_pts'] = (g.scenario - g.base) * 100
    hidden = int((g.records < min_n).sum())
    return {'base': float(base.mean()), 'scenario': float(scen.mean()), 'observed': float(d[TARGET].mean()),
            'affected_records': int(mask.sum()), 'by_department': g[g.records >= min_n].reset_index(drop=True),
            'hidden': hidden}


def sensitivity(model, d, factor: str, values, departments=None) -> pd.DataFrame:
    """Mean model score as one factor is shifted across ``values`` (other factors unchanged)."""
    key = {'weekly_hours': 'hours', 'engagement': 'engagement', 'training_hours': 'training',
           'promotion_wait_years': 'promotion_wait'}[factor]
    out = [{'shift': v, 'mean_score': scenario_detail(model, d, departments=departments, **{key: v})['scenario']}
           for v in values]
    return pd.DataFrame(out)


def evaluation_summary(ev: Evaluation) -> dict:
    """JSON-friendly summary of the evaluation (aggregate numbers only)."""
    f = lambda v: round(float(v), 4)
    m = ev.metrics
    return {
        'holdout': {k: f(m[k]) for k in ('roc_auc', 'average_precision', 'brier', 'precision', 'recall', 'f1',
                                          'accuracy', 'flagged_share', 'threshold')},
        'holdout_at_0_5': {k: f(v) for k, v in m['default_threshold'].items() if k not in ('tp', 'fp', 'fn', 'tn')},
        'confusion_matrix_at_threshold': ev.cm.tolist(),
        'rows': {'train': m['train_rows'], 'test': m['test_rows']},
        'class_balance': {k: (f(v) if isinstance(v, float) else v) for k, v in ev.imbalance.items()},
        'comparison': [{k: (f(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
                       for r in ev.comparison.to_dict('records')],
        'permutation_importance_auc_drop': {r.feature: f(r.auc_drop) for r in ev.importance.itertuples()},
        'odds_ratio_per_sd': {r.feature: f(r.odds_ratio_per_sd) for r in ev.coefficients.itertuples()},
        'temporal_check': {k: (f(v) if isinstance(v, float) else v) for k, v in ev.temporal.items()},
        'meta': ev.meta,
        'note': 'Synthetic data. Metrics describe this demonstration only and are not evidence of real-world accuracy.',
    }
