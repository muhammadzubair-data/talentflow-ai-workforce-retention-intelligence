"""Extended tests: data generation, upload validation, privacy suppression, leakage-safe evaluation and scenarios."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from talentflow.core import (FEATURES, MIN_GROUP, TARGET, YEAR, aggregate, band, band_rates, best_threshold,
                             classification, department_profile, evaluate, evaluation_summary, generate, group_rates,
                             heatmap, make_model, prepare, scenario, scenario_detail, sensitivity, split,
                             temporal_check, validate, wilson, yearly)


@pytest.fixture(scope="module")
def demo():
    return generate()


@pytest.fixture(scope="module")
def ev(demo):
    return evaluate(demo)


def original_generate(n=1800, seed=42):
    """Verbatim copy of the original project's generator (used to prove the data did not change)."""
    r = np.random.default_rng(seed)
    d = pd.DataFrame({'department': r.choice(['Operations', 'Engineering', 'Sales', 'Support', 'Finance'], n),
                      'tenure_years': r.uniform(.2, 15, n), 'weekly_hours': r.uniform(30, 60, n),
                      'engagement': r.uniform(1, 5, n), 'training_hours': r.uniform(0, 80, n),
                      'promotion_wait_years': r.uniform(0, 8, n)})
    z = (-1.4 + .08 * (d.weekly_hours - 40) - .65 * (d.engagement - 3) + .18 * d.promotion_wait_years
         - .025 * d.training_hours - .06 * d.tenure_years)
    d['left'] = r.binomial(1, 1 / (1 + np.exp(-z)))
    return d.round(2)


# ------------------------------------------------------------------ data generation
def test_original_columns_unchanged(demo):
    assert demo[FEATURES + [TARGET]].equals(original_generate())


@pytest.mark.parametrize("seed", [1, 7, 123])
def test_original_columns_unchanged_other_seeds(seed):
    assert generate(500, seed)[FEATURES + [TARGET]].equals(original_generate(500, seed))


def test_year_column(demo):
    assert set(demo[YEAR].unique()) == {2021, 2022, 2023, 2024, 2025}
    rates = demo.groupby(YEAR)[TARGET].mean()
    assert rates.max() - rates.min() > .05          # a visible historical pattern
    assert YEAR not in generate(200, with_year=False).columns


def test_generate_deterministic():
    assert generate(900, 5).equals(generate(900, 5))
    assert not generate(900, 5).equals(generate(900, 6))


# ------------------------------------------------------------------ strict validation
def test_validate_keeps_year_and_types(demo):
    v = validate(demo)
    assert YEAR in v.columns and v[TARGET].dtype.kind == 'i'


@pytest.mark.parametrize("mutate", [
    lambda d: d.assign(engagement=d.engagement.where(d.index != 3, 7)),
    lambda d: d.assign(weekly_hours=d.weekly_hours.where(d.index != 3, -1)),
    lambda d: d.assign(left=d.left.where(d.index != 3, 2)),
    lambda d: d.head(80),
    lambda d: d.assign(tenure_years=d.tenure_years.where(d.index != 3, np.nan)),
    lambda d: d.assign(department=d.department.where(d.index != 3, '  ')),
])
def test_validate_rejects(demo, mutate):
    with pytest.raises(ValueError):
        validate(mutate(demo.copy()))


# ------------------------------------------------------------------ forgiving upload preparation
def messy(n=600, seed=11):
    d = generate(n, seed)
    r = np.random.default_rng(3)
    m = pd.DataFrame({'EmployeeID': range(1, n + 1), 'Dept': d.department, 'Years At Company': d.tenure_years.astype(object),
                      'Hours per week': d.weekly_hours, 'Engagement Score': d.engagement, 'Training': d.training_hours,
                      'YearsSinceLastPromotion': d.promotion_wait_years, 'Attrition': d.left.map({1: 'Yes', 0: 'No'}),
                      'Gender': r.choice(['F', 'M'], n), 'Age': r.integers(20, 60, n), 'Fiscal Year': d.year, 'Office': 'HQ'})
    m.loc[:4, 'Engagement Score'] = np.nan
    m.loc[5:7, 'Hours per week'] = 200
    m.loc[8:9, 'Attrition'] = 'maybe'
    m.loc[10, 'Years At Company'] = 'n/a'
    return m, d


def test_prepare_maps_and_cleans():
    m, d = messy()
    c, rep = prepare(m)
    assert list(c.columns) == FEATURES + [TARGET, YEAR]
    assert rep.dropped_missing == 6 and rep.dropped_range == 3 and rep.dropped_label == 2
    assert rep.rows_out == len(m) - 11 == len(c)
    assert rep.renamed['Attrition'] == 'left' and rep.renamed['Fiscal Year'] == 'year'
    assert c[TARGET].isin([0, 1]).all() and set(c.department) == set(d.department)


def test_prepare_excludes_protected_and_identifiers():
    m, _ = messy()
    c, rep = prepare(m.assign(Ethnicity='x', MaritalStatus='Single', Email='a@b.c', Salary=1))
    assert set(rep.ignored_protected) == {'Gender', 'Age', 'Ethnicity', 'MaritalStatus'}
    assert {'EmployeeID', 'Email', 'Salary'} <= set(rep.ignored_identifier)
    assert not ({'Gender', 'Age', 'Ethnicity', 'EmployeeID', 'Email', 'Salary'} & set(c.columns))
    assert rep.notes and 'Protected' in rep.notes[0]


def test_prepare_label_variants():
    d = generate(300, 2)
    for yes, no in [('Yes', 'No'), ('TRUE', 'FALSE'), ('left', 'stayed'), (1, 0)]:
        c, _ = prepare(d.assign(left=d.left.map({1: yes, 0: no})))
        assert c[TARGET].sum() == d[TARGET].sum()


def test_prepare_errors():
    d = generate(300, 2)
    with pytest.raises(ValueError, match='Missing required columns: engagement'):
        prepare(d.drop(columns='engagement'))
    with pytest.raises(ValueError, match='At least 100 rows'):
        prepare(d.head(60))
    with pytest.raises(ValueError, match='At least 100 rows'):
        prepare(d.assign(left=0))
    with pytest.raises(ValueError, match='departments'):
        prepare(d.assign(department=[f'D{i}' for i in range(len(d))]))
    with pytest.raises(ValueError, match='limit'):
        prepare(d, max_rows=100)


def test_prepare_bad_year_dropped():
    d = generate(300, 2)
    c, rep = prepare(d.assign(year=d.year.astype(object).where(d.index != 5, 'soon')))
    assert YEAR not in c.columns and not rep.has_year and rep.notes


# ------------------------------------------------------------------ aggregates and privacy suppression
def test_wilson_interval():
    lo, hi = wilson(20, 100)
    assert lo < .2 < hi and abs(lo - .1333) < 1e-3 and abs(hi - .2888) < 1e-3


def test_bands_cover_all_records(demo):
    for col in ['tenure_years', 'weekly_hours', 'engagement', 'training_hours', 'promotion_wait_years']:
        assert band(demo, col).notna().all(), col


def test_group_suppression():
    d = generate(300, 2)
    d.loc[:6, 'department'] = 'Tiny'
    g, hidden = group_rates(d, 'department')
    assert hidden == 1 and 'Tiny' not in set(g.department)
    assert 'Tiny' not in set(aggregate(d).department)
    g50, hidden50 = group_rates(d, 'department', min_n=70)
    assert (g50.records >= 70).all() and hidden50 >= 1


def test_heatmap_masks_small_cells(demo):
    rate, n, hidden = heatmap(demo.head(300), 'engagement', 'weekly_hours', 10)
    assert rate.where(n < 10).isna().all().all()
    assert hidden == int(((n > 0) & (n < 10)).sum().sum())


def test_rates_match_counts(demo):
    g, _ = group_rates(demo, 'department')
    assert g.records.sum() == len(demo)
    assert np.allclose(g.set_index('department').turnover, demo.groupby('department')[TARGET].mean().loc[g.department])
    b, _ = band_rates(demo, 'engagement')
    assert b.records.sum() == len(demo)
    y, _ = yearly(demo)
    assert y.records.sum() == len(demo)


def test_department_profile_scores_are_group_means(demo, ev):
    dp, _ = department_profile(demo, ev.model)
    assert len(dp) == demo.department.nunique()       # one row per department, never per person
    full = pd.Series(ev.model.predict_proba(demo[FEATURES])[:, 1]).groupby(demo.department).mean()
    assert np.allclose(dp.set_index('department').mean_model_score, full.loc[dp.department])


# ------------------------------------------------------------------ model evaluation
def test_evaluation_metrics_are_recomputable(demo, ev):
    xtr, xte, ytr, yte = split(demo)
    p = ev.model.predict_proba(xte)[:, 1]
    assert ev.metrics['roc_auc'] == pytest.approx(roc_auc_score(yte, p))
    assert ev.metrics['average_precision'] == pytest.approx(average_precision_score(yte, p))
    assert ev.cm.sum() == len(yte) == ev.metrics['test_rows']
    c = classification(yte.to_numpy(), p, ev.metrics['threshold'])
    assert c['f1'] == pytest.approx(ev.metrics['f1'])


def test_holdout_is_disjoint_and_stratified(demo):
    xtr, xte, ytr, yte = split(demo)
    assert not set(xtr.index) & set(xte.index)
    assert abs(ytr.mean() - yte.mean()) < .01


def test_threshold_uses_training_rows_only(demo, ev):
    xtr, xte, ytr, yte = split(demo)
    oof = cross_val_predict(make_model(), xtr, ytr.to_numpy(), cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            method='predict_proba')[:, 1]
    assert ev.metrics['threshold'] == best_threshold(ytr.to_numpy(), oof)
    # Scrambling holdout features (the split depends only on labels) must not change anything learned from training.
    d2 = demo.copy()
    rng = np.random.default_rng(0)
    for col in ['engagement', 'weekly_hours', 'training_hours']:
        d2.loc[xte.index, col] = rng.permutation(d2.loc[xte.index, col].to_numpy())
    e2 = evaluate(d2)
    assert e2.metrics['threshold'] == ev.metrics['threshold']
    assert np.allclose(e2.model[-1].coef_, ev.model[-1].coef_)
    assert e2.metrics['roc_auc'] < ev.metrics['roc_auc'] - .1


def test_model_beats_baselines(ev):
    c = ev.comparison.set_index('model')
    app = c.loc['Logistic regression, scaled (app model)']
    assert c.loc['Prior baseline (training turnover rate)', 'roc_auc'] == .5
    assert app.roc_auc > c.loc['Engagement-only rule (lower engagement → higher score)', 'roc_auc'] + .03
    assert app.average_precision > c.loc['Prior baseline (training turnover rate)', 'average_precision'] + .2
    assert app.brier < c.loc['Logistic regression, scaled + balanced class weights', 'brier']
    assert len(c) == 5 and c.cv_auc_std.max() < .1


def test_tuned_threshold_improves_recall(ev):
    assert ev.metrics['recall'] > ev.metrics['default_threshold']['recall'] + .2
    assert ev.metrics['f1'] > ev.metrics['default_threshold']['f1']


def test_threshold_table_monotone_recall(ev):
    assert (np.diff(ev.thresholds.recall) <= 1e-12).all()


def test_importance_and_coefficients(ev):
    assert ev.importance.iloc[0].feature == 'engagement'
    assert abs(ev.importance.set_index('feature').auc_drop['department']) < .02
    oc = ev.coefficients.set_index('feature').odds_ratio_per_sd
    assert oc['weekly_hours'] > 1 and oc['promotion_wait_years'] > 1
    assert oc['engagement'] < 1 and oc['training_hours'] < 1 and oc['tenure_years'] < 1


def test_calibration_and_imbalance(ev):
    assert ev.calibration.records.sum() == ev.metrics['test_rows']
    assert abs(ev.calibration.mean_score.mean() - ev.calibration.observed_rate.mean()) < .05
    assert ev.imbalance['leavers'] + ev.imbalance['stayers'] == 1800


def test_temporal_check(demo, ev):
    tc = ev.temporal
    assert tc['test_year'] == 2025 and tc['train_rows'] + tc['test_rows'] == len(demo)
    assert .6 < tc['roc_auc'] < .95
    assert temporal_check(demo.drop(columns=YEAR)) == {}


def test_evaluation_deterministic(demo, ev):
    assert evaluate(demo).metrics == ev.metrics


def test_summary_is_json(ev):
    s = json.loads(json.dumps(evaluation_summary(ev), default=str))
    assert s['holdout']['roc_auc'] == round(ev.metrics['roc_auc'], 4) and 'Synthetic' in s['note']


# ------------------------------------------------------------------ scenarios
def test_scenario_directions(demo, ev):
    b, s = scenario(ev.model, demo)
    assert b == s
    assert scenario(ev.model, demo, engagement=.5)[1] < b
    assert scenario(ev.model, demo, hours=5)[1] > b
    assert scenario(ev.model, demo, training=20)[1] < b


def test_scenario_detail_matches_original(demo, ev):
    r = scenario_detail(ev.model, demo, hours=3, engagement=-.2, training=10)
    b, s = scenario(ev.model, demo, 3, -.2, 10)
    assert r['base'] == pytest.approx(b) and r['scenario'] == pytest.approx(s)
    assert r['affected_records'] == len(demo)


def test_scenario_scope_only_changes_selected(demo, ev):
    r = scenario_detail(ev.model, demo, engagement=1, departments=['Sales'])
    g = r['by_department'].set_index('department')
    assert g.loc['Sales', 'delta_pts'] < -1
    assert (g.drop('Sales').delta_pts.abs() < 1e-9).all()
    assert r['affected_records'] == (demo.department == 'Sales').sum()


def test_scenario_clipping(demo, ev):
    r = scenario_detail(ev.model, demo, engagement=10, hours=-500, training=-500, promotion_wait=-50)
    assert 0 <= r['scenario'] <= 1


def test_sensitivity_monotone(demo, ev):
    s = sensitivity(ev.model, demo, 'weekly_hours', [-6, -3, 0, 3, 6])
    assert s.mean_score.is_monotonic_increasing


def test_upload_dataset_evaluates():
    m, _ = messy(800, 4)
    c, _ = prepare(m)
    e = evaluate(c)
    assert .6 < e.metrics['roc_auc'] < 1 and e.temporal


def test_min_group_constant():
    assert MIN_GROUP == 10
