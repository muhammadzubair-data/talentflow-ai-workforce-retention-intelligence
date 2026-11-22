# Model card: TalentFlow AI turnover model

| | |
|:--|:--|
| **Model** | Logistic regression (L2, C = 1, `max_iter=2000`) on one-hot `department` + standardised numeric features |
| **Version** | 2.0.0 (`talentflow/core.py`) |
| **Owner** | ZubairLabs, Muhammad Zubair |
| **Intended use** | Demonstrating aggregate people analytics: how turnover associates with workplace factors, and how an honest evaluation looks |
| **Not intended for** | Any decision about an individual, including hiring, firing, promotion, pay, discipline, surveillance or "flight-risk" lists |
| **Data** | Synthetic by default (1,800 employee-year records, seed 42); optionally an anonymised upload held in session memory |

## Inputs

`department`, `tenure_years`, `weekly_hours`, `engagement` (1–5), `training_hours` and `promotion_wait_years`.
Protected characteristics (age, gender, ethnicity, religion, marital status, disability, nationality and similar),
identifiers and pay columns are excluded by the upload validator and never reach the model.

## Training and evaluation

- Stratified 75/25 holdout (seed 42). The holdout is split off before any fitting and scored once.
- Encoding and scaling are fitted inside the pipeline on training rows only.
- 5-fold stratified cross-validation on training rows gives the out-of-fold scores, and the F1-maximising decision
  threshold is chosen from them.
- Baselines on the same holdout:
  - the training-rate prior
  - an engagement-only rule
  - the original unscaled pipeline
  - a class-balanced variant
- Time-ordered check: train on 2021–2024, test on 2025.

## Results on the synthetic demo (holdout, 450 rows, base rate 0.218)

| Metric | Value |
|:--|--:|
| ROC-AUC | 0.792 (CV 0.793 ± 0.028) |
| Average precision | 0.555 |
| Brier score | 0.134 |
| Threshold (from training folds) | 0.24 |
| Precision / recall / F1 | 0.432 / 0.684 / 0.530 |
| At threshold 0.5 | 0.705 / 0.316 / 0.437 |
| Engagement-only rule ROC-AUC | 0.733 |
| Time-ordered ROC-AUC (2025) | 0.804; mean score 21.9% vs observed 16.6% |

Permutation importance ranks engagement first, then weekly hours and training hours; department adds nothing,
matching the generator, where department has no effect.

## Ethical considerations

- **Synthetic labels:** these numbers say nothing about real-world predictability of turnover.
- **Aggregate outputs only:**
  - model scores are averaged per department or organisation;
  - groups under 10 records are hidden everywhere.
- **No fairness audit:** one is not possible without protected attributes, which are deliberately excluded. A real
  deployment would need a governed audit before any use.
- **Scenarios are associations:**
  - the "what-if" changes inputs and reads the model's response;
  - it does not estimate the effect of a policy;
  - confounding, selection and feedback are not modelled.
- **Drift:** the time-ordered check shows ranking can hold while calibration drifts. Real use would need monitoring and
  recalibration.

## Caveats

- Logistic regression assumes linear log-odds and models no interactions.
- Synthetic features are independent; real ones are correlated.
- Wilson intervals carry no multiple-comparison correction.
- Small-group suppression reduces, but does not eliminate, re-identification risk.
