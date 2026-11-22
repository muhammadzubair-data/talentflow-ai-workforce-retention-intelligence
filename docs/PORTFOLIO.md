# TalentFlow AI: portfolio and announcement material

Live app: https://talentflow-ai-zubairlabs.streamlit.app
Code: https://github.com/muhammadzubair-data/talentflow-ai-workforce-retention-intelligence

All figures below come from the synthetic demo dataset. They show the method, not real-world accuracy.

---

## 1. Technical case study

**TalentFlow AI: people analytics that answers the organisational question, not the individual one**

**Context.** Most retention dashboards end in a ranked list of "flight-risk" employees. That list is hard to defend:
it raises privacy and fairness questions, and the accuracy behind it is rarely tested honestly. I wanted to show a
different approach: tell leaders *where* retention is weak and *which working conditions move with it*, and be
completely open about how good the model is.

**What I built.** A seven-module Streamlit app on top of a small, tested analytics package:
- **Workforce Command Center:** yearly turnover with uncertainty bands, plus auto-written notes that say when a gap could just be noise.
- **Retention and Department pages:** turnover by tenure, year, department and workplace factor, each with 95% Wilson intervals.
- **Engagement Intelligence:** any two factors as a heatmap, with small cells hidden.
- **What-If Simulator:** shifts hours, engagement, training or promotion wait for chosen departments and shows the change in the model's *average* score.
- **Model Performance:** compares the model against three baselines on one holdout, with ROC and PR curves, a threshold explorer, calibration, permutation importance and a time-ordered check.
- **Data Explorer:** validates messy uploads (header mapping, Yes/No labels, range checks) and returns a plain-language report.

**Engineering decisions**
- **Leakage-safe by construction:** the holdout is split off first; encoding, scaling, cross-validation and threshold choice use training rows only. A test scrambles the holdout features and checks that nothing learned from training changes.
- **Better threshold, same model:** the original version classified at 0.5 and found 31 of 98 leavers in the holdout. Choosing the threshold on training folds (0.24) finds 67 of 98, with F1 rising from 0.437 to 0.530, at the cost of more false alarms. The app shows that trade-off openly.
- **Honest baselines:** the logistic regression reaches ROC-AUC 0.792 against 0.733 for a one-line engagement rule. Class weighting didn't improve ranking and made probabilities worse (Brier 0.184 vs 0.134), so I kept the calibrated model.
- **Drift made visible:** training on 2021–2024 and testing on 2025, the ranking held (AUC 0.804), but the average score was 21.9% against 16.6% observed. That is a real-world warning, shown in the app.
- **Privacy by default:**
  - no individual scores anywhere;
  - groups under 10 records hidden in every chart, table and download;
  - min/max statistics left out;
  - protected and identifier columns ignored;
  - uploads kept only in the browser session.

**Quality.** 59 automated tests (the original 6 untouched), a Playwright browser test covering uploads, session
persistence and downloads, and GitHub Actions on Python 3.11 and 3.12.

**Result.** A people-analytics demo that a cautious HR leader could look at without wincing: useful aggregate insight,
evaluation numbers that are easy to check, and clear limits on what the model can and can't support.

---

## 2. LinkedIn announcement

Most "employee attrition" projects end with a list of people who might quit. I built one that deliberately doesn't.

TalentFlow AI is a workforce analytics app that answers the organisational question: where is retention weakest, and
which working conditions move with it?

What's inside:
- Turnover by department, tenure, year and workplace factor, with uncertainty shown so noise isn't mistaken for signal
- A what-if simulator that reports the model's average response by department (and says clearly it isn't a causal estimate)
- A model performance page that compares the model against simple baselines on data it never saw
- An upload checker that maps messy HR headers and quietly drops protected characteristics like age and gender

A few things I learned on the synthetic data:
→ The default 0.5 cut-off found only 31 of 98 leavers. A threshold picked on training data found 67.
→ A one-line "low engagement" rule already gets ROC-AUC 0.733. The model reaches 0.792, which is useful but not magic.
→ Tested on a later year, the ranking held but the average score ran 5 points too high. Models drift.

Privacy is built in: no individual scores, groups under 10 hidden, uploads never leave the browser session.

Built with Python, scikit-learn, Plotly and Streamlit. All results are from synthetic data.

Live demo: https://talentflow-ai-zubairlabs.streamlit.app
Code: https://github.com/muhammadzubair-data/talentflow-ai-workforce-retention-intelligence

#PeopleAnalytics #HRAnalytics #DataScience #MachineLearning #ResponsibleAI #Streamlit #Python

---

## 3. Upwork portfolio entry (plain text)

Title:
TalentFlow AI: Workforce Analytics and Employee Retention Dashboard (Python, Streamlit, ML)

Description:
A people analytics web app that shows where employee turnover concentrates and which working conditions move with it,
built for aggregate, privacy-safe insight rather than individual "flight-risk" lists.

What it does:
- 7 dashboard modules: command center, retention trends, department comparison, engagement analysis, what-if simulator, model performance and data explorer
- Turnover by department, tenure, year and workplace factor, with 95% confidence intervals
- What-if scenarios showing how the model's average score responds to changes in hours, engagement, training or promotion timing
- Honest model evaluation: holdout testing, comparison with simple baselines, threshold tuning, calibration and a time-based check
- CSV upload with automatic header mapping, data cleaning and a validation report
- Light and dark themes, downloadable aggregate reports

Privacy and responsible AI:
- No individual scores or rankings
- Groups under 10 people hidden everywhere
- Protected characteristics (age, gender, etc.) and ID columns ignored automatically
- Uploaded data stays in the browser session only

Results on synthetic demo data: ROC-AUC 0.792 vs 0.733 for a simple engagement rule; a better threshold raised recall
from 32% to 68%.

Tools: Python, pandas, scikit-learn, Plotly, Streamlit, pytest, Playwright, GitHub Actions

Live demo: https://talentflow-ai-zubairlabs.streamlit.app
Code: https://github.com/muhammadzubair-data/talentflow-ai-workforce-retention-intelligence

---

## 4. Six LinkedIn carousel concepts

1. **"Where, not who"**
   - Slide 1: a ranked "flight-risk" list crossed out, set against a department turnover chart with intervals.
   - Message: people analytics should answer organisational questions.
   - Close on the three pills: aggregate only · groups under 10 hidden · no protected data.
2. **"The 0.5 trap"**
   - Show the confusion matrix at 0.5 (31 of 98 leavers found), then at 0.24 (67 of 98).
   - Explain choosing the threshold on training folds, never on the test set.
   - End with the precision–recall trade-off chart.
3. **"Beat the one-liner"**
   - Prior baseline 0.500 → engagement rule 0.733 → logistic regression 0.792.
   - Point: always show what a simple rule achieves before celebrating a model.
4. **"Models drift"**
   - Train on 2021–2024, test on 2025: ranking holds (AUC 0.804), but the average score is 21.9% vs 16.6% observed.
   - Close with "monitor calibration, not just accuracy".
5. **"What-if ≠ what will happen"**
   - The simulator screenshot (hours −4, engagement +0.3 → mean score 21.7% → 15.3%).
   - Explain association vs causation in plain words, with three reasons the real effect could differ.
6. **"Messy HR data in, clean report out"**
   - Before: the messy CSV headers, with Gender, Age and EmployeeID highlighted.
   - After: the validation report (1,187 of 1,200 rows kept, protected columns ignored, the 5-person team hidden).

---

## 5. ZubairLabs website project description

**TalentFlow AI: Workforce & Retention Intelligence**

A seven-module people-analytics dashboard that shows where turnover concentrates and which working conditions move
with it, without ever scoring individuals. It includes:
- retention trends with uncertainty, department and engagement analysis
- a what-if simulator
- an upload checker that handles messy HR files and ignores protected characteristics
- a model performance page that compares a logistic regression honestly against simple baselines

Privacy is the default: groups under 10 are hidden and uploads stay in the browser session. Built with Python,
scikit-learn, Plotly and Streamlit, with automated tests and CI. Demonstrated on synthetic data.

[Live demo](https://talentflow-ai-zubairlabs.streamlit.app) · [Source code](https://github.com/muhammadzubair-data/talentflow-ai-workforce-retention-intelligence)
