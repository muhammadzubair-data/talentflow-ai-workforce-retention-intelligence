"""Browser end-to-end check of the running TalentFlow dashboard (Playwright + Chromium).

Not collected by pytest (file name). Usage:
    streamlit run app.py --server.port 8501 &
    python tests/e2e_ui.py http://localhost:8501 [--shots DIR] [--chrome /path/to/chrome]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from talentflow.core import generate  # noqa: E402

PAGES = [("", "command"), ("retention", "retention"), ("departments", "departments"), ("engagement", "engagement"),
         ("simulator", "simulator"), ("model", "model"), ("data", "data")]
TITLES = {"command": "Workforce Command Center", "retention": "Retention Intelligence",
          "departments": "Department Analytics", "engagement": "Engagement Intelligence",
          "simulator": "What-If Simulator", "model": "Model Performance & Methodology", "data": "Data Explorer"}
MIN_CHARTS = {"command": 4, "retention": 7, "departments": 4, "engagement": 5, "simulator": 3, "model": 6, "data": 1}


def idle(page, extra=2.0):
    page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=180_000)
    time.sleep(1.5)
    for _ in range(600):
        if page.locator("[data-testid='stSpinner']").count() + page.locator("[data-testid='stStatusWidget']").count() == 0:
            break
        time.sleep(0.5)
    time.sleep(extra)


def check(page, name, failures, min_charts=0):
    exc = page.locator("[data-testid='stException']").count()
    errs = [e.inner_text()[:160] for e in page.locator("[data-testid='stAlertContentError']").all()]
    charts = page.locator(".js-plotly-plot").count()
    title_ok = TITLES[name] in page.locator(".tf-hero h1").first.inner_text()
    print(f"  {name:<12} exceptions={exc} errors={len(errs)} charts={charts} title={'ok' if title_ok else 'MISSING'}")
    if exc or errs or charts < min_charts or not title_ok:
        failures.append((name, exc, errs, charts, title_ok))


def kpi(page, i):
    return page.locator("[data-testid='stMain'] .kpi .value").nth(i).inner_text()


def messy_csv(path: Path) -> None:
    d = generate(1200, 11)
    r = np.random.default_rng(3)
    m = pd.DataFrame({'EmployeeID': range(1, 1201), 'Dept': d.department, 'Years At Company': d.tenure_years.astype(object),
                      'Hours per week': d.weekly_hours, 'Engagement Score': d.engagement, 'Training': d.training_hours,
                      'YearsSinceLastPromotion': d.promotion_wait_years, 'Attrition': d.left.map({1: 'Yes', 0: 'No'}),
                      'Gender': r.choice(['F', 'M'], 1200), 'Age': r.integers(20, 60, 1200), 'Fiscal Year': d.year})
    m.loc[:5, 'Engagement Score'] = np.nan
    m.loc[6:9, 'Hours per week'] = 200
    m.loc[10:12, 'Attrition'] = 'maybe'
    m.loc[13:17, 'Dept'] = 'Legal'
    m.to_csv(path, index=False)


def sidebar_go(page, title):
    page.locator("[data-testid='stSidebarNav'] a", has_text=title).first.click()
    idle(page)


def download(page, label) -> bytes:
    with page.expect_download() as dl:
        page.get_by_role("button", name=label).first.click()
    return Path(dl.value.path()).read_bytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="http://localhost:8501")
    ap.add_argument("--shots")
    ap.add_argument("--chrome", default=os.environ.get("CHROME_PATH"))
    a = ap.parse_args()
    base = a.url.rstrip("/")
    failures: list = []
    with sync_playwright() as p:
        launch = {"args": ["--no-sandbox"]}
        if a.chrome:
            launch["executable_path"] = a.chrome
        browser = p.chromium.launch(**launch)
        for theme in (["light", "dark"] if a.shots else ["light"]):
            print(f"Pages ({theme}):")
            page = browser.new_page(viewport={"width": 1440, "height": 900}, color_scheme=theme)
            for path, name in PAGES:
                page.goto(f"{base}/{path}", wait_until="networkidle")
                idle(page)
                check(page, name, failures, MIN_CHARTS[name])
                if a.shots:
                    Path(a.shots).mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=f"{a.shots}/{name}-{theme}.png")
            page.close()

        print("Interactions:")
        page = browser.new_page(viewport={"width": 1440, "height": 900}, accept_downloads=True)

        # What-if simulator: raise engagement with the keyboard, the scenario score must fall.
        page.goto(f"{base}/simulator", wait_until="networkidle")
        idle(page)
        before = kpi(page, 1)
        page.locator("[data-testid='stMain'] [data-testid='stSlider']").nth(1).get_by_role("slider").focus()
        for _ in range(5):
            page.keyboard.press("ArrowRight")
        idle(page, 3)
        after = kpi(page, 1)
        print(f"  simulator    scenario score {before} -> {after}")
        if not float(after.rstrip('%')) < float(before.rstrip('%')):
            failures.append(("simulator", before, after))
        csv = pd.read_csv(io.BytesIO(download(page, "Download scenario summary (CSV)")))
        print(f"  simulator    scenario CSV rows={len(csv)}")
        if len(csv) != 6 or "ALL (organisation)" not in set(csv.department):
            failures.append(("scenario csv", len(csv)))

        # Model page: JSON export parses and matches the KPI.
        page.goto(f"{base}/model", wait_until="networkidle")
        idle(page)
        summary = json.loads(download(page, "Download evaluation summary (JSON)"))
        auc = kpi(page, 0)
        print(f"  model        ROC-AUC kpi={auc} json={summary['holdout']['roc_auc']}")
        if abs(float(auc) - summary["holdout"]["roc_auc"]) > 0.001:
            failures.append(("model json", auc))

        # Upload a messy file, check validation, then move between pages with the sidebar (same session).
        page.goto(f"{base}/data", wait_until="networkidle")
        idle(page)
        tmp = Path(tempfile.mkdtemp()) / "messy_hr.csv"
        messy_csv(tmp)
        page.locator("[data-testid='stFileUploader'] input[type='file']").set_input_files(str(tmp))
        idle(page, 4)
        body = page.locator("[data-testid='stMain']").inner_text()
        ok = "messy_hr.csv" in body and "Gender, Age" in body and "Dept → department" in body
        print(f"  upload       accepted={'messy_hr.csv' in body} validation_report={ok}")
        if not ok:
            failures.append(("upload", body[:400]))
        if a.shots:
            page.screenshot(path=f"{a.shots}/upload-light.png")
        dept = pd.read_csv(io.BytesIO(download(page, "Department summary")))
        print(f"  upload       department summary rows={len(dept)} min records={dept.records.min()}")
        if "Legal" in set(dept.department) or dept.records.min() < 10 or len(dept) != 5:
            failures.append(("department csv", dept.department.tolist()))
        for title in ["Department Analytics", "Model Performance", "Workforce Command Center"]:
            sidebar_go(page, title)
            uploaded = page.locator(".pill.ok").count() > 0
            print(f"  session      {title:<26} uses upload={uploaded}")
            if not uploaded:
                failures.append(("session", title))
        sidebar_go(page, "Data Explorer")
        page.get_by_role("button", name="Remove upload").click()
        idle(page)
        back = page.locator(".pill.warn").count() > 0
        print(f"  upload       removed, back to synthetic={back}")
        if not back:
            failures.append(("remove upload",))

        # Errors are friendly: a file without the outcome column.
        bad = tmp.with_name("bad.csv")
        generate(300).drop(columns="left").to_csv(bad, index=False)
        page.locator("[data-testid='stFileUploader'] input[type='file']").set_input_files(str(bad))
        idle(page, 3)
        err = page.locator("[data-testid='stAlertContentError']").all_inner_texts()
        print(f"  upload       bad file error shown={bool(err)}")
        if not err or "Missing required columns: left" not in err[0]:
            failures.append(("bad upload", err))
        browser.close()
    print("FAILURES:" if failures else "All browser checks passed.", *failures, sep="\n  ")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
