"""Streamlit AppTest checks: every page renders, the upload path works and settings take effect."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from talentflow.core import generate, prepare

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["", "retention", "departments", "engagement", "simulator", "model", "data"]


def run(page="", **state):
    at = AppTest.from_file(APP, default_timeout=180)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if page:
        at._page_hash = next(h for h, i in at._registered_pages.items() if i["url_pathname"] == page)
        at.run()
    return at


def text(at):
    return " ".join(m.value for m in at.markdown)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    assert "Synthetic demonstration data" in text(at)


def test_upload_session_used_everywhere():
    clean, rep = prepare(generate(600, 9).assign(Gender="F"))
    state = dict(upload_df=clean, upload_report=rep, upload_name="team.csv", upload_key="k1", use_upload=True)
    for page in ["", "departments", "model", "data"]:
        at = run(page, **state)
        assert not at.exception, [e.value for e in at.exception]
        assert "Your uploaded file" in text(at)
    at = run("data", **state)
    assert any("team.csv" in s.value for s in at.success)


def test_privacy_threshold_hides_small_departments():
    d = generate(600, 9)
    d.loc[:14, "department"] = "Tiny"                   # 15 records
    clean, rep = prepare(d)
    at = run("departments", upload_df=clean, upload_report=rep, upload_name="t.csv", upload_key="k2", use_upload=True)
    assert "Tiny" in set(at.dataframe[0].value.department)
    at.sidebar.select_slider[0].set_value(20).run()
    assert not at.exception
    assert "Tiny" not in set(at.dataframe[0].value.department)
    assert any("hidden to protect privacy" in c.value for c in at.caption)


def test_simulator_reacts_to_sliders():
    at = run("simulator")
    before = text(at)
    at.slider[1].set_value(0.5).run()          # engagement +0.5
    after = text(at)
    assert not at.exception and before != after and "lower" in after


def test_original_import_shim():
    import core  # noqa: F401  (top-level module from the original project)
    from core import aggregate, generate as g, scenario, train, validate  # noqa: F401
    assert len(validate(g())) == 1800
