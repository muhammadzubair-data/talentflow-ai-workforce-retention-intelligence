"""TalentFlow AI — Workforce Analytics & Employee Retention Intelligence (Streamlit).

Aggregate people analytics on synthetic demonstration data. Run: streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from talentflow.core import (BANDS, band, FEATURES, LABELS, MIN_GROUP, NUMERIC, TARGET, YEAR, band_rates, department_profile,
                             evaluate, evaluation_summary, generate, group_rates, heatmap, prepare, scenario_detail,
                             sensitivity, yearly)

ASSETS = Path(__file__).parent / "assets"
WEBSITE = "https://zubairlabs.com"
REPO = "https://github.com/muhammadzubair-data/talentflow-ai-workforce-retention-intelligence"
NAVY, EMERALD, CYAN, VIOLET, AMBER, ROSE, SLATE, BLUE = ("#0B1B3B", "#10B981", "#06B6D4", "#8B5CF6", "#F59E0B",
                                                        "#F43F5E", "#94A3B8", "#3B82F6")
SEQ = [EMERALD, CYAN, VIOLET, AMBER, BLUE, ROSE, "#84CC16", "#EC4899", "#F97316", "#64748B", "#FACC15", "#14B8A6"]
RISK_SCALE = [[0, "#10B981"], [.5, "#FBBF24"], [1, "#F43F5E"]]
MAX_UPLOAD_MB = 10

st.set_page_config(page_title="TalentFlow AI | ZubairLabs", page_icon=str(ASSETS / "icon.svg"), layout="wide",
                   initial_sidebar_state="auto",
                   menu_items={"About": "TalentFlow AI by ZubairLabs — aggregate workforce analytics on synthetic data. "
                                        "Not for individual employment decisions.",
                               "Report a bug": REPO + "/issues"})

st.markdown(f"""<style>
.block-container {{max-width: 1440px; padding-top: 3.2rem; padding-bottom: 3rem;}}
.tf-hero {{padding: 24px 28px; border-radius: 20px; margin-bottom: 16px; color: #F8FAFC;
  background: radial-gradient(760px 260px at 0% 0%, rgba(16,185,129,.30), transparent 60%),
              radial-gradient(640px 260px at 100% 0%, rgba(6,182,212,.22), transparent 62%),
              radial-gradient(700px 300px at 100% 100%, rgba(139,92,246,.34), transparent 60%),
              linear-gradient(120deg, #06112A 0%, #0B1B3B 55%, #1A1D4A 100%);
  border: 1px solid rgba(148,163,184,.22); box-shadow: 0 12px 32px rgba(2,8,23,.22);}}
.tf-hero .eyebrow {{letter-spacing: .2em; font-size: 11px; font-weight: 700; color: {EMERALD}; text-transform: uppercase;}}
.tf-hero h1 {{font-size: 32px; margin: 6px 0 4px 0; padding: 0; color: #F8FAFC; font-weight: 800;}}
.tf-hero p {{margin: 0; color: #CBD5E1; max-width: 980px;}}
.pill {{display: inline-block; margin: 10px 6px 0 0; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18); color: #E2E8F0;}}
.pill.warn {{background: rgba(245,158,11,.18); border-color: rgba(245,158,11,.5); color: #FDE68A;}}
.pill.ok {{background: rgba(16,185,129,.18); border-color: rgba(16,185,129,.5); color: #A7F3D0;}}
.pill.vio {{background: rgba(139,92,246,.22); border-color: rgba(139,92,246,.55); color: #DDD6FE;}}
.kpi-grid {{display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 6px 0 14px 0;}}
.kpi {{border-radius: 16px; padding: 14px 16px; border: 1px solid rgba(148,163,184,.28); position: relative; overflow: hidden;
  background: linear-gradient(160deg, rgba(16,185,129,.10), rgba(6,182,212,.05) 55%, rgba(139,92,246,.08));}}
.kpi:before {{content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
  background: linear-gradient(180deg, {EMERALD}, {CYAN}, {VIOLET});}}
.kpi .label {{font-size: 11.5px; font-weight: 700; opacity: .7; text-transform: uppercase; letter-spacing: .06em;}}
.kpi .value {{font-size: 26px; font-weight: 800; margin-top: 4px; line-height: 1.1;}}
.kpi .sub {{font-size: 12px; opacity: .65; margin-top: 4px;}}
.sec-title {{font-size: 24px; font-weight: 800; margin: 2px 0;}}
.sec-sub {{opacity: .7; margin-bottom: 14px; font-size: 14px;}}
.note {{border-left: 4px solid {EMERALD}; padding: 10px 14px; border-radius: 10px; background: rgba(16,185,129,.09); margin: 6px 0 12px 0;}}
.note.warn {{border-left-color: {AMBER}; background: rgba(245,158,11,.10);}}
.note.vio {{border-left-color: {VIOLET}; background: rgba(139,92,246,.10);}}
.insight {{border-radius: 14px; padding: 12px 16px; margin-bottom: 10px; border: 1px solid rgba(148,163,184,.28);
  background: linear-gradient(90deg, rgba(6,182,212,.10), rgba(139,92,246,.05));}}
.insight b {{font-size: 15px;}}
.insight .why {{opacity: .78; font-size: 13.2px; margin-top: 3px;}}
.footer {{opacity: .6; font-size: 12px; text-align: center; margin-top: 28px;}}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def kpis(items):
    e = html.escape
    st.markdown("<div class='kpi-grid'>" + "".join(
        f"<div class='kpi'><div class='label'>{e(a)}</div><div class='value'>{e(b)}</div><div class='sub'>{e(c)}</div></div>"
        for a, b, c in items) + "</div>", unsafe_allow_html=True)


def section(title, sub):
    st.markdown(f"<div class='sec-title'>{html.escape(title)}</div><div class='sec-sub'>{html.escape(sub)}</div>",
                unsafe_allow_html=True)


def note(text, kind=""):
    st.markdown(f"<div class='note {kind}'>{text}</div>", unsafe_allow_html=True)


def chart(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="", title_font=dict(size=15))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object or str(out[c].dtype).startswith(("string", "category")):
            out[c] = out[c].astype(str).map(lambda v: "'" + v if v[:1] in ("=", "+", "-", "@") else v)
    return out.to_csv(index=False).encode()


def rnd(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    num = df.select_dtypes("number").columns
    return df.assign(**{c: df[c].round(n) for c in num})


def pct(v, d=1):
    return f"{v * 100:.{d}f}%"


def hidden_note(n, what="groups"):
    if n:
        st.caption(f"🔒 {n} {what} with fewer than {min_n} records are hidden to protect privacy.")


def esc(s):
    return html.escape(str(s))


PCT = st.column_config.ProgressColumn
COLS = {"turnover": PCT("turnover", min_value=0.0, max_value=1.0, format="percent"),
        "share": PCT("share", min_value=0.0, max_value=1.0, format="percent"),
        "ci_low": st.column_config.NumberColumn("95% CI low", format="percent"),
        "ci_high": st.column_config.NumberColumn("95% CI high", format="percent"),
        "mean_model_score": st.column_config.NumberColumn("mean model score", format="percent"),
        "vs_org_pts": st.column_config.NumberColumn("vs org (pts)", format="%+.1f")}


# ------------------------------------------------------------------ cached computation (demo only; uploads stay in session)
@st.cache_data(show_spinner=False, max_entries=4)
def load_demo(n: int, seed: int) -> pd.DataFrame:
    return generate(n, seed)


@st.cache_resource(show_spinner=False, max_entries=4, ttl=3600)
def demo_eval(n: int, seed: int):
    return evaluate(load_demo(n, seed), seed=42)


def upload_eval():
    """Upload results are kept in this browser session only — never in a shared cache."""
    if ss.get("upload_eval_key") != ss.get("upload_key"):
        ss["upload_eval"] = evaluate(ss["upload_df"])
        ss["upload_eval_key"] = ss["upload_key"]
    return ss["upload_eval"]


# ------------------------------------------------------------------ sidebar: data source and settings
ss = st.session_state
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), link=WEBSITE, size="large")

with st.sidebar:
    st.markdown("**Data source**")
    has_upload = "upload_df" in ss
    options = ["Synthetic demo"] + ([f"Uploaded: {ss['upload_name']}"] if has_upload else [])
    # The choice lives in plain session state; the radio mirrors it (widget state can reset on page switches).
    use_upload = bool(has_upload and ss.get("use_upload"))
    ss["source_choice"] = options[-1] if use_upload else options[0]

    def _pick_source():
        ss["use_upload"] = ss["source_choice"] != "Synthetic demo"

    st.radio("Data source", options, key="source_choice", on_change=_pick_source, label_visibility="collapsed")
    if not use_upload:
        n_rows = st.select_slider("Synthetic records", [900, 1800, 3600, 7200], value=1800,
                                  help="Employee-year records to generate. 1,800 matches the original project.")
        seed = st.number_input("Seed", 0, 9999, 42, help="Same seed, same data and results.")
    if not has_upload:
        st.caption("Have an anonymised CSV? Upload it on **Data Explorer**.")
    min_n = st.select_slider("Smallest group shown", [10, 15, 20, 30, 50], value=MIN_GROUP,
                             help="Any department, band or cell with fewer records is hidden everywhere. "
                                  "The minimum is 10 and cannot be lowered.")
    with st.expander("Responsible use", icon="🛡️"):
        st.markdown("- Aggregate insights only: no individual scores, rankings or lists.\n"
                    "- No protected characteristics are used; uploaded ones are ignored.\n"
                    "- Synthetic results are **not** evidence of real-world accuracy.\n"
                    "- Do not use for hiring, firing, promotion or pay decisions.\n"
                    "- Scenario results are model associations, not causal effects.")
    st.caption("Theme: ⋮ menu (top right) → Light, Dark or System")
    st.caption(f"[ZubairLabs]({WEBSITE}) · [Source code]({REPO})")

if use_upload:
    data = ss["upload_df"]
    source_label, is_demo = ss["upload_name"], False
else:
    data = load_demo(int(n_rows), int(seed))
    source_label, is_demo = f"Synthetic demo · {len(data):,} records · seed {int(seed)}", True
has_year = YEAR in data.columns


def get_eval():
    with st.spinner("Training and evaluating the model…"):
        return demo_eval(int(n_rows), int(seed)) if is_demo else upload_eval()


def hero(eyebrow, title, text):
    pills = ("<span class='pill warn'>Synthetic demonstration data</span>" if is_demo
             else "<span class='pill ok'>Your uploaded file · this session only</span>")
    pills += ("<span class='pill vio'>Aggregate insights only</span>"
              "<span class='pill'>Not for individual employment decisions</span>")
    st.markdown(f"<div class='tf-hero'><div class='eyebrow'>{esc(eyebrow)}</div><h1>{esc(title)}</h1>"
                f"<p>{esc(text)}</p>{pills}</div>", unsafe_allow_html=True)


def footer():
    st.markdown(f"<div class='footer'>TalentFlow AI · built by <a href='{WEBSITE}' target='_blank'>ZubairLabs</a> · "
                f"{esc(source_label)} · groups under {min_n} records hidden</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ pages
def page_command():
    hero("ZubairLabs · People analytics", "Workforce Command Center",
         "Organisation-level view of headcount records, turnover and engagement. Every figure is an aggregate; "
         "groups smaller than the privacy threshold are hidden.")
    d = data
    rate = d[TARGET].mean()
    kpis([("Records", f"{len(d):,}", "employee-year records"),
          ("Observed turnover", pct(rate), f"{int(d[TARGET].sum()):,} leavers"),
          ("Retention", pct(1 - rate), "stayed through the period"),
          ("Mean engagement", f"{d.engagement.mean():.2f} / 5", f"median {d.engagement.median():.2f}"),
          ("Mean tenure", f"{d.tenure_years.mean():.1f} yrs", f"weekly hours {d.weekly_hours.mean():.1f}"),
          ("Departments", f"{d.department.nunique()}", f"years {int(d[YEAR].min())}–{int(d[YEAR].max())}" if has_year else "no year column")])

    c1, c2 = st.columns([1.35, 1])
    with c1:
        if has_year:
            y, hid = yearly(d, min_n)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(y[YEAR]) + list(y[YEAR])[::-1], y=list(y.ci_high) + list(y.ci_low)[::-1],
                                     fill="toself", fillcolor="rgba(6,182,212,.16)", line=dict(width=0), mode="lines",
                                     hoverinfo="skip", name="95% interval"))
            fig.add_trace(go.Scatter(x=y[YEAR], y=y.turnover, mode="lines+markers+text", name="Turnover",
                                     line=dict(color=EMERALD, width=3), marker=dict(size=9),
                                     text=[pct(v) for v in y.turnover], textposition="top center",
                                     customdata=y.records, hovertemplate="%{x}: %{y:.1%} of %{customdata} records<extra></extra>"))
            fig.add_hline(y=rate, line_dash="dot", line_color=SLATE, annotation_text=f"overall {pct(rate)}")
            fig.update_layout(title="Historical turnover by year", yaxis_tickformat=".0%", xaxis=dict(dtick=1),
                              showlegend=False)
            chart(fig)
            hidden_note(hid, "years")
        else:
            empty_state = "This file has no year column, so the yearly trend is not available."
            st.info(empty_state, icon="📅")
    with c2:
        comp, hid = group_rates(d, "department", min_n)
        fig = px.pie(comp, names="department", values="records", hole=.58, color_discrete_sequence=SEQ,
                     title="Workforce composition by department")
        fig.update_traces(textinfo="percent", textposition="inside", insidetextfont=dict(color="white"),
                          hovertemplate="%{label}: %{value} records (%{percent})<extra></extra>")
        fig.update_layout(legend=dict(orientation="v", x=1.02, y=.5))
        chart(fig)
        hidden_note(hid, "departments")

    c3, c4 = st.columns(2)
    with c3:
        t, hid = band_rates(d, "tenure_years", min_n)
        fig = px.bar(t, x="band", y="records", color="turnover", color_continuous_scale=RISK_SCALE,
                     title="Tenure mix (colour = turnover rate)", text="records",
                     labels={"band": "Tenure", "records": "Records", "turnover": "Turnover"})
        fig.update_coloraxes(colorbar_tickformat=".0%")
        chart(fig, 340)
        hidden_note(hid, "bands")
    with c4:
        dp, hid = department_profile(d, None, min_n)
        fig = go.Figure(go.Bar(x=dp.turnover, y=dp.department, orientation="h", marker_color=VIOLET,
                               error_x=dict(type="data", symmetric=False, array=dp.ci_high - dp.turnover,
                                            arrayminus=dp.turnover - dp.ci_low, color=SLATE),
                               text=[pct(v) for v in dp.turnover], textposition="inside", insidetextanchor="start",
                               textfont=dict(color="white"),
                               hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        fig.add_vline(x=rate, line_dash="dot", line_color=SLATE)
        fig.update_layout(title="Turnover by department (with 95% intervals)", xaxis_tickformat=".0%",
                          yaxis=dict(autorange="reversed"))
        chart(fig, 340)

    st.markdown("#### What stands out")
    for title, why in insights(d):
        st.markdown(f"<div class='insight'><b>{esc(title)}</b><div class='why'>{esc(why)}</div></div>",
                    unsafe_allow_html=True)
    note("These are observed patterns in the data, not explanations of why people leave. Intervals overlap when a "
         "difference could easily be noise.", "vio")
    footer()


def insights(d) -> list[tuple[str, str]]:
    out = []
    rate = d[TARGET].mean()
    dp, _ = department_profile(d, None, min_n)
    if len(dp) >= 2:
        hi, lo = dp.iloc[0], dp.iloc[-1]
        separated = hi.ci_low > lo.ci_high
        out.append((f"{hi.department} has the highest turnover ({pct(hi.turnover)}); {lo.department} the lowest ({pct(lo.turnover)}).",
                    "The 95% intervals do not overlap, so the gap is unlikely to be noise." if separated else
                    "The 95% intervals overlap, so this gap may be noise at this sample size."))
    e, _ = band_rates(d, "engagement", min_n)
    if len(e) >= 2:
        out.append((f"Turnover is {pct(e.iloc[0].turnover)} in the lowest engagement band versus {pct(e.iloc[-1].turnover)} in the highest.",
                    f"Engagement bands: {e.iloc[0].band} ({int(e.iloc[0].records)} records) and {e.iloc[-1].band} ({int(e.iloc[-1].records)})."))
    h, _ = band_rates(d, "weekly_hours", min_n)
    if len(h) >= 2:
        top = h.sort_values("turnover").iloc[-1]
        out.append((f"Records working {top.band} per week show {pct(top.turnover)} turnover (overall {pct(rate)}).",
                    "Long hours are one of the strongest associations in this dataset."))
    if has_year:
        y, _ = yearly(d, min_n)
        if len(y) >= 2:
            pk = y.sort_values("turnover").iloc[-1]
            out.append((f"Turnover peaked in {int(pk[YEAR])} at {pct(pk.turnover)}; the latest year was {pct(y.iloc[-1].turnover)}.",
                        "Year-to-year movement in synthetic data is generated, not caused by real events."))
    return out


def page_retention():
    hero("Retention intelligence", "Retention Intelligence",
         "How retention changes with tenure and over time, and which organisational factors move with turnover. "
         "All views are group averages with uncertainty shown.")
    d = data
    c1, c2 = st.columns(2)
    with c1:
        t, hid = band_rates(d, "tenure_years", min_n)
        t["retention"] = 1 - t.turnover
        fig = go.Figure()
        fig.add_trace(go.Bar(x=t.band, y=t.retention, marker_color=EMERALD, text=[pct(v) for v in t.retention],
                             textposition="inside", insidetextanchor="start", textfont=dict(color="white"), name="Retention",
                             error_y=dict(type="data", symmetric=False, array=(1 - t.ci_low) - t.retention,
                                          arrayminus=t.retention - (1 - t.ci_high), color=SLATE)))
        fig.update_layout(title="Retention rate by tenure band", yaxis=dict(tickformat=".0%", range=[0, 1.08]),
                          showlegend=False)
        chart(fig)
        hidden_note(hid, "bands")
    with c2:
        if has_year:
            y, hid = yearly(d, min_n, by="department")
            fig = px.line(y, x=YEAR, y="turnover", color="department", markers=True, color_discrete_sequence=SEQ,
                          title="Turnover trend by department", labels={YEAR: "Year", "turnover": "Turnover"})
            fig.update_layout(yaxis_tickformat=".0%", xaxis=dict(dtick=1))
            chart(fig)
            hidden_note(hid, "department-year cells")
        else:
            st.info("Add a `year` column to see trends over time.", icon="📅")

    st.markdown("#### Organisational risk factors")
    st.caption("Observed turnover rate by band of each workplace factor, with 95% intervals. Dotted line = organisation average.")
    rate = d[TARGET].mean()
    factors = ["engagement", "weekly_hours", "training_hours", "promotion_wait_years"]
    cols = st.columns(2)
    total_hidden = 0
    for i, f in enumerate(factors):
        b, hid = band_rates(d, f, min_n)
        total_hidden += hid
        fig = go.Figure(go.Bar(x=b.band.astype(str), y=b.turnover, marker_color=SEQ[i + 1],
                               error_y=dict(type="data", symmetric=False, array=b.ci_high - b.turnover,
                                            arrayminus=b.turnover - b.ci_low, color=SLATE),
                               customdata=b.records, hovertemplate="%{x}: %{y:.1%} (%{customdata} records)<extra></extra>"))
        fig.add_hline(y=rate, line_dash="dot", line_color=SLATE)
        fig.update_layout(title=LABELS[f], yaxis_tickformat=".0%")
        with cols[i % 2]:
            chart(fig, 300)
    hidden_note(total_hidden, "bands")

    ev = get_eval()
    co = ev.coefficients.copy()
    co["direction"] = np.where(co.odds_ratio_per_sd >= 1, "Higher value, higher turnover odds", "Higher value, lower turnover odds")
    fig = go.Figure()
    for lab, colr in [("Higher value, higher turnover odds", ROSE), ("Higher value, lower turnover odds", EMERALD)]:
        part = co[co.direction == lab]
        fig.add_trace(go.Bar(x=part.odds_ratio_per_sd - 1, base=1, y=part.label, orientation="h", name=lab,
                             marker_color=colr, text=part.odds_ratio_per_sd.map(lambda v: f"{v:.2f}×"),
                             textposition="outside", customdata=part.sd,
                             hovertemplate="%{y}: odds × %{text} per +1 SD (SD = %{customdata:.2f})<extra></extra>"))
    lo, hi = float(co.odds_ratio_per_sd.min()), float(co.odds_ratio_per_sd.max())
    fig.add_vline(x=1, line_color=SLATE)
    fig.update_layout(title="Model view: change in turnover odds per +1 standard deviation (other factors held fixed)",
                      xaxis=dict(title="Odds ratio (1 = no association)", range=[min(lo, 1) - .25, max(hi, 1) + .3]),
                      legend=dict(orientation="h", y=-0.25), barmode="overlay")
    chart(fig, 340)
    note("Odds ratios come from the logistic regression fitted on training rows. They summarise associations in this "
         "dataset; they do not show what would happen if a factor were changed.", "vio")
    footer()


def page_departments():
    hero("Department analytics", "Department Analytics",
         "Compare departments on turnover, engagement, workload and development, against the organisation average.")
    d = data
    ev = get_eval()
    dp, hid = department_profile(d, ev.model, min_n)
    if dp.empty:
        st.warning(f"No department has at least {min_n} records. Lower the privacy threshold or use a larger file.")
        return
    rate = d[TARGET].mean()
    hi = dp.iloc[0]
    kpis([("Departments shown", f"{len(dp)}", f"{hid} hidden (small)"),
          ("Highest turnover", hi.department, pct(hi.turnover)),
          ("Lowest turnover", dp.iloc[-1].department, pct(dp.iloc[-1].turnover)),
          ("Largest department", dp.sort_values("records").iloc[-1].department,
           f"{int(dp.records.max()):,} records"),
          ("Organisation turnover", pct(rate), "reference line")])
    show = dp[["department", "records", "share", "turnover", "ci_low", "ci_high", "vs_org_pts", "engagement",
               "weekly_hours", "training_hours", "promotion_wait_years", "tenure_years", "mean_model_score"]]
    st.dataframe(rnd(show, 3), hide_index=True, width="stretch", column_config=COLS)
    st.caption("Mean model score is the average predicted probability for the whole department — a calibration check "
               "against observed turnover, never a score for any person.")
    hidden_note(hid, "departments")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(dp, x="department", y="vs_org_pts", color="vs_org_pts", color_continuous_scale=RISK_SCALE,
                     color_continuous_midpoint=0, text=dp.vs_org_pts.map(lambda v: f"{v:+.1f}"),
                     title="Turnover versus organisation average (percentage points)",
                     labels={"vs_org_pts": "points", "department": ""})
        fig.update_layout(coloraxis_showscale=False)
        chart(fig)
    with c2:
        long = dp.melt(id_vars="department", value_vars=["turnover", "mean_model_score"], var_name="measure",
                       value_name="rate").replace({"turnover": "Observed turnover", "mean_model_score": "Mean model score"})
        fig = px.bar(long, x="department", y="rate", color="measure", barmode="group",
                     color_discrete_sequence=[VIOLET, CYAN], title="Observed turnover vs average model score",
                     labels={"rate": "", "department": ""})
        fig.update_layout(yaxis_tickformat=".0%")
        chart(fig)

    st.markdown("#### Workforce distribution")
    c3, c4 = st.columns(2)
    with c3:
        x = d.assign(band=pd.Categorical(pd.cut(d.weekly_hours, BANDS["weekly_hours"][0], labels=BANDS["weekly_hours"][1],
                                                right=False).astype(str), categories=BANDS["weekly_hours"][1], ordered=True))
        x = x[x.department.isin(dp.department)]
        mix = x.groupby(["department", "band"], observed=True).size().rename("records").reset_index()
        mix["share"] = mix.records / mix.groupby("department").records.transform("sum")
        fig = px.bar(mix, x="share", y="department", color="band", orientation="h",
                     color_discrete_sequence=[EMERALD, "#34D399", CYAN, BLUE, VIOLET, ROSE],
                     title="Weekly-hours mix within each department", labels={"share": "", "department": "", "band": ""})
        fig.update_layout(xaxis_tickformat=".0%")
        chart(fig)
    with c4:
        sel = st.multiselect("Compare departments", list(dp.department), default=list(dp.department[:3]),
                             max_selections=5)
        metrics = ["engagement", "weekly_hours", "training_hours", "promotion_wait_years", "tenure_years"]
        org = d[metrics].mean()
        if sel:
            prof = dp.set_index("department").loc[sel, metrics]
            idx = ((prof / org - 1) * 100).reset_index().melt(id_vars="department", var_name="factor", value_name="diff")
            idx["factor"] = idx.factor.map(LABELS)
            fig = px.bar(idx, x="diff", y="factor", color="department", barmode="group", orientation="h",
                         color_discrete_sequence=SEQ, title="Profile vs organisation average (% difference)",
                         labels={"diff": "% above (+) or below (−) the organisation average", "factor": ""})
            fig.add_vline(x=0, line_color=SLATE)
            chart(fig)
        else:
            st.info("Pick at least one department to compare.", icon="👆")
    footer()


def page_engagement():
    hero("Engagement intelligence", "Engagement Intelligence",
         "Engagement scores, working hours and development time, and how they combine with turnover.")
    d = data
    hi_hours = (d.weekly_hours >= 50).mean()
    kpis([("Mean engagement", f"{d.engagement.mean():.2f} / 5", f"SD {d.engagement.std():.2f}"),
          ("Low engagement (<2)", pct((d.engagement < 2).mean()), "share of records"),
          ("Mean weekly hours", f"{d.weekly_hours.mean():.1f} h", f"{pct(hi_hours)} at 50 h or more"),
          ("Mean training", f"{d.training_hours.mean():.0f} h", "per record"),
          ("Years since promotion", f"{d.promotion_wait_years.mean():.1f}", "average")])
    c1, c2 = st.columns(2)
    with c1:
        e, hid = band_rates(d, "engagement", min_n)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=e.band.astype(str), y=e.records, name="Records", marker_color="rgba(6,182,212,.45)", yaxis="y"))
        fig.add_trace(go.Scatter(x=e.band.astype(str), y=e.turnover, name="Turnover", mode="lines+markers+text",
                                 line=dict(color=ROSE, width=3), text=[pct(v) for v in e.turnover],
                                 textposition="top center", yaxis="y2"))
        fig.update_layout(title="Engagement distribution and turnover", yaxis=dict(title="Records"),
                          yaxis2=dict(title="Turnover", overlaying="y", side="right", tickformat=".0%",
                                      range=[0, max(e.turnover.max() * 1.3, .1)]),
                          legend=dict(orientation="h", y=-0.15))
        chart(fig)
        hidden_note(hid, "bands")
    with c2:
        dp, hid = department_profile(d, None, min_n)
        fig = px.scatter(dp, x="weekly_hours", y="engagement", size="records", color="department", size_max=48,
                         text="department", color_discrete_sequence=SEQ, hover_data={"turnover": ":.1%"},
                         title="Department averages: hours vs engagement (size = records)",
                         labels={"weekly_hours": "Mean weekly hours", "engagement": "Mean engagement"})
        fig.update_traces(textposition="top center", cliponaxis=False)
        pad_y = max((dp.engagement.max() - dp.engagement.min()) * .35, .05)
        pad_x = max((dp.weekly_hours.max() - dp.weekly_hours.min()) * .25, .3)
        fig.update_layout(showlegend=False, yaxis_range=[dp.engagement.min() - pad_y, dp.engagement.max() + pad_y],
                          xaxis_range=[dp.weekly_hours.min() - pad_x, dp.weekly_hours.max() + pad_x])
        chart(fig)

    st.markdown("#### Workplace-factor combinations")
    opts = {LABELS[k]: k for k in ["engagement", "weekly_hours", "training_hours", "promotion_wait_years", "tenure_years"]}
    a, b = st.columns(2)
    rlab = a.selectbox("Rows", list(opts), index=0)
    clab = b.selectbox("Columns", [k for k in opts if k != rlab], index=0)
    rate, n, hid = heatmap(d, opts[rlab], opts[clab], min_n)
    txt = np.where(rate.isna(), "hidden", (rate * 100).round(0).astype("Int64").astype(str) + "%")
    fig = go.Figure(go.Heatmap(z=rate.values, x=[str(c) for c in rate.columns], y=[str(r) for r in rate.index],
                               colorscale=RISK_SCALE, text=txt, texttemplate="%{text}", customdata=n.values,
                               hovertemplate="%{y} × %{x}: %{z:.1%} of %{customdata} records<extra></extra>",
                               colorbar=dict(tickformat=".0%", title="Turnover")))
    fig.update_layout(title=f"Turnover rate: {rlab} × {clab}", xaxis_title=clab, yaxis_title=rlab)
    chart(fig, 420)
    hidden_note(hid, "cells")

    c3, c4 = st.columns(2)
    with c3:
        low = d.assign(low=(d.engagement < 2).astype(int)).groupby("department").agg(records=("low", "size"),
                                                                                       low=("low", "mean")).reset_index()
        low = low[low.records >= min_n].sort_values("low")
        fig = px.bar(low, x="low", y="department", orientation="h", color_discrete_sequence=[CYAN],
                     text=low["low"].map(pct), title="Share of records with low engagement (< 2) by department",
                     labels={"low": "", "department": ""})
        fig.update_layout(xaxis_tickformat=".0%")
        chart(fig, 330)
    with c4:
        x = d.assign(group=np.where(d.engagement >= 3, "Engagement 3–5", "Engagement 1–3"),
                     band=band(d, "weekly_hours"))
        g, _ = group_rates(x, ["group", "band"], min_n)
        g["band"] = pd.Categorical(g.band.astype(str), BANDS["weekly_hours"][1], ordered=True)
        g = g.sort_values("band")
        fig = px.line(g, x=g.band.astype(str), y="turnover", color="group", markers=True,
                      color_discrete_map={"Engagement 1–3": ROSE, "Engagement 3–5": EMERALD},
                      title="Turnover by weekly hours, lower vs higher engagement",
                      labels={"x": "Weekly hours", "band": "Weekly hours", "turnover": "Turnover", "group": ""})
        fig.update_layout(yaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.25))
        chart(fig, 330)
    note("Long hours and low engagement compound: the gap between the two lines widens as hours rise. In the synthetic "
         "generator these factors were drawn independently; in real organisations they usually move together, which "
         "makes single-factor views easy to misread.", "warn")
    footer()


def page_simulator():
    hero("Scenario analysis", "What-If Simulator",
         "Shift workplace factors for the whole organisation or selected departments and see how the model's "
         "average score responds. This is model sensitivity, not a forecast of what a policy would achieve.")
    d = data
    ev = get_eval()
    depts = sorted(d.department.unique())
    left, right = st.columns([1, 2.1])
    with left:
        st.markdown("**Hypothetical changes**")
        scope = st.multiselect("Apply to departments", depts, placeholder="All departments")
        h = st.slider("Weekly hours", -10.0, 10.0, 0.0, 0.5, format="%+.1f h")
        e = st.slider("Engagement score", -1.0, 1.0, 0.0, 0.1, format="%+.1f")
        t = st.slider("Training hours", -20, 40, 0, 5, format="%+d h")
        p = st.slider("Years since promotion", -2.0, 2.0, 0.0, 0.25, format="%+.2f")
        st.caption("Values are clipped to valid ranges (engagement 1–5, hours 0–168, no negative training).")
    res = scenario_detail(ev.model, d, h, e, t, p, scope or None, min_n)
    delta = (res["scenario"] - res["base"]) * 100
    with right:
        kpis([("Baseline mean score", pct(res["base"]), f"observed turnover {pct(res['observed'])}"),
              ("Scenario mean score", pct(res["scenario"]), f"{delta:+.2f} percentage points"),
              ("Records affected", f"{res['affected_records']:,}", "all" if not scope else ", ".join(scope)[:40]),
              ("Direction", "lower" if delta < -0.05 else "higher" if delta > 0.05 else "no change", "model association")])
        g = res["by_department"]
        long = g.melt(id_vars=["department"], value_vars=["base", "scenario"], var_name="case", value_name="score")
        long["case"] = long.case.map({"base": "Baseline", "scenario": "Scenario"})
        fig = px.bar(long, x="department", y="score", color="case", barmode="group",
                     color_discrete_sequence=[SLATE, EMERALD if delta <= 0 else ROSE],
                     title="Average model score by department", labels={"score": "", "department": ""})
        fig.update_layout(yaxis_tickformat=".0%")
        chart(fig, 360)
        hidden_note(res["hidden"], "departments")
    st.markdown("#### Sensitivity curves")
    st.caption("Average model score as one factor shifts on its own (others unchanged), for the chosen scope.")
    c1, c2 = st.columns(2)
    for i, (f, vals, fmt) in enumerate([("weekly_hours", np.arange(-10, 10.1, 2), "%+.0f h"),
                                         ("engagement", np.round(np.arange(-1, 1.01, .25), 2), "%+.2f")]):
        s = sensitivity(ev.model, d, f, vals, scope or None)
        fig = px.line(s, x="shift", y="mean_score", markers=True, title=f"Shift in {LABELS[f].lower()}",
                      labels={"shift": "Shift", "mean_score": "Mean model score"}, color_discrete_sequence=[VIOLET])
        fig.update_layout(yaxis_tickformat=".0%")
        fig.update_xaxes(tickformat=fmt.replace("%", "").replace(" h", "") or None)
        with (c1, c2)[i]:
            chart(fig, 300)
    note("<b>How to read this:</b> the model learned associations from historical (here, synthetic) records. Changing "
         "an input moves the score along those associations; it does not mean the same change in a real organisation "
         "would cause the same change in turnover. Confounding, selection and feedback effects are not modelled.", "warn")
    export = pd.concat([pd.DataFrame([{"department": "ALL (organisation)", "records": len(d), "base": res["base"],
                                       "scenario": res["scenario"], "delta_pts": delta}]),
                        g[["department", "records", "base", "scenario", "delta_pts"]]])
    export.insert(0, "settings", f"hours {h:+}, engagement {e:+}, training {t:+}, promotion wait {p:+}, "
                                 f"scope {'all' if not scope else '/'.join(scope)}")
    st.download_button("Download scenario summary (CSV)", csv_bytes(rnd(export, 4)), "talentflow_scenario.csv",
                       "text/csv", icon="⬇️")
    footer()


def page_model():
    hero("Model performance", "Model Performance & Methodology",
         "Verified, calculated-at-runtime evaluation of the logistic regression against simple baselines, on a "
         "holdout set the model never saw during training.")
    ev = get_eval()
    m = ev.metrics
    kpis([("ROC-AUC (holdout)", f"{m['roc_auc']:.3f}", f"CV {ev.cv['Logistic regression, scaled (app model)'][0]:.3f} ± {ev.cv['Logistic regression, scaled (app model)'][1]:.3f}"),
          ("Average precision", f"{m['average_precision']:.3f}", f"base rate {ev.imbalance['test_rate']:.3f}"),
          ("F1 at chosen threshold", f"{m['f1']:.3f}", f"threshold {m['threshold']:.2f}"),
          ("Recall", f"{m['recall']:.3f}", f"at 0.50: {m['default_threshold']['recall']:.3f}"),
          ("Precision", f"{m['precision']:.3f}", f"at 0.50: {m['default_threshold']['precision']:.3f}"),
          ("Brier score", f"{m['brier']:.3f}", f"{m['train_rows']:,} train / {m['test_rows']:,} test")])
    if is_demo:
        note("These numbers come from <b>synthetic</b> labels produced by a known formula. They show the method works "
             "end to end; they say nothing about how well turnover can be predicted in a real organisation.", "warn")

    st.markdown("#### Model comparison (same holdout)")
    comp = ev.comparison.copy()
    st.dataframe(rnd(comp, 3), hide_index=True, width="stretch",
                 column_config={"roc_auc": PCT("ROC-AUC", min_value=0.0, max_value=1.0, format="%.3f"),
                                "average_precision": PCT("avg precision", min_value=0.0, max_value=1.0, format="%.3f"),
                                "cv_auc_mean": st.column_config.NumberColumn("CV AUC (train)", format="%.3f"),
                                "cv_auc_std": st.column_config.NumberColumn("CV ± sd", format="%.3f")})
    st.caption("Thresholds are chosen to maximise F1 on out-of-fold predictions from the training rows only, then "
               "applied once to the holdout. Balanced class weights shift scores upward (worse Brier) without "
               "improving ranking, so the app keeps the unweighted model for calibrated averages.")

    c1, c2 = st.columns(2)
    names = list(ev.curves)
    short = {names[0]: "Logistic regression", names[1]: "Engagement-only rule", names[2]: "Prior baseline"}
    with c1:
        fig = go.Figure()
        for i, nme in enumerate(names):
            cv = ev.curves[nme]
            fig.add_trace(go.Scatter(x=cv["fpr"], y=cv["tpr"], name=short[nme], line=dict(color=[EMERALD, AMBER, SLATE][i], width=3 - i * .7)))
        fig.update_layout(title="ROC curve (holdout)", xaxis_title="False positive rate", yaxis_title="True positive rate",
                          legend=dict(orientation="h", y=-0.2))
        chart(fig)
    with c2:
        fig = go.Figure()
        for i, nme in enumerate(names[:2]):
            cv = ev.curves[nme]
            fig.add_trace(go.Scatter(x=cv["recall"], y=cv["precision"], name=short[nme], line=dict(color=[EMERALD, AMBER][i], width=3 - i * .7)))
        fig.add_hline(y=ev.imbalance["test_rate"], line_dash="dot", line_color=SLATE, annotation_text="base rate")
        fig.update_layout(title="Precision–recall curve (holdout)", xaxis_title="Recall", yaxis_title="Precision",
                          legend=dict(orientation="h", y=-0.2))
        chart(fig)

    st.markdown("#### Threshold explorer")
    st.caption("Exploring thresholds on the holdout is for understanding trade-offs only; picking one here would leak "
               "test information. The reported threshold above was fixed on training data.")
    thr = st.select_slider("Decision threshold", [round(x, 2) for x in ev.thresholds.threshold],
                           value=min(ev.thresholds.threshold, key=lambda v: abs(v - m["threshold"])))
    row = ev.thresholds.loc[(ev.thresholds.threshold - thr).abs().idxmin()]
    c3, c4 = st.columns([1, 1.2])
    with c3:
        cm = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        fig = px.imshow(cm, x=["Predicted stay", "Predicted leave"], y=["Actual stay", "Actual leave"], text_auto=True,
                        color_continuous_scale=[[0, "#A5F3FC"], [1, "#A78BFA"]], title=f"Confusion matrix at {thr:.2f}")
        fig.update_traces(textfont=dict(color="#0B1B3B", size=20))
        fig.update_layout(coloraxis_showscale=False)
        chart(fig, 340)
        st.caption(f"Precision {row.precision:.3f} · recall {row.recall:.3f} · F1 {row.f1:.3f} · flags "
                   f"{pct(row.flagged_share)} of holdout records")
    with c4:
        tl = ev.thresholds.melt(id_vars="threshold", value_vars=["precision", "recall", "f1"], var_name="metric")
        fig = px.line(tl, x="threshold", y="value", color="metric", color_discrete_sequence=[CYAN, EMERALD, VIOLET],
                      title="Precision, recall and F1 across thresholds")
        fig.add_vline(x=m["threshold"], line_dash="dot", line_color=SLATE, annotation_text="chosen on training data")
        chart(fig, 340)

    c5, c6 = st.columns(2)
    with c5:
        imp = ev.importance
        fig = go.Figure(go.Bar(x=imp.auc_drop, y=imp.label, orientation="h", marker_color=CYAN,
                               error_x=dict(type="data", array=imp["std"], color=SLATE)))
        fig.update_layout(title="Aggregate feature importance (drop in holdout AUC when shuffled)",
                          yaxis=dict(autorange="reversed"), xaxis_title="AUC drop")
        chart(fig)
    with c6:
        cal = ev.calibration
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=SLATE), name="Perfect"))
        fig.add_trace(go.Scatter(x=cal.mean_score, y=cal.observed_rate, mode="lines+markers", name="Model",
                                 line=dict(color=VIOLET, width=3), customdata=cal.records,
                                 hovertemplate="score %{x:.2f} → observed %{y:.2f} (%{customdata} records)<extra></extra>"))
        mx = float(max(cal.mean_score.max(), cal.observed_rate.max()) * 1.1)
        fig.update_layout(title="Calibration (holdout deciles)", xaxis=dict(title="Mean predicted", range=[0, mx]),
                          yaxis=dict(title="Observed turnover", range=[0, mx]), legend=dict(orientation="h", y=-0.2))
        chart(fig)

    c7, c8 = st.columns(2)
    with c7:
        st.markdown("#### Class imbalance")
        im = ev.imbalance
        kpis([("Leavers", f"{im['leavers']:,}", pct(im['overall_rate'])), ("Stayers", f"{im['stayers']:,}", f"{im['ratio']:.1f} : 1"),
              ("Train / test rate", f"{im['train_rate']:.3f} / {im['test_rate']:.3f}", "stratified split")])
        st.caption("With about one leaver per three to four stayers, accuracy is misleading (always predicting 'stay' "
                   f"scores {pct(1 - im['test_rate'])}). Average precision and recall are the more honest measures.")
    with c8:
        st.markdown("#### Time-ordered check")
        tc = ev.temporal
        if tc:
            kpis([("Train → test", f"{tc['train_years']} → {tc['test_year']}", f"{tc['train_rows']:,} / {tc['test_rows']:,} rows"),
                  ("ROC-AUC", f"{tc['roc_auc']:.3f}", f"engagement rule {tc['engagement_rule_auc']:.3f}"),
                  ("Avg score vs observed", f"{pct(tc['mean_score'])} vs {pct(tc['observed_rate'])}",
                   f"training years {pct(tc['train_rate'])}")])
            st.caption("Ranking holds up on the latest year, but the average score tracks the older, higher turnover "
                       "rate. In real use a model needs recalibration as conditions change.")
        else:
            st.info("Needs a `year` column with at least two years and enough leavers per year.", icon="📅")

    with st.expander("Methodology and limitations", expanded=False, icon="📘"):
        st.markdown(f"""
- **Data:** {'synthetic employee-year records; labels come from a fixed logistic formula plus randomness' if is_demo else 'your uploaded, cleaned file'}.
- **Features:** department, tenure, weekly hours, engagement, training hours, years since promotion. No protected characteristics.
- **Split:** stratified 75/25 holdout with a fixed seed (42). Encoding and scaling are fitted inside the pipeline on training rows only.
- **Model:** logistic regression (L2, C = 1) on one-hot department + standardised numeric inputs.
- **Baselines:** training-rate prior and an engagement-only rule, scored on the same holdout.
- **Threshold:** F1-maximising threshold from 5-fold out-of-fold training predictions.
- **Importance:** permutation importance (20 repeats, holdout AUC) and standardised odds ratios.
- **Limits:** random holdout is not prospective validation; synthetic labels are cleaner than reality; no fairness audit is possible without (appropriately governed) demographic data; results must not drive decisions about individuals.
""")
    st.download_button("Download evaluation summary (JSON)", json.dumps(evaluation_summary(ev), indent=2, default=str),
                       "talentflow_evaluation.json", "application/json", icon="⬇️")
    footer()


def page_data():
    hero("Data explorer", "Data Explorer",
         "Upload an anonymised CSV, check how it was validated, filter the workforce and download aggregate reports. "
         "Uploads stay in this browser session and are never saved.")
    up = st.file_uploader("Anonymised workforce CSV", type=["csv"], max_upload_size=MAX_UPLOAD_MB,
                          key=f"uploader_{ss.get('uploader_gen', 0)}",
                          help="Required columns: " + ", ".join(FEATURES + [TARGET]) + ". Optional: year.")
    if up is not None:
        raw_bytes = up.getvalue()
        key = hashlib.sha256(raw_bytes).hexdigest()[:16]
        if ss.get("upload_key") != key:
            try:
                if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
                    raise ValueError(f"The file is larger than {MAX_UPLOAD_MB} MB.")
                raw = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
                clean, rep = prepare(raw)
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
                st.error(f"This file could not be used: {exc}", icon="⚠️")
                ss["upload_error"] = str(exc)
            else:
                ss.update(upload_df=clean, upload_report=rep, upload_name=up.name, upload_key=key, use_upload=True)
                ss.pop("upload_error", None)
                st.rerun()
    if "upload_df" in ss:
        c1, c2 = st.columns([3, 1])
        c1.success(f"Using **{esc(ss['upload_name'])}** for every page in this session "
                   f"({len(ss['upload_df']):,} valid rows).", icon="✅")
        if c2.button("Remove upload", icon="🗑️", width="stretch"):
            for k in ("upload_df", "upload_report", "upload_name", "upload_key", "upload_eval", "upload_eval_key",
                      "use_upload"):
                ss.pop(k, None)
            ss["uploader_gen"] = ss.get("uploader_gen", 0) + 1     # also clears the file widget
            st.rerun()
        rep = ss["upload_report"]
        with st.expander("Validation report", expanded=not is_demo, icon="🧾"):
            st.table(rep.as_frame().set_index("check"))
            for n_ in rep.notes:
                st.caption("• " + n_)

    with st.expander("Data contract and template", icon="📄"):
        st.markdown("""
| column | type | rule |
|:--|:--|:--|
| department | text | up to 50 departments |
| tenure_years | number | 0–70 |
| weekly_hours | number | 0–168 |
| engagement | number | 1–5 |
| training_hours | number | 0–2000 |
| promotion_wait_years | number | 0–70 |
| left | 0/1 or Yes/No | outcome in the period |
| year *(optional)* | integer | snapshot year, enables trends |

Common header names such as *Attrition*, *Dept*, *YearsAtCompany* are mapped automatically. Columns that look like
protected characteristics (age, gender, ethnicity, marital status…) or identifiers/pay are **ignored**. At least 100
valid rows and 10 of each outcome are needed. Do not upload names, IDs or other personal data.
""")
        st.download_button("Download CSV template (synthetic, 200 rows)", csv_bytes(generate(200, 7)),
                           "talentflow_template.csv", "text/csv", icon="⬇️")

    d = data
    st.markdown(f"#### Filter the workforce · {esc(source_label)}")
    f1, f2, f3 = st.columns([2, 1.3, 1.3])
    depts = sorted(d.department.unique())
    sel = f1.multiselect("Departments", depts, placeholder="All departments")
    tmin, tmax = float(np.floor(d.tenure_years.min())), float(np.ceil(d.tenure_years.max()))
    tr = f2.slider("Tenure (years)", tmin, tmax, (tmin, tmax), 0.5)
    if has_year:
        y0, y1 = int(d[YEAR].min()), int(d[YEAR].max())
        yr = f3.slider("Years", y0, y1, (y0, y1)) if y1 > y0 else (y0, y1)
    else:
        yr = None
        f3.caption("No year column")
    m_ = d.tenure_years.between(*tr)
    if sel:
        m_ &= d.department.isin(sel)
    if yr:
        m_ &= d[YEAR].between(*yr)
    fd = d[m_]
    if len(fd) < min_n:
        st.warning(f"Only {len(fd)} records match — below the privacy threshold of {min_n}. Widen the filters.", icon="🔒")
        footer()
        return
    kpis([("Records matched", f"{len(fd):,}", f"{pct(len(fd) / len(d))} of all"),
          ("Turnover", pct(fd[TARGET].mean()), f"overall {pct(d[TARGET].mean())}"),
          ("Mean engagement", f"{fd.engagement.mean():.2f}", "/ 5"),
          ("Mean weekly hours", f"{fd.weekly_hours.mean():.1f}", "h")])
    summ, hid = group_rates(fd, "department", min_n)
    st.dataframe(rnd(summ.drop(columns=["leavers"]), 3), hide_index=True, width="stretch", column_config=COLS)
    hidden_note(hid, "departments")

    st.markdown("**Column profile (aggregate statistics only)**")
    prof = fd[NUMERIC].describe(percentiles=[.05, .25, .5, .75, .95]).T.drop(columns=["min", "max"])
    prof = prof.reset_index().rename(columns={"index": "column"})
    prof["column"] = prof.column.map(LABELS)
    st.caption("Minimum and maximum are left out because they describe a single record; 5th and 95th percentiles are shown instead.")
    st.dataframe(rnd(prof, 2), hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        f = st.selectbox("Distribution of", [LABELS[k] for k in NUMERIC])
        key = {v: k for k, v in LABELS.items()}[f]
        edges = np.linspace(fd[key].quantile(.01), fd[key].quantile(.99), 13)
        bins = pd.cut(fd[key].clip(edges[0], edges[-1]), edges, include_lowest=True)
        h = (fd.assign(bin=bins, outcome=fd[TARGET].map({0: "Stayed", 1: "Left"}))
             .groupby(["bin", "outcome"], observed=False).size().rename("records").reset_index())
        hidden_bins = int(((h.records > 0) & (h.records < min_n)).sum())
        h.loc[h.records < min_n, "records"] = 0
        h["mid"] = h.bin.map(lambda iv: iv.mid).astype(float)
        fig = px.bar(h, x="mid", y="records", color="outcome", color_discrete_map={"Stayed": CYAN, "Left": ROSE},
                     title=f"{f}: stayed vs left", labels={"mid": f, "records": "Records", "outcome": ""})
        fig.update_layout(bargap=.05)
        chart(fig, 340)
        st.caption("Outer 1% on each side folded into the end bins" +
                   (f"; {hidden_bins} bar segments under {min_n} records hidden." if hidden_bins else "."))
    with c2:
        bands = []
        for col in BANDS:
            b, _ = band_rates(fd, col, min_n)
            bands.append(b.assign(factor=LABELS[col])[["factor", "band", "records", "turnover", "ci_low", "ci_high"]])
        band_tbl = pd.concat(bands, ignore_index=True)
        band_tbl["band"] = band_tbl.band.astype(str)
        st.markdown("**Turnover by factor band**")
        st.dataframe(rnd(band_tbl, 3), hide_index=True, width="stretch", height=300, column_config=COLS)

    st.markdown("#### Downloads (aggregates only)")
    dl = st.columns(4)
    dl[0].download_button("Department summary", csv_bytes(rnd(summ, 4)), "department_summary.csv", "text/csv",
                          icon="⬇️", width="stretch")
    dl[1].download_button("Factor-band summary", csv_bytes(rnd(band_tbl, 4)), "factor_band_summary.csv", "text/csv",
                          icon="⬇️", width="stretch")
    if has_year:
        yt, _ = yearly(fd, min_n)
        dl[2].download_button("Yearly turnover", csv_bytes(rnd(yt, 4)), "yearly_turnover.csv", "text/csv",
                              icon="⬇️", width="stretch")
    else:
        dl[2].button("Yearly turnover", disabled=True, width="stretch", help="Needs a year column")
    if "upload_report" in ss and not is_demo:
        dl[3].download_button("Validation report", csv_bytes(ss["upload_report"].as_frame()), "validation_report.csv",
                              "text/csv", icon="⬇️", width="stretch")
    else:
        dl[3].download_button("Synthetic example", csv_bytes(load_demo(1800, 42)), "synthetic_example.csv", "text/csv",
                              icon="⬇️", width="stretch")
    st.caption("No download contains row-level model scores. Department and band files omit groups under the "
               "privacy threshold.")
    footer()


PAGES = [st.Page(page_command, title="Workforce Command Center", icon="🏢", url_path="command", default=True),
         st.Page(page_retention, title="Retention Intelligence", icon="🔁", url_path="retention"),
         st.Page(page_departments, title="Department Analytics", icon="🏬", url_path="departments"),
         st.Page(page_engagement, title="Engagement Intelligence", icon="💡", url_path="engagement"),
         st.Page(page_simulator, title="What-If Simulator", icon="🎛️", url_path="simulator"),
         st.Page(page_model, title="Model Performance", icon="📈", url_path="model"),
         st.Page(page_data, title="Data Explorer", icon="🗂️", url_path="data")]
st.navigation(PAGES).run()
