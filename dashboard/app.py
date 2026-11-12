"""
dashboard/app.py

Workflow Bottleneck Analyzer: upload a workflow event log (or pick a sample)
and see where time is lost.

Run from the project root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.theme import (  # noqa: E402
    ACCENT, ACCENT_SOFT, CHART_CONFIG, INK, MUTED, NEUTRAL, VARIANT_COLORS, WARN,
    apply_theme, brand, chain_card, data_table, esc, finding_card, fmt_minutes, kpi, risk_bar,
    style_fig, variant_card, variants_table,
)
from src.automation_engine import ASSUMED_REDUCTION, identify_automation_opportunities  # noqa: E402
from src.bottleneck_detector import (  # noqa: E402
    detect_bottlenecks, find_handoff_delay, generate_priority_actions, generate_recommendation,
    handoff_action, has_slow_tail, slow_tail_action, summarize_steps,
)
from src.data_loader import apply_mapping, read_csv_bytes, suggest_mapping  # noqa: E402
from src.health_analyzer import WEIGHTS, compute_health_score  # noqa: E402
from src.decision_chain import build_decision_chain  # noqa: E402
from src.preprocessing import (  # noqa: E402
    compute_case_durations, detect_exception_flows, infer_step_order, preprocess_workflow,
)
from src.step_risk import score_step_risk  # noqa: E402
from src.variant_miner import (  # noqa: E402
    generate_variant_insights, generate_variant_recommendations, mine_variants,
    score_variants, tag_variant_type,
)

st.set_page_config(page_title="Workflow Bottleneck Analyzer",
                   page_icon=os.path.join(PROJECT_ROOT, "assets", "logo.png"),
                   layout="wide")
apply_theme()

SAMPLES = {
    "Sales pipeline (sample)": "sample_sales_pipeline.csv",
    "Support tickets (sample)": "sample_support_tickets.csv",
}
UPLOAD = "Upload your own CSV"
PAGES = ["Summary", "Bottlenecks", "Process variants", "Recommendations"]
REQUIRED_COLUMNS = [
    ("case_id", "Identifier of one workflow instance (order, ticket, lead)"),
    ("task", "Name of the step"),
    ("start_time", "When the step started"),
    ("end_time", "When the step finished"),
    ("user", "Person or system that performed the step (optional)"),
]
FIELD_LABELS = [
    ("case_id", "Case ID (one order, ticket, lead)", True),
    ("task", "Step name", True),
    ("start_time", "Step start time", True),
    ("end_time", "Step end time", True),
    ("user", "Person or system", False),
]
REQUIRED_COLUMNS_ORDER = ("case_id", "task", "start_time", "end_time")
NONE_CHOICE = "(none)"


@st.cache_data(show_spinner=False)
def read_upload(content: bytes) -> pd.DataFrame:
    return read_csv_bytes(content)


@st.cache_data(show_spinner=False)
def load_raw(content: bytes, mapping: tuple) -> pd.DataFrame:
    return apply_mapping(read_csv_bytes(content), dict(mapping))


def pct(value: float) -> str:
    return f"{value:.0f}%"


# Sidebar: data source, settings, navigation
with st.sidebar:
    st.markdown(brand(), unsafe_allow_html=True)
    st.markdown("## Data")
    source = st.radio("Dataset", [*SAMPLES, UPLOAD], label_visibility="collapsed")
    uploaded = None
    if source == UPLOAD:
        uploaded = st.file_uploader("CSV event log", type=["csv"], label_visibility="collapsed")

    st.markdown("## View")
    page = st.radio("View", PAGES, label_visibility="collapsed", key="nav")

# Load data
content: bytes | None = None
dataset_label = source
if source == UPLOAD:
    if uploaded is not None:
        content, dataset_label = uploaded.getvalue(), uploaded.name
else:
    with open(os.path.join(PROJECT_ROOT, "data", SAMPLES[source]), "rb") as handle:
        content = handle.read()

mapping_pairs: tuple | None = None
if content is not None and source == UPLOAD:
    try:
        preview = read_upload(content)
    except ValueError as exc:
        st.error(f"The file could not be loaded: {exc}")
        st.stop()
    guess = suggest_mapping(list(preview.columns))
    with st.sidebar:
        with st.expander("Match your columns", expanded=not all(guess[f] for f in REQUIRED_COLUMNS_ORDER)):
            st.caption("Tell the app which column in your file holds each field.")
            chosen: dict[str, str | None] = {}
            options = [NONE_CHOICE, *map(str, preview.columns)]
            for field, label, required in FIELD_LABELS:
                default = guess.get(field)
                index = options.index(str(default)) if default is not None else 0
                pick = st.selectbox(label + ("" if required else " (optional)"), options, index=index,
                                    key=f"map-{field}-{dataset_label}")
                chosen[field] = None if pick == NONE_CHOICE else pick
    mapping_pairs = tuple(sorted(chosen.items(), key=lambda kv: kv[0]))

if content is None:
    st.title("Workflow Bottleneck Analyzer")
    st.markdown('<div class="kicker">Find which step slows a process down, and what fixing it is worth.</div>',
                unsafe_allow_html=True)
    st.info("Upload a CSV event log in the sidebar, or choose a sample dataset.")
    st.markdown("**Expected columns**")
    st.dataframe(pd.DataFrame(REQUIRED_COLUMNS, columns=["Column", "Meaning"]),
                 hide_index=True, width="stretch")
    st.stop()

if mapping_pairs is None:  # the bundled samples already use our column names
    mapping_pairs = tuple((f, f) for f in (*REQUIRED_COLUMNS_ORDER, "user"))
try:
    raw = load_raw(content, mapping_pairs)
except (ValueError, FileNotFoundError) as exc:
    st.error(f"The file could not be loaded: {exc}")
    st.stop()

order = infer_step_order(raw)
default_sla = preprocess_workflow(raw).attrs["sla_minutes"]

with st.sidebar:
    with st.expander("SLA targets per step"):
        st.caption("Defaults are 2x each step's median duration. Replace them with your real targets.")
        sla_editor = st.data_editor(
            pd.DataFrame({"Step": order, "Target (min)": [float(default_sla[t]) for t in order]}),
            key=f"sla-{dataset_label}-{len(raw)}",
            hide_index=True,
            disabled=["Step"],
            column_config={"Target (min)": st.column_config.NumberColumn(min_value=1, step=5, format="%d")},
            width="stretch",
        )

sla_minutes = dict(zip(sla_editor["Step"], sla_editor["Target (min)"]))
df = preprocess_workflow(raw, sla_minutes)
if df.empty:
    st.error("No usable rows remain after cleaning. Check the timestamp columns.")
    st.stop()

# Analysis
case_durations = compute_case_durations(df)
exceptions_df, n_exception_cases = detect_exception_flows(df, order)
task_stats, bn_task, bn_total_minutes, bn_pct = detect_bottlenecks(df)
steps = summarize_steps(df, order)
n_cases = df["case_id"].nunique()
health = compute_health_score(df, bn_pct, n_exception_cases)
quality = df.attrs["quality"]
bn = steps.loc[bn_task]
median_cycle_min = float(case_durations["total_duration_minutes"].median())


def top_wait_step() -> tuple[str, float]:
    waits = steps["avg_wait_before"]
    return waits.idxmax(), float(waits.max())


# Page: Summary
if page == "Summary":
    st.title("Workflow Bottleneck Analyzer")
    first, last = df["start_time"].min(), df["end_time"].max()
    st.markdown(
        f'<div class="kicker">{dataset_label} &nbsp;·&nbsp; {n_cases:,} cases &nbsp;·&nbsp; '
        f'{first:%d %b %Y} to {last:%d %b %Y}</div>',
        unsafe_allow_html=True,
    )

    detail = f"It accounts for {pct(bn_pct)} of all step time. "
    if bn["avg_duration"] > 1.5 * bn["median_duration"]:
        detail += (
            f"A typical run takes {fmt_minutes(bn['median_duration'])}, but the average is "
            f"{fmt_minutes(bn['avg_duration'])}: a minority of very slow runs drive the delay "
            f"({pct(bn['sla_breach_pct'])} exceed the {fmt_minutes(bn['sla_target'])} SLA target)."
        )
    else:
        detail += (f"Runs average {fmt_minutes(bn['avg_duration'])} "
                   f"(median {fmt_minutes(bn['median_duration'])}).")
    wait_step, wait_value = top_wait_step()
    if wait_step != bn_task and wait_value > bn["avg_duration"] * 0.5:
        detail += (f" Separately, cases wait {fmt_minutes(wait_value)} on average before "
                   f"'{wait_step}' starts, the longest handoff delay.")
    st.markdown(finding_card("Main bottleneck", bn_task, detail), unsafe_allow_html=True)

    cols = st.columns(5)
    tiles = [
        ("Cases analysed", f"{n_cases:,}", f"{len(df):,} step executions"),
        ("Median end-to-end time", fmt_minutes(median_cycle_min), "first start to last end"),
        ("Step SLA breaches", pct(df["sla_violation"].mean() * 100), "of step executions"),
        ("Cases with rework or skips", pct(n_exception_cases / n_cases * 100), f"{n_exception_cases:,} cases"),
        ("Process health index", f"{health['score']:.0f} / 100", ""),
    ]
    for col, (label, value, note) in zip(cols, tiles):
        status = health["status"] if label == "Process health index" else None
        col.markdown(kpi(label, value, note, status=status), unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("## Where the time goes")
        st.caption("Average minutes per step: working on it (blue) and waiting for it to start (light blue). "
                   "The bottleneck is shown in orange.")
        ordered = steps.reset_index()
        fig = go.Figure()
        fig.add_bar(y=ordered["task"], x=ordered["avg_duration"], orientation="h", name="Working",
                    marker_color=[WARN if t == bn_task else ACCENT for t in ordered["task"]],
                    hovertemplate="%{y}<br>Working: %{x:,.0f} min<extra></extra>")
        fig.add_bar(y=ordered["task"], x=ordered["avg_wait_before"], orientation="h",
                    name="Waiting to start", marker_color=ACCENT_SOFT,
                    hovertemplate="%{y}<br>Waiting: %{x:,.0f} min<extra></extra>")
        fig.update_layout(barmode="stack", yaxis=dict(autorange="reversed"),
                          xaxis_title="Average minutes per execution")
        st.plotly_chart(style_fig(fig, 330, legend=False), width="stretch", config=CHART_CONFIG, theme=None)

    with right:
        st.markdown("## End-to-end time per case")
        durations_h = case_durations["total_duration_minutes"] / 60
        cap = durations_h.quantile(0.99)
        shown = durations_h[durations_h <= cap]
        st.caption(f"Median {fmt_minutes(median_cycle_min)}. "
                   + (f"{len(durations_h) - len(shown)} case(s) above the 99th percentile are not shown."
                      if len(shown) < len(durations_h) else "All cases shown."))
        fig = px.histogram(shown, nbins=30, color_discrete_sequence=[ACCENT])
        fig.add_vline(x=median_cycle_min / 60, line_dash="dash", line_color=INK,
                      annotation_text="median", annotation_position="top")
        fig.update_layout(xaxis_title="Hours from first start to last end", yaxis_title="Cases")
        fig.update_traces(hovertemplate="%{x:.0f} h: %{y} cases<extra></extra>")
        style_fig(fig, 330, legend=False).update_layout(margin=dict(l=10, r=10, t=34, b=10))  # room for the label
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, theme=None)

    with st.expander("How the health index is calculated"):
        st.markdown("A heuristic index from 0 to 100, not a statistical model. Four measures are weighted "
                    "and subtracted from 100.")
        explain = pd.DataFrame({
            "Measure": list(WEIGHTS),
            "Current value": [f"{health['metrics'][name]:.0f}%" for name in WEIGHTS],
            "Weight": [f"{weight:.0%}" for weight in WEIGHTS.values()],
        })
        st.dataframe(explain, hide_index=True, width="stretch")
        st.caption("Bottleneck concentration measures how far the slowest step's share of step time exceeds "
                   "an even split across steps.")

    with st.expander("Data quality"):
        st.markdown(
            f"{quality['rows_used']:,} of {quality['rows_in']:,} rows were used. "
            f"Removed: {quality['unparseable_timestamps']:,} with unreadable timestamps and "
            f"{quality['end_before_start']:,} where the end time is before the start time."
        )

# Page: Bottlenecks
elif page == "Bottlenecks":
    st.title("Bottlenecks")
    st.markdown('<div class="kicker">Duration, waiting and SLA performance for every step.</div>',
                unsafe_allow_html=True)

    table = steps.reset_index().rename(columns={
        "task": "Step", "executions": "Runs", "avg_duration": "Average (min)",
        "median_duration": "Median (min)", "p90_duration": "90th pct (min)",
        "avg_wait_before": "Wait before (min)", "sla_target": "SLA target (min)",
        "sla_breach_pct": "SLA breaches", "total_hours": "Total time (h)",
    })
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Runs": st.column_config.NumberColumn(format="%d"),
            "Average (min)": st.column_config.NumberColumn(format="%.0f"),
            "Median (min)": st.column_config.NumberColumn(format="%.0f"),
            "90th pct (min)": st.column_config.NumberColumn(format="%.0f"),
            "Wait before (min)": st.column_config.NumberColumn(format="%.0f"),
            "SLA target (min)": st.column_config.NumberColumn(format="%.0f"),
            "SLA breaches": st.column_config.NumberColumn(format="%.0f%%"),
            "Total time (h)": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.download_button("Download step summary (CSV)", table.round(1).to_csv(index=False).encode(),
                       file_name="step_summary.csv", mime="text/csv")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("## Duration spread by step")
        st.caption("Box = middle 50% of runs, line = median. A long upper whisker means a few runs take far longer. "
                   "Log scale, so fast steps stay readable next to slow ones.")
        fig = px.box(df, x="task", y=df["duration_minutes"] / 60, color="task", points=False,
                     category_orders={"task": order},
                     color_discrete_map={t: (WARN if t == bn_task else ACCENT) for t in order})
        fig.update_layout(xaxis_title=None, yaxis_title="Time per run (log scale)")
        lo, hi = float(df["duration_minutes"].clip(lower=0.1).min()), float(df["duration_minutes"].max())
        ticks = [(1, "1 min"), (5, "5 min"), (15, "15 min"), (30, "30 min"), (60, "1 h"), (120, "2 h"),
                 (300, "5 h"), (600, "10 h"), (1440, "1 day"), (2880, "2 days"), (7200, "5 days")]
        ticks = [(m, label) for m, label in ticks if lo / 2 <= m <= hi * 2]
        fig.update_yaxes(type="log", tickvals=[m / 60 for m, _ in ticks], ticktext=[label for _, label in ticks],
                         minor=dict(showgrid=False))
        st.plotly_chart(style_fig(fig, 340, legend=False), width="stretch", config=CHART_CONFIG, theme=None)
    with right:
        st.markdown("## Wait before each step")
        handoff = find_handoff_delay(steps)
        st.caption("Average time between the previous step finishing and this one starting. "
                   + ("Orange marks a wait far above the other steps."
                      if handoff else "No step stands out: waits are similar across steps."))
        waits = steps.reset_index()
        fig = go.Figure(go.Bar(
            x=waits["task"], y=waits["avg_wait_before"],
            marker_color=[WARN if handoff and t == handoff["task"] else NEUTRAL for t in waits["task"]],
            hovertemplate="%{x}<br>%{y:,.0f} min<extra></extra>"))
        fig.update_layout(xaxis_title=None, yaxis_title="Average minutes")
        st.plotly_chart(style_fig(fig, 340, legend=False), width="stretch", config=CHART_CONFIG, theme=None)

    st.markdown("## What-if: speed up the bottleneck")
    st.caption(f"Reduce the duration of '{bn_task}' and see the effect on average end-to-end time. "
               "Assumes the saved time shortens the case directly.")
    default_cut = generate_recommendation(bn_task, float(bn["avg_duration"]), int(bn["executions"]))["target_reduction_pct"]
    cut = st.slider("Reduction in step duration", 0, 100, int(default_cut), 5, format="%d%%")
    current = float(case_durations["total_duration_minutes"].mean())
    saved = bn_total_minutes / n_cases * cut / 100
    simulated = max(0.0, current - saved)
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("Average end-to-end today", fmt_minutes(current)), unsafe_allow_html=True)
    c2.markdown(kpi("Average with change", fmt_minutes(simulated), f"{fmt_minutes(saved)} saved per case"),
                unsafe_allow_html=True)
    c3.markdown(kpi("Overall improvement", f"{saved / current * 100:.1f}%" if current else "0%"),
                unsafe_allow_html=True)

# Page: Process variants
elif page == "Process variants":
    st.title("Process variants")
    st.markdown('<div class="kicker">The distinct paths cases take through the process, and what each costs.</div>',
                unsafe_allow_html=True)

    default_limit = int(min(336, max(4, round(median_cycle_min / 60 * 1.5))))
    sla_hours = st.slider("Case SLA: maximum acceptable end-to-end time (hours)", 4, 336, default_limit, 1)

    variant_df = mine_variants(df)
    if variant_df.empty:
        st.warning("No variants could be mined from this dataset.")
        st.stop()

    standard_path = variant_df.iloc[0]["fingerprint"].split("|")
    variant_df["variant_type"] = variant_df["fingerprint"].apply(lambda fp: tag_variant_type(fp, standard_path))
    variant_df = score_variants(df, variant_df, sla_limit_hours=sla_hours)
    variant_df = generate_variant_recommendations(variant_df)

    total = int(variant_df["frequency"].sum())
    conformant = variant_df[variant_df["variant_type"] == "Conformant"]
    rework = variant_df[variant_df["variant_type"] == "Rework loop"]
    std_hours = float(conformant["avg_cycle_hours"].mean()) if len(conformant) else float("nan")
    rework_extra = (float((rework["avg_cycle_hours"] * rework["frequency"]).sum() / rework["frequency"].sum()) - std_hours
                    if len(rework) and len(conformant) else 0.0)

    cols = st.columns(4)
    cols[0].markdown(kpi("Distinct paths", f"{len(variant_df)}", "unique step sequences"), unsafe_allow_html=True)
    cols[1].markdown(kpi("Cases on the standard path", pct(conformant["frequency"].sum() / total * 100), "most common sequence"),
                     unsafe_allow_html=True)
    cols[2].markdown(kpi("Cases with a rework loop", pct(rework["frequency"].sum() / total * 100),
                         f"{int(rework['frequency'].sum())} cases"), unsafe_allow_html=True)
    cols[3].markdown(kpi("Extra time from rework", f"{rework_extra:+.1f} h" if len(rework) else "None",
                         "per case vs standard path"), unsafe_allow_html=True)

    st.markdown("## Paths")
    ranked = variant_df.sort_values("frequency", ascending=False)
    main = ranked[ranked["frequency"] >= max(2, int(total * 0.01))]
    if main.empty:
        main = ranked
    rare = ranked.drop(main.index)

    def table_rows(frame):
        return [{"variant": r.variant_label, "type": r.variant_type, "cases": int(r.frequency),
                 "share": r.frequency / total * 100, "avg_cycle": float(r.avg_cycle_hours),
                 "breach": float(r.sla_breach_rate) * 100, "wait": float(r.avg_wait_ratio) * 100}
                for r in frame.itertuples()]

    st.markdown(variants_table(table_rows(main)), unsafe_allow_html=True)
    if len(rare):
        with st.expander(f"Rare paths ({len(rare)} paths, {int(rare['frequency'].sum())} cases)"):
            st.markdown(variants_table(table_rows(rare)), unsafe_allow_html=True)

    st.markdown("## Path details")
    st.caption("Red steps are repeated (rework). Dashed steps were skipped. Purple steps are not part of the standard path.")
    for r in main.itertuples():
        meta = f"{int(r.frequency)} cases, {r.frequency / total * 100:.0f}% of total, {r.avg_cycle_hours:.1f} h average"
        action = ("The most common path. It is the baseline the other variants are compared against."
                  if r.variant_type == "Conformant" else r.recommendation)
        st.markdown(variant_card(r.variant_label, r.variant_type, meta, r.fingerprint.split("|"),
                                 standard_path, action), unsafe_allow_html=True)


    st.markdown("## Key insights")
    for line in generate_variant_insights(variant_df):
        st.markdown(f"- {line}")

    # one-off paths are noise, so they are not eligible to be called the highest risk
    min_cases = max(2, int(total * 0.01))
    deviant = variant_df[(variant_df["variant_type"] != "Conformant")
                         & (variant_df["frequency"] >= min_cases)].copy()
    if len(deviant):
        deviant["_risk"] = (0.5 * deviant["avg_cycle_hours"] / (deviant["avg_cycle_hours"].max() + 1e-9)
                            + 0.5 * deviant["sla_breach_rate"] / (deviant["sla_breach_rate"].max() + 1e-9))
        worst = deviant.loc[deviant["_risk"].idxmax()]
        saving = max(0.0, worst["avg_cycle_hours"] - std_hours)
        st.markdown(finding_card(
            "Highest-risk deviation",
            f"{worst['variant_label']} ({worst['variant_type']})",
            f"{int(worst['frequency'])} {'case' if int(worst['frequency']) == 1 else 'cases'} "
            f"({worst['frequency'] / total * 100:.0f}%), averaging "
            f"{worst['avg_cycle_hours']:.1f} h end-to-end with a {worst['sla_breach_rate'] * 100:.0f}% SLA breach rate. "
            f"If it matched the standard path, each case would save about {saving:.1f} h "
            f"({saving * worst['frequency']:.0f} h in total). Risk ranking weighs cycle time and breach rate equally.",
        ), unsafe_allow_html=True)

    st.markdown("## Cycle time by variant")
    chart_df = variant_df[variant_df["frequency"] >= 2].sort_values("avg_cycle_hours", ascending=False)
    if (variant_df["frequency"] < 2).any():
        st.caption("Single-case variants are left out here. They are listed under Rare paths above.")
    fig = px.bar(chart_df, x="variant_label", y="avg_cycle_hours", color="variant_type",
                 color_discrete_map=VARIANT_COLORS, text="avg_cycle_hours",
                 hover_data={"frequency": True, "variant_type": False})
    fig.update_traces(texttemplate="%{text:.1f} h", textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title=None, yaxis_title="Average hours, start to finish", legend_title_text=None)
    st.plotly_chart(style_fig(fig, 340), width="stretch", config=CHART_CONFIG, theme=None)

    with st.expander(f"Cases with rework or skipped steps ({n_exception_cases:,})"):
        if exceptions_df.empty:
            st.write("No cases deviate from the expected step order.")
        else:
            counts = exceptions_df.groupby("issue_type")["case_id"].nunique().rename("Cases").reset_index()
            counts["issue_type"] = counts["issue_type"].map({"loop": "Repeated step (rework)",
                                                             "deviation": "Out-of-order or skipped step"})
            st.dataframe(counts.rename(columns={"issue_type": "Issue"}), hide_index=True, width="stretch")
            readable = exceptions_df.assign(issue_type=exceptions_df["issue_type"].map({
                "loop": "Repeated step (rework)", "deviation": "Out-of-order or skipped step"}))
            st.dataframe(readable.rename(columns={"case_id": "Case", "issue_type": "Issue",
                                                  "affected_task": "Step"}),
                         hide_index=True, width="stretch")

# Page: Recommendations
else:
    st.title("Recommendations")
    st.markdown('<div class="kicker">What to fix first, why it matters, and a planning target.</div>',
                unsafe_allow_html=True)

    rec = generate_recommendation(bn_task, float(bn["avg_duration"]), int(bn["executions"]))
    if has_slow_tail(bn):
        rec["recommendation_text"] = "Fix the slow runs, not the typical ones: " + slow_tail_action(bn_task, bn)
    wait_share = bn["avg_wait_before"] / (bn["avg_wait_before"] + bn["avg_duration"]) * 100
    insight = build_decision_chain(bn_task, float(bn["avg_duration"]), float(wait_share), rec)
    st.markdown("## Primary bottleneck")
    st.markdown(chain_card([("Finding", insight["problem"]), ("Cause", insight["cause"]),
                            ("Action", insight["action"]), ("Target", insight["impact"])]),
                unsafe_allow_html=True)

    handoff = find_handoff_delay(steps)
    if handoff:
        ratio = handoff["work_ratio"]
        ratio_text = "much more than" if ratio > 50 else f"{ratio:.1f}x" if ratio < 10 else f"{ratio:.0f}x"
        st.markdown("## Biggest handoff delay")
        st.markdown(chain_card([
            ("Finding", f"Cases wait {fmt_minutes(handoff['avg_wait'])} on average before "
                        f"'{handoff['task']}' starts, {ratio_text} the time spent working on it. "
                        f"Other steps typically wait {fmt_minutes(handoff['typical_wait'])}."),
            ("Cause", "This is queueing time, not work: nothing is happening to the case while it waits."),
            ("Action", handoff_action(handoff["task"], handoff["sla_target"])),
        ]), unsafe_allow_html=True)

    st.markdown("## Priority order")
    st.caption("Steps ranked by total time consumed (average duration x number of runs).")
    for rank, action in enumerate(generate_priority_actions(task_stats, steps=steps), start=1):
        st.markdown(f"{rank}. {action}")

    st.markdown("## Automation candidates")
    candidates = [c for c in identify_automation_opportunities(df, task_stats) if c["task"] != bn_task]
    if candidates:
        st.markdown(data_table(
            ["Step", "Why it qualifies", "Suggested approach", "Minutes saved per case"],
            [[f"<b>{esc(c['task'])}</b>", esc(c["reason"]), esc(c["suggestion"]),
              f"{c['minutes_saved_per_case']:.0f}"] for c in candidates],
            numeric={3}), unsafe_allow_html=True)
        st.caption(f"Savings apply a planning assumption of {ASSUMED_REDUCTION:.0%} to each step's average duration. "
                   "They are estimates for discussion, not forecasts.")
    else:
        st.write("No other step is both high-volume and slow to start.")

    st.markdown("## Step risk index")
    risks = score_step_risk(df, task_stats)
    st.markdown(data_table(
        ["Step", "Risk index", "Drivers"],
        [[f"<b>{esc(r['task'])}</b>", risk_bar(float(r["risk_score"])),
          esc(", ".join(r["factors"]) or "No single factor stands out")] for r in risks]), unsafe_allow_html=True)
    st.caption("Weighted heuristic: 50% average duration, 30% variability, 20% volume, each scaled to the "
               "highest step. It ranks steps; it does not forecast breaches.")