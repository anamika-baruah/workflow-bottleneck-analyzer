"""
dashboard/app.py

Streamlit dashboard for analyzing workflow bottlenecks.

Run from the project root with:
    streamlit run dashboard/app.py
"""

import sys
import os
import tempfile
import io

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Make the project's src/ package importable regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_workflow_data          # noqa: E402
from src.preprocessing import (
    preprocess_workflow,
    compute_case_durations,
    detect_exception_flows,
)  # noqa: E402
from src.bottleneck_detector import (
    detect_bottlenecks,
    generate_recommendation,
    generate_priority_actions,
)  # noqa: E402
from src.risk_predictor import predict_bottleneck_risk  # noqa: E402
from src.health_analyzer import calculate_workflow_health  # noqa: E402
from src.context_analyzer import detect_workflow_context  # noqa: E402
from src.automation_engine import identify_automation_opportunities  # noqa: E402
from src.insight_engine import generate_smart_insight  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Workflow Bottleneck Analyzer",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 16px 20px;
    }
    /* Bottleneck alert */
    .bottleneck-alert {
        background: linear-gradient(135deg, #ff4b4b22 0%, #ff4b4b11 100%);
        border-left: 4px solid #ff4b4b;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 16px 0;
    }
    .bottleneck-alert h2 {
        margin: 0 0 8px 0;
        color: #ff4b4b;
    }
    .bottleneck-alert p {
        margin: 0;
        font-size: 1.05rem;
    }
    /* Landing page feature card */
    .feature-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
        text-align: center;
    }
    .feature-card h3 { margin: 8px 0 4px 0; font-size: 1.05rem; }
    .feature-card p  { margin: 0; font-size: 0.9rem; color: #aaa; }
    /* Upload validation banner */
    .col-valid   { color: #2ecc71; font-weight: 600; }
    .col-missing { color: #e74c3c; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════
st.title("🔍 AI Workflow Bottleneck Analyzer")
st.markdown(
    "Upload a workflow event-log CSV to automatically detect "
    "which step is slowing down your process."
)

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — File Upload & Navigation
# ═══════════════════════════════════════════════════════════════════════════
REQUIRED_COLS = {"case_id", "task", "start_time", "end_time", "user"}

with st.sidebar:
    st.header("📂 Data Source")

    uploaded_file = st.file_uploader(
        "Choose a workflow CSV file",
        type=["csv"],
        help="The CSV must contain: case_id, task, start_time, end_time, user",
    )

    # ── Quick column validation preview (shown immediately on upload) ──
    if uploaded_file is not None:
        try:
            peek_df = pd.read_csv(uploaded_file, nrows=0)
            uploaded_file.seek(0)          # rewind so downstream reads still work
            found_cols = set(peek_df.columns.str.strip())
            missing_cols = REQUIRED_COLS - found_cols

            if missing_cols:
                st.error(
                    f"**Missing columns:** {', '.join(sorted(missing_cols))}\n\n"
                    "Please fix your CSV and re-upload.",
                    icon="❌",
                )
                st.caption("Detected columns: " + ", ".join(sorted(found_cols)))
            else:
                st.success("✅ All required columns found!", icon="✅")
        except Exception as exc:
            st.warning(f"Could not peek at file: {exc}")
            missing_cols = set()
    else:
        missing_cols = {"_placeholder"}      # keeps upload_file block self-contained

    # ── Navigation (only after valid file) ────────────────────────────
    if uploaded_file is not None and not missing_cols:
        st.divider()
        st.header("🧭 Navigation")
        page = st.radio(
            "Select View:",
            ["Overview", "Bottleneck Analysis", "Exception Analysis", "Risk & Insights"],
            index=0,
        )
    else:
        page = "Overview"   # default; main block won't run if file is None/invalid

    st.divider()
    st.markdown(
        "**Expected columns:**\n"
        "- `case_id` — unique workflow instance\n"
        "- `task` — step name\n"
        "- `start_time` — when the step started\n"
        "- `end_time` — when the step finished\n"
        "- `user` — who performed the step"
    )

    # ── Sample data download ───────────────────────────────────────────
    st.divider()
    st.markdown("**Need sample data?**")

    sample_csv = """case_id,task,start_time,end_time,user
C001,Lead Created,2024-01-01 09:00,2024-01-01 09:03,alice
C001,Lead Reviewed,2024-01-01 09:03,2024-01-01 09:35,bob
C001,Manager Approval,2024-01-01 09:35,2024-01-01 11:55,carol
C001,Proposal Sent,2024-01-01 11:55,2024-01-01 12:55,alice
C001,Deal Closed,2024-01-01 12:55,2024-01-01 13:55,bob
C002,Lead Created,2024-01-02 10:00,2024-01-02 10:04,dave
C002,Lead Reviewed,2024-01-02 10:04,2024-01-02 10:40,alice
C002,Manager Approval,2024-01-02 10:40,2024-01-02 13:30,carol
C002,Proposal Sent,2024-01-02 13:30,2024-01-02 14:30,dave
C002,Deal Closed,2024-01-02 14:30,2024-01-02 15:30,alice
"""
    st.download_button(
        label="⬇️ Download Sample CSV",
        data=sample_csv,
        file_name="sample_workflow.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Landing page — shown when no file is uploaded yet
# ═══════════════════════════════════════════════════════════════════════════
if uploaded_file is None:

    st.markdown("---")

    # Hero banner
    st.markdown(
        """
        <div style="text-align:center; padding: 32px 0 16px 0;">
            <div style="font-size: 3.5rem;">📊</div>
            <h2 style="margin: 8px 0 4px 0;">Drop your workflow log to get started</h2>
            <p style="color: #888; font-size: 1.05rem;">
                Paste a CSV into the sidebar uploader and get instant AI-powered analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature grid
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.markdown(
            '<div class="feature-card"><div style="font-size:2rem">⚠️</div>'
            '<h3>Bottleneck Detection</h3>'
            '<p>Pinpoints the exact step slowing your process with duration stats</p></div>',
            unsafe_allow_html=True,
        )
    with fc2:
        st.markdown(
            '<div class="feature-card"><div style="font-size:2rem">❤️</div>'
            '<h3>Health Score</h3>'
            '<p>0–100 score weighted by SLA compliance, exceptions & wait time</p></div>',
            unsafe_allow_html=True,
        )
    with fc3:
        st.markdown(
            '<div class="feature-card"><div style="font-size:2rem">🎯</div>'
            '<h3>Risk Prediction</h3>'
            '<p>Flags high-variability tasks before they escalate into SLA breaches</p></div>',
            unsafe_allow_html=True,
        )
    with fc4:
        st.markdown(
            '<div class="feature-card"><div style="font-size:2rem">🤖</div>'
            '<h3>Smart Insights</h3>'
            '<p>Context-aware recommendations with projected savings estimates</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # How it works + CSV format side by side
    hw_col, fmt_col = st.columns([1, 1])

    with hw_col:
        st.subheader("🚀 How it works")
        st.markdown(
            """
1. **Upload** your workflow event-log CSV using the sidebar uploader.
2. The system **validates** your columns instantly and previews the data.
3. Navigate between **4 analysis views** using the sidebar menu.
4. Export insights or download sample data to test the system immediately.

> 💡 Use the **Download Sample CSV** button in the sidebar to try the system right now.
            """
        )

    with fmt_col:
        st.subheader("📋 Required CSV format")
        st.markdown("Your file must contain these **5 columns** (header row required):")

        format_df = pd.DataFrame({
            "Column": ["case_id", "task", "start_time", "end_time", "user"],
            "Type": ["string", "string", "datetime", "datetime", "string"],
            "Example": ["CASE-001", "Manager Approval", "2024-01-15 09:00", "2024-01-15 11:30", "alice"],
            "Description": [
                "Unique ID for each workflow instance",
                "Name of the workflow step",
                "When this step started",
                "When this step finished",
                "Who performed the step",
            ],
        })
        st.dataframe(format_df, hide_index=True, use_container_width=True)

        st.caption("Datetime format: `YYYY-MM-DD HH:MM` or ISO 8601")

    st.markdown("---")

    # Accordion: common issues
    with st.expander("❓ Common upload issues & how to fix them"):
        st.markdown(
            """
**Wrong column names?**
Rename your headers to exactly: `case_id`, `task`, `start_time`, `end_time`, `user`.
Column names are case-sensitive.

**Datetime parsing errors?**
Make sure timestamps use a standard format like `2024-01-15 09:00:00` or `2024-01-15T09:00`.
Avoid locale-specific formats such as `15/01/24`.

**File won't upload?**
The uploader accepts `.csv` files only. If you have an Excel file, export it to CSV first via
*File → Save As → CSV (Comma delimited)*.

**Too many rows?**
The system handles files up to ~50 MB comfortably. For larger logs, pre-filter to a
representative time window before uploading.
            """
        )

    st.stop()   # don't render the rest of the page until a file is present


# ═══════════════════════════════════════════════════════════════════════════
# Guard: stop if uploaded file has missing columns (already shown error above)
# ═══════════════════════════════════════════════════════════════════════════
if missing_cols:
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════════════

# ── Demo mode check ─────────────────────────────────────────────────────
sample_filenames = {"workflow_logs.csv", "broken_workflow_logs.csv", "data.csv", "sample_workflow.csv"}
if uploaded_file.name in sample_filenames:
    st.info("⚠️ **Demo Mode**: Using simulated workflow data for demonstration purposes.", icon="💡")

# ── Load & validate ──────────────────────────────────────────────────────
try:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    with st.spinner("⏳ Loading and validating your data…"):
        df_raw = load_workflow_data(tmp_path)

except (FileNotFoundError, ValueError) as exc:
    st.error(f"❌ **Data loading error:** {exc}")
    st.stop()
finally:
    if "tmp_path" in locals() and os.path.exists(tmp_path):
        os.unlink(tmp_path)

# ── Quick data-quality banner ────────────────────────────────────────────
n_rows   = len(df_raw)
n_cases  = df_raw["case_id"].nunique()
n_tasks  = df_raw["task"].nunique()
date_min = df_raw["start_time"].min().strftime("%Y-%m-%d")
date_max = df_raw["end_time"].max().strftime("%Y-%m-%d")

st.success(
    f"✅ **{uploaded_file.name}** loaded — "
    f"**{n_rows:,}** events · **{n_cases:,}** cases · "
    f"**{n_tasks}** unique steps · Date range: {date_min} → {date_max}",
    icon="📂",
)

# ── Preprocess ───────────────────────────────────────────────────────────
with st.spinner("⚙️ Preprocessing workflow data…"):
    df = preprocess_workflow(df_raw)
    case_durations = compute_case_durations(df)
    exceptions_df, total_exception_cases = detect_exception_flows(df)

# ── Detect bottlenecks ───────────────────────────────────────────────────
with st.spinner("🔍 Detecting bottlenecks…"):
    task_stats, bottleneck_task, total_bn_time, bottleneck_pct = detect_bottlenecks(df)


# ═══════════════════════════════════════════════════════════════════════════
# Page Content Routing
# ═══════════════════════════════════════════════════════════════════════════

if page == "Overview":
    # 1. Workflow Health Score
    st.markdown("---")
    st.header("📊 Workflow Health Score")

    health_data = calculate_workflow_health(df, bottleneck_pct, total_exception_cases)

    h_score  = health_data["score"]
    h_status = health_data["status"]
    h_interp = health_data["interpretation"]

    if h_score >= 80:
        st.success(f"**Current Status:** {h_status}", icon="✅")
    elif h_score >= 50:
        st.warning(f"**Current Status:** {h_status}", icon="⚠️")
    else:
        st.error(f"**Current Status:** {h_status}", icon="🔴")

    hcol1, hcol2 = st.columns([1, 4])
    hcol1.metric("Health Score", f"{h_score}/100")
    with hcol2:
        st.markdown(f"**AI Interpretation:** {h_interp}")
        if health_data.get("top_contributors"):
            st.markdown("**🔍 Key Contributors to Low Score:**")
            for contributor in health_data["top_contributors"]:
                st.markdown(f"* {contributor}")

    # 2. Workflow Context
    st.markdown("---")
    st.header("🧠 Workflow Context")
    context = detect_workflow_context(df)

    ctx_col1, ctx_col2 = st.columns(2)
    ctx_col1.info(f"**Workflow Type:** {context['workflow_type']}", icon="🏷️")

    with ctx_col2:
        st.markdown("**Process Breakdown:**")
        if context["human_tasks"]:
            st.markdown(f"👤 **Human Tasks:** {', '.join(context['human_tasks'])}")
        if context["system_tasks"]:
            st.markdown(f"⚙️ **System Tasks:** {', '.join(context['system_tasks'])}")

    st.markdown("---")
    st.header("📈 Workflow Overview")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    mcol1.metric("📊 Total Events",    f"{len(df):,}")
    mcol2.metric("📁 Unique Cases",    f"{df['case_id'].nunique():,}")
    mcol3.metric("📋 Workflow Steps",  f"{df['task'].nunique()}")
    mcol4.metric("⚠ Top Bottleneck",  bottleneck_task)

    st.markdown("---")
    st.header("📉 Duration Distribution")
    st.markdown("Shows the distribution of total end-to-end time for all cases.")

    fig_hist = px.histogram(
        case_durations,
        x="total_duration_minutes",
        nbins=20,
        title="<b>Case Completion Time Distribution</b>",
        labels={"total_duration_minutes": "Total Duration (minutes)"},
        color_discrete_sequence=["#636efa"],
    )
    fig_hist.update_layout(
        xaxis_title="End-to-End Duration (minutes)",
        yaxis_title="Number of Cases",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(t=50, b=40, l=40, r=40),
    )
    st.plotly_chart(fig_hist, width="stretch")

    # SLA Compliance
    st.markdown("---")
    st.header("⏱ SLA Compliance Analysis")
    st.markdown("Monitors task execution against defined Service Level Agreements (SLA).")

    SLA_MAP = {
        "Lead Created": 5, "Lead Reviewed": 30, "Manager Approval": 120,
        "Proposal Sent": 60, "Deal Closed": 60
    }

    sla_data = df.groupby("task").agg(
        avg_dur=("duration_minutes", "mean"),
        violations=("sla_violation", "sum"),
        total=("sla_violation", "count")
    ).reset_index()
    sla_data["violation_pct"] = (sla_data["violations"] / sla_data["total"]) * 100
    sla_data["target_sla"] = sla_data["task"].map(SLA_MAP)
    sla_data = sla_data.sort_values("violation_pct", ascending=False)

    scol1, scol2 = st.columns([2, 3])

    with scol1:
        st.subheader("📋 Compliance Summary")
        table_display = sla_data[["task", "target_sla", "avg_dur", "violation_pct"]].copy()
        table_display.columns = ["Task", "SLA Target (min)", "Avg Duration", "Violation %"]
        st.dataframe(
            table_display.style.format({
                "Avg Duration": "{:,.1f}m",
                "Violation %": "{:.1f}%",
                "SLA Target (min)": "{:,.0f}m"
            }).highlight_between(left=80, right=100, subset=["Violation %"], color="#ff4b4b33"),
            width="stretch", hide_index=True
        )
        top_violator = sla_data.iloc[0]
        if top_violator["violation_pct"] > 50:
            st.warning(
                f"**{top_violator['task']}** exceeds SLA in **{top_violator['violation_pct']:.1f}%** of cases, "
                "indicating a critical performance issue requiring attention.",
                icon="⏱️"
            )
        else:
            st.success("SLA compliance is within acceptable parameters.", icon="✅")

    with scol2:
        st.subheader("📊 Violation Rate by Step")
        fig_sla = px.bar(
            sla_data, x="task", y="violation_pct",
            title="<b>SLA Violation % per Workflow Step</b>",
            labels={"violation_pct": "Violation Rate (%)", "task": "Workflow Step"},
            color_discrete_sequence=["#e74c3c"]
        )
        fig_sla.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=350)
        st.plotly_chart(fig_sla, width="stretch")

    st.divider()
    with st.expander("🔎 Preview Raw Data", expanded=False):
        st.dataframe(df.head(50), width="stretch", hide_index=True)


elif page == "Bottleneck Analysis":
    st.markdown("---")
    st.header("⚠ Bottleneck Analysis")
    avg_val    = task_stats.loc[bottleneck_task, "avg_duration_minutes"]
    median_val = task_stats.loc[bottleneck_task, "median_duration_minutes"]

    if df["task"].nunique() > 1 and avg_val > 0:
        st.markdown(
            f"""
            <div class="bottleneck-alert">
                <h2>⚠️ {bottleneck_task}</h2>
                <p>
                    This step has the <strong>highest average duration</strong>
                    across all workflow cases.<br>
                    Average: <strong>{avg_val:,.2f} min</strong> &nbsp;|&nbsp;
                    Median: <strong>{median_val:,.2f} min</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.success("No significant bottleneck detected.", icon="✅")

    st.markdown("---")
    st.header("🎯 Bottleneck Impact Analysis")
    total_workflow_time = df["duration_minutes"].sum()
    other_time          = total_workflow_time - total_bn_time
    bn_case_count       = int(task_stats.loc[bottleneck_task, "count"])

    ic1, ic2, ic3 = st.columns(3)
    ic1.metric("⏳ Total Time in Bottleneck",      f"{total_bn_time:,.0f} min")
    ic2.metric("📉 Share of Total Workflow Time",  f"{bottleneck_pct:.1f}%")
    ic3.metric("🗂️ Cases Affected",               f"{bn_case_count:,}")

    donut_labels = [f"🔴 {bottleneck_task}", "🔵 All Other Steps"]
    donut_values = [total_bn_time, max(other_time, 0)]
    donut_colors = ["#ff4b4b", "#636efa"]

    fig_donut = go.Figure(
        go.Pie(
            labels=donut_labels,
            values=donut_values,
            hole=0.55,
            marker=dict(colors=donut_colors, line=dict(color="#1e1e2f", width=2)),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:,.0f} min<extra></extra>",
        )
    )
    fig_donut.update_layout(
        title="<b>Bottleneck Share of Total Workflow Time</b>",
        height=380,
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_donut, width="stretch")

    st.markdown("---")
    st.header("📋 Task Duration Statistics")
    timeline_df = task_stats.reset_index().rename(columns={
        "task": "Task",
        "avg_duration_minutes": "Avg Duration (min)",
        "median_duration_minutes": "Median Duration (min)",
        "count": "Executions",
    })
    timeline_df["Avg Duration (min)"] = timeline_df["Avg Duration (min)"].round(1)
    timeline_df["Median Duration (min)"] = timeline_df["Median Duration (min)"].round(1)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stats Table", "📈 Timeline", "📉 Bar Chart", "📦 Box Plot"])

    with tab1:
        st.dataframe(
           timeline_df,
        column_config={
            "Avg Duration (min)": st.column_config.NumberColumn(format="%.1f"),
            "Median Duration (min)": st.column_config.NumberColumn(format="%.1f"),
        },
        hide_index=True,
        width="stretch"
    )

    with tab2:
        fig_line = px.line(
            timeline_df, x="Task", y="Avg Duration (min)",
            title="<b>Average Duration Timeline by Step</b>",
            markers=True, color_discrete_sequence=["#636efa"]
        )
        fig_line.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=380)
        st.plotly_chart(fig_line, width="stretch")

    with tab3:
        timeline_df["Color"] = timeline_df["Task"].apply(
            lambda t: "#ff4b4b" if t == bottleneck_task else "#636efa"
        )
        fig_bar = px.bar(
            timeline_df, x="Task", y="Avg Duration (min)",
            text="Avg Duration (min)",
            title="<b>Mean Execution Time by Step</b>",
            labels={"Avg Duration (min)": "Minutes", "Task": "Workflow Step"},
            color="Color", color_discrete_map="identity"
        )
        fig_bar.update_traces(texttemplate='%{text:.1f}m', textposition='outside')
        fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=400, showlegend=False)
        st.plotly_chart(fig_bar, width="stretch")

    with tab4:
        fig_box = px.box(
            df, x="task", y="duration_minutes",
            color="task", points="outliers",
            title="<b>Variance and Outlier Detection</b>",
            labels={"duration_minutes": "Duration (min)", "task": "Step Name"}
        )
        fig_box.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=400, showlegend=False)
        st.plotly_chart(fig_box, width="stretch")


elif page == "Exception Analysis":
    st.markdown("---")
    st.header("⚠ Workflow Exceptions")
    st.markdown("Identifies cases with process violations like rework loops or unexpected task sequences.")

    total_unique_cases = df["case_id"].nunique()
    exception_pct      = (total_exception_cases / total_unique_cases) * 100 if total_unique_cases > 0 else 0

    if total_exception_cases > 0:
        st.warning(
            f"**{total_exception_cases}** exception cases detected (**{exception_pct:.1f}%** of all workflows).\n\n"
            "Repeated task patterns detected. This indicates rework and process inefficiency.",
            icon="⚠️"
        )
        st.dataframe(
            exceptions_df,
            column_config={"case_id": "Case ID", "issue_type": "Issue Type", "affected_task": "Affected Task"},
            use_container_width=True, hide_index=True,
        )
    else:
        st.success("No major workflow exceptions detected. Process is stable.", icon="✅")

    st.subheader("💡 Exception Insights")
    if total_exception_cases == 0:
        st.info("**Process is Healthy.**\n\nNo task loops or sequence deviations detected.", icon="✅")
    elif exception_pct < 5:
        st.info(f"**Low Exception Rate Detected ({exception_pct:.1f}%).**\n\nThe process is mostly stable.", icon="💡")
    else:
        st.warning(f"**High Exception Rate Detected ({exception_pct:.1f}%).**\n\nProcess inefficiency is significant.", icon="⚠️")


elif page == "Risk & Insights":
    # 1. Smart Insights
    st.markdown("---")
    st.header("🧠 Smart Insights")

    avg_val  = task_stats.loc[bottleneck_task, "avg_duration_minutes"]
    wp_stats = df.groupby("task")[["processing_time_minutes", "waiting_time_minutes"]].mean()
    bn_wait  = wp_stats.loc[bottleneck_task, "waiting_time_minutes"]
    bn_total = wp_stats.loc[bottleneck_task].sum()
    wait_pct = (bn_wait / bn_total) * 100 if bn_total > 0 else 0

    health_data   = calculate_workflow_health(df, bottleneck_pct, total_exception_cases)
    risk_results  = predict_bottleneck_risk(df, task_stats)
    rec_data      = generate_recommendation(bottleneck_task, avg_val, int(task_stats.loc[bottleneck_task, "count"]))

    smart_insight = generate_smart_insight(
        bottleneck_task=bottleneck_task,
        avg_duration=avg_val,
        wait_pct=wait_pct,
        recommendation=rec_data,
        task_stats=task_stats,
        health_score=health_data["score"],
        risk_results=risk_results,
        total_exception_cases=total_exception_cases,
        total_cases=df["case_id"].nunique(),
    )

    with st.container():
        st.info(
            f"**Problem:** {smart_insight['problem']}\n\n"
            f"**Root Cause:** {smart_insight['cause']}\n\n"
            f"**Recommended Action:** {smart_insight['action']}\n\n"
            f"**Expected Impact:** {smart_insight['impact']}",
            icon="💡"
        )
        if smart_insight.get("secondary_findings"):
            with st.expander("📌 Additional findings"):
                for finding in smart_insight["secondary_findings"]:
                    st.markdown(f"- {finding}")

    st.markdown("---")

    # 2. Risk Alerts
    st.subheader("⚠️ Workflow Risk Alerts")
    high_risks = [r for r in risk_results if r["risk_score"] >= 50]

    if high_risks:
        for r in high_risks:
            with st.container():
                st.warning(
                    f"**{r['task']}** — Risk Score: **{r['risk_score']} / 100**\n\n{r['explanation']}",
                    icon="⚠️"
                )
                if r.get("factors"):
                    st.markdown("**Why this risk?**")
                    for factor in r["factors"]:
                        st.markdown(f"* {factor}")
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.success("No high-risk steps detected.", icon="✅")

    # 3. Priority Actions
    st.subheader("🎯 Priority Actions")
    priority_list = generate_priority_actions(task_stats, total_bn_time)

    if priority_list:
        for i, action in enumerate(priority_list, 1):
            st.markdown(f"{i}. **{action}**")
    else:
        st.info("No immediate priority actions identified.", icon="💡")

    # 4. Automation Opportunities
    st.markdown("---")
    st.subheader("⚙️ Automation Opportunities")
    automation_ops = identify_automation_opportunities(df, task_stats)

    if automation_ops:
        for op in automation_ops[:3]:
            with st.expander(f"Opportunity: **{op['task']}**", expanded=True):
                st.markdown(f"**🔴 Reason:** {op['reason']}")
                st.markdown(f"**✅ Suggestion:** {op['suggestion']}")
                if op.get("impact_text"):
                    st.success(f"**💡 Expected Impact:** {op['impact_text']}")
    else:
        st.success("Your process appears highly automated with no major manual delays detected.", icon="⚙️")


# ═══════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 0.85rem; padding: 20px 0;">
        <strong>AI Workflow Intelligence System</strong><br>
        Process analytics, bottleneck detection, and optimization insights
    </div>
    """,
    unsafe_allow_html=True
)