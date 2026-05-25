"""
src/insight_engine.py

Synthesises data from bottleneck detection, risk prediction, and health analysis
into a structured, business-ready 'Smart Insight'.

Upgrade over the original 31-line version:
- Reads 6 data signals instead of 2
- Produces secondary findings (up to 3 extra observations)
- Uses severity tiers to vary tone and urgency
- Generates specific, numeric impact statements rather than generic templates
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_smart_insight(
    bottleneck_task: str,
    avg_duration: float,
    wait_pct: float,
    recommendation: dict,
    task_stats: pd.DataFrame | None = None,
    health_score: float | None = None,
    risk_results: list[dict] | None = None,
    total_exception_cases: int = 0,
    total_cases: int = 0,
) -> dict:
    """Combine multiple analysis signals into a structured decision chain.

    Parameters
    ----------
    bottleneck_task : str
        Name of the detected bottleneck step.
    avg_duration : float
        Average duration of the bottleneck step in minutes.
    wait_pct : float
        Percentage of bottleneck time spent waiting (idle), 0-100.
    recommendation : dict
        Output from ``generate_recommendation()``.
        Must contain ``recommendation_text`` and ``estimated_savings``.
    task_stats : pd.DataFrame, optional
        Per-task summary stats (avg_duration_minutes, count, median).
        Used to add context about second-worst tasks.
    health_score : float, optional
        Overall Workflow Health Score (0-100).
    risk_results : list[dict], optional
        Output from ``predict_bottleneck_risk()``.
    total_exception_cases : int
        Number of cases with detected exceptions/rework loops.
    total_cases : int
        Total unique workflow cases in the dataset.

    Returns
    -------
    dict with keys:
        problem           — concise problem statement (string)
        cause             — root-cause analysis (string)
        action            — recommended action (string)
        impact            — expected outcome with numeric estimate (string)
        secondary_findings — list of up to 3 additional observations (list[str])
    """

    # ── 1. Severity tier ────────────────────────────────────────────────
    # Drives tone: critical (>4h avg), high (1-4h), moderate (<1h).
    severity = _severity_tier(avg_duration)

    # ── 2. Problem statement ────────────────────────────────────────────
    problem = _build_problem(bottleneck_task, avg_duration, severity, task_stats)

    # ── 3. Root-cause analysis ──────────────────────────────────────────
    cause = _build_cause(bottleneck_task, avg_duration, wait_pct, task_stats)

    # ── 4. Recommended action ───────────────────────────────────────────
    action = recommendation.get(
        "recommendation_text",
        f"Investigate and reduce manual handoffs in the '{bottleneck_task}' step."
    )

    # ── 5. Impact statement ─────────────────────────────────────────────
    impact = _build_impact(
        bottleneck_task, avg_duration, wait_pct,
        recommendation.get("estimated_savings", "15–25%"),
        task_stats, health_score,
    )

    # ── 6. Secondary findings ───────────────────────────────────────────
    secondary_findings = _build_secondary_findings(
        bottleneck_task=bottleneck_task,
        task_stats=task_stats,
        health_score=health_score,
        risk_results=risk_results,
        total_exception_cases=total_exception_cases,
        total_cases=total_cases,
    )

    return {
        "problem": problem,
        "cause": cause,
        "action": action,
        "impact": impact,
        "secondary_findings": secondary_findings,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _severity_tier(avg_duration: float) -> str:
    """Return 'critical', 'high', or 'moderate' based on average duration."""
    if avg_duration > 240:
        return "critical"
    if avg_duration > 60:
        return "high"
    return "moderate"


def _build_problem(
    bottleneck_task: str,
    avg_duration: float,
    severity: str,
    task_stats: pd.DataFrame | None,
) -> str:
    """Build a specific problem statement that includes comparative context."""
    prefix_map = {
        "critical": "🔴 **Critical bottleneck detected:**",
        "high":     "🟠 **Significant bottleneck detected:**",
        "moderate": "🟡 **Process delay identified:**",
    }
    prefix = prefix_map.get(severity, "⚠️")

    base = (
        f"{prefix} **{bottleneck_task}** averages **{avg_duration:,.1f} minutes** per execution"
    )

    # Add how much slower it is vs the median step
    if task_stats is not None and not task_stats.empty and len(task_stats) > 1:
        other_avgs = task_stats["avg_duration_minutes"].drop(bottleneck_task, errors="ignore")
        if not other_avgs.empty:
            median_other = other_avgs.median()
            if median_other > 0:
                ratio = avg_duration / median_other
                base += f" — **{ratio:.1f}× slower** than the median workflow step ({median_other:,.0f} min)"

    base += "."
    return base


def _build_cause(
    bottleneck_task: str,
    avg_duration: float,
    wait_pct: float,
    task_stats: pd.DataFrame | None,
) -> str:
    """Diagnose the most likely root cause using available signals."""
    task_lower = bottleneck_task.lower()

    # High idle/waiting ratio → human handoff or queue dependency
    if wait_pct > 65:
        cause = (
            f"**{wait_pct:.0f}%** of time in this step is idle waiting — "
            "strongly indicating a manual handoff, queue dependency, or approval gate "
            "without defined SLA enforcement."
        )

    # Approval/review tasks that are slow → reviewer bottleneck
    elif "approval" in task_lower or "review" in task_lower:
        cause = (
            f"Approval and review steps typically stall on reviewer availability or unclear "
            f"decision criteria. With an average of {avg_duration:,.0f} min, cases are "
            "likely queuing for human action rather than being actively processed."
        )

    # Creation/generation tasks → data dependency or template absence
    elif any(kw in task_lower for kw in ("creat", "generat", "build", "prepar")):
        cause = (
            f"Document creation and generation steps slow down when input data must be "
            f"manually gathered. An average of {avg_duration:,.0f} min suggests missing "
            "templates, pre-fetched data, or system integrations."
        )

    # Routing/assignment → rules-based decision overhead
    elif any(kw in task_lower for kw in ("routing", "assignment", "assign", "dispatch")):
        cause = (
            "Routing and assignment logic executed manually introduces variable delay. "
            f"At {avg_duration:,.0f} min average, this step is a strong candidate "
            "for rule-based or ML-driven automation."
        )

    # High variability fallback — check if stats show wide spread
    else:
        if task_stats is not None and bottleneck_task in task_stats.index:
            avg  = task_stats.loc[bottleneck_task, "avg_duration_minutes"]
            med  = task_stats.loc[bottleneck_task, "median_duration_minutes"]
            if avg > 0 and (avg - med) / avg > 0.3:
                cause = (
                    f"The gap between average ({avg:,.0f} min) and median ({med:,.0f} min) "
                    "suggests high execution variability — a small number of cases are taking "
                    "significantly longer, pulling up the mean. Outlier investigation is recommended."
                )
            else:
                cause = (
                    f"Consistently high execution time ({avg_duration:,.0f} min average) points "
                    "to structural process complexity rather than isolated outliers. "
                    "Task decomposition or automation should be explored."
                )
        else:
            cause = (
                f"High execution complexity or unclear task scope is likely driving the "
                f"{avg_duration:,.0f}-minute average. Process mapping and task decomposition "
                "are recommended first steps."
            )

    return cause


def _build_impact(
    bottleneck_task: str,
    avg_duration: float,
    wait_pct: float,
    estimated_savings: str,
    task_stats: pd.DataFrame | None,
    health_score: float | None,
) -> str:
    """Build a specific, numeric impact projection."""
    parts: list[str] = []

    # Core savings estimate from recommendation engine
    parts.append(
        f"Resolving this bottleneck is projected to reduce **{bottleneck_task}** "
        f"execution time by **{estimated_savings}**."
    )

    # Absolute time saving per execution
    try:
        low_pct = float(estimated_savings.split("–")[0].replace("%", "").strip()) / 100
        time_saved = avg_duration * low_pct
        parts.append(
            f"At the low end of that range, each case would save at least "
            f"**{time_saved:,.0f} minutes** at this step."
        )
    except (ValueError, IndexError):
        pass

    # Compound savings if high-frequency step
    if task_stats is not None and bottleneck_task in task_stats.index:
        count = int(task_stats.loc[bottleneck_task, "count"])
        if count > 50:
            try:
                low_pct = float(estimated_savings.split("–")[0].replace("%", "").strip()) / 100
                total_saved = avg_duration * low_pct * count
                parts.append(
                    f"Across the **{count:,}** recorded executions, this translates to "
                    f"a potential **{total_saved:,.0f} minutes** (~{total_saved/60:,.0f} hours) "
                    "saved in total."
                )
            except (ValueError, IndexError):
                pass

    # Health score improvement note
    if health_score is not None and health_score < 75:
        parts.append(
            f"With the current Workflow Health Score at **{health_score}/100**, "
            "addressing this bottleneck is the single highest-impact action available."
        )

    return " ".join(parts)


def _build_secondary_findings(
    bottleneck_task: str,
    task_stats: pd.DataFrame | None,
    health_score: float | None,
    risk_results: list[dict] | None,
    total_exception_cases: int,
    total_cases: int,
) -> list[str]:
    """Generate up to 3 additional observations from ancillary signals."""
    findings: list[str] = []

    # Finding 1: Second-worst task (if close to the bottleneck)
    if task_stats is not None and len(task_stats) > 1:
        others = task_stats.drop(bottleneck_task, errors="ignore").sort_values(
            "avg_duration_minutes", ascending=False
        )
        if not others.empty:
            second_task = others.index[0]
            second_avg  = others.iloc[0]["avg_duration_minutes"]
            bn_avg      = task_stats.loc[bottleneck_task, "avg_duration_minutes"]
            if bn_avg > 0 and second_avg / bn_avg > 0.6:
                findings.append(
                    f"**{second_task}** is a secondary concern at {second_avg:,.0f} min average "
                    f"({second_avg/bn_avg*100:.0f}% of the bottleneck's duration) — "
                    "address it after the primary bottleneck is resolved."
                )

    # Finding 2: High-risk tasks beyond the bottleneck
    if risk_results:
        high_risk_others = [
            r for r in risk_results
            if r["risk_score"] >= 60 and r["task"] != bottleneck_task
        ]
        if high_risk_others:
            names = ", ".join(f"**{r['task']}**" for r in high_risk_others[:2])
            findings.append(
                f"{names} {'carry' if len(high_risk_others) > 1 else 'carries'} a high risk score "
                "independent of the primary bottleneck and may cause SLA breaches during peak load."
            )

    # Finding 3: Exception rate warning
    if total_cases > 0:
        exception_pct = (total_exception_cases / total_cases) * 100
        if exception_pct >= 10:
            findings.append(
                f"**{exception_pct:.0f}%** of cases show rework loops or sequence exceptions. "
                "This amplifies the impact of every bottleneck because affected cases traverse "
                "slow steps multiple times."
            )
        elif exception_pct >= 5:
            findings.append(
                f"A **{exception_pct:.0f}% exception rate** is borderline — monitor for upward trends "
                "and investigate the most common rework triggers."
            )

    return findings[:3]