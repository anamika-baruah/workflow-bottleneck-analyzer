"""
health_analyzer.py

Produces a single 0-100 score that summarizes how well the workflow is running.
Combines SLA compliance, bottleneck intensity, exceptions, and idle time.
"""

import pandas as pd


def compute_health_score(
    df: pd.DataFrame,
    bottleneck_pct: float,
    total_exception_cases: int
) -> dict:
    """Compute overall workflow health on a 0-100 scale.

    Score is built by penalizing four things:
    - SLA violations (30%)
    - Bottleneck impact (30%)
    - Exception rate (20%)
    - Waiting time ratio (20%)

    100 = everything running smoothly, 0 = complete chaos.
    """
    if df.empty:
        return {"score": 0, "status": "No Data", "interpretation": "No data available."}

    # quick check for SLA breaches
    sla_violation_rate = (df["sla_violation"].sum() / len(df)) * 100

    # bottleneck_pct already tells us how dominant the worst step is
    bottleneck_score = bottleneck_pct

    # what % of cases hit some kind of exception?
    total_cases = df["case_id"].nunique()
    exception_rate = (total_exception_cases / total_cases) * 100 if total_cases > 0 else 0

    # how much of total time is just waiting in a queue vs. actual work?
    total_duration = df["duration_minutes"].sum()
    total_waiting = df["waiting_time_minutes"].sum()
    waiting_ratio = (total_waiting / total_duration) * 100 if total_duration > 0 else 0

    # weighted penalties — each pulls the score down
    penalties = [
        ("SLA violation rate", sla_violation_rate * 0.3),
        ("Bottleneck impact", bottleneck_score * 0.3),
        ("Workflow exceptions", exception_rate * 0.2),
        ("Waiting-to-processing ratio", waiting_ratio * 0.2)
    ]

    total_penalty = sum(p[1] for p in penalties)
    health_score = max(0, min(100, 100 - total_penalty))

    # surface the top 2 things dragging the score down
    sorted_penalties = sorted(penalties, key=lambda x: x[1], reverse=True)
    top_contributors = [p[0] for p in sorted_penalties if p[1] > 0][:2]

    if health_score >= 80:
        status = "Healthy"
        interpretation = "Workflow is performing optimally with minimal delays and high compliance."
    elif health_score >= 50:
        status = "Moderate"
        interpretation = "Workflow is moderately efficient but affected by specific operational constraints."
    else:
        status = "Critical"
        interpretation = "Workflow efficiency is low. Significant bottlenecks and non-compliance detected."

    return {
        "score": round(health_score, 1),
        "status": status,
        "interpretation": interpretation,
        "top_contributors": top_contributors,
        "metrics": {
            "sla_viol_rate": round(sla_violation_rate, 1),
            "btl_impact": round(bottleneck_score, 1),
            "exc_rate": round(exception_rate, 1),
            "wait_ratio": round(waiting_ratio, 1)
        }
    }


# keep backward-compatible alias — dashboard imports calculate_workflow_health
calculate_workflow_health = compute_health_score
