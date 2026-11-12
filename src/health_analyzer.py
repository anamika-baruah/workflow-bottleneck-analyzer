"""
health_analyzer.py

A single 0-100 index summarising how smoothly a process runs.

This is a heuristic index, not a statistical model: four penalties with fixed
weights are subtracted from 100. The weights are exposed in ``WEIGHTS`` so
they can be changed or discussed with a client.
"""

from __future__ import annotations

import pandas as pd

WEIGHTS = {
    "SLA breaches": 0.30,
    "Bottleneck concentration": 0.30,
    "Rework and deviations": 0.20,
    "Handoff waiting": 0.20,
}


def compute_health_score(
    df: pd.DataFrame,
    bottleneck_pct: float,
    total_exception_cases: int,
) -> dict:
    """Score workflow health from 0 (poor) to 100 (smooth).

    Inputs, all expressed as percentages:
    * SLA breaches - share of step executions over their SLA target
    * Bottleneck concentration - how far the slowest step's share of total
      step time exceeds an even split across steps
    * Rework and deviations - share of cases with a loop or skipped step
    * Handoff waiting - share of elapsed time spent waiting between steps
    """
    if df.empty:
        return {"score": 0.0, "status": "No data", "interpretation": "No data available.",
                "top_contributors": [], "metrics": {}}

    sla_rate = df["sla_violation"].mean() * 100

    n_steps = df["task"].nunique()
    even_share = 100 / n_steps if n_steps else 100
    if n_steps > 1:
        concentration = max(0.0, bottleneck_pct - even_share) / (100 - even_share) * 100
    else:
        concentration = 0.0

    n_cases = df["case_id"].nunique()
    exception_rate = total_exception_cases / n_cases * 100 if n_cases else 0.0

    work = df["duration_minutes"].sum()
    wait = df["waiting_time_minutes"].sum()
    wait_ratio = wait / (work + wait) * 100 if (work + wait) > 0 else 0.0

    components = {
        "SLA breaches": sla_rate,
        "Bottleneck concentration": concentration,
        "Rework and deviations": exception_rate,
        "Handoff waiting": wait_ratio,
    }
    penalties = {name: value * WEIGHTS[name] for name, value in components.items()}
    score = max(0.0, min(100.0, 100 - sum(penalties.values())))

    drivers = sorted(penalties, key=penalties.get, reverse=True)
    top_contributors = [
        f"{name}: {components[name]:.0f}%" for name in drivers[:2] if penalties[name] > 0
    ]

    if score >= 80:
        status, interpretation = "Healthy", "Few delays or deviations; the process runs close to plan."
    elif score >= 60:
        status, interpretation = "Needs attention", "Generally working, with specific steps or paths losing time."
    else:
        status, interpretation = "At risk", "Significant delay, rework or SLA pressure across the process."

    return {
        "score": round(score, 1),
        "status": status,
        "interpretation": interpretation,
        "top_contributors": top_contributors,
        "metrics": {k: float(round(v, 1)) for k, v in components.items()},
    }


calculate_workflow_health = compute_health_score
