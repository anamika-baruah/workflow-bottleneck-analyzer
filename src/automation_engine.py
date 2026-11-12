"""
automation_engine.py

Shortlists steps that are frequent and slow to start, which are the usual
candidates for automation or rule-based routing.
"""

from __future__ import annotations

import pandas as pd

# Planning assumption used for every estimate; results are labelled as such.
ASSUMED_REDUCTION = 0.25


def identify_automation_opportunities(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    assumed_reduction: float = ASSUMED_REDUCTION,
) -> list[dict]:
    """Return candidate steps, ordered by (executions x average handoff wait).

    A step qualifies when its volume and its handoff wait are both at or above
    the process average. Time-saved figures apply ``assumed_reduction`` to the
    step's average duration and are deterministic for the same input.
    """
    if df.empty or stats_df.empty:
        return []

    if "waiting_time_minutes" in df.columns:
        wait_stats = df.groupby("task")["waiting_time_minutes"].mean()
    else:
        wait_stats = pd.Series(0.0, index=stats_df.index)

    avg_freq = stats_df["count"].mean()
    avg_wait = wait_stats.mean()
    opportunities = []

    for task in stats_df.index:
        count = stats_df.loc[task, "count"]
        wait = wait_stats.get(task, 0.0)
        if count < avg_freq or wait < avg_wait:
            continue

        name = task.lower()
        if "approval" in name:
            reason = "High volume with long waits for a human decision."
            solution = "Introduce rule-based auto-approval for low-risk cases."
        elif "review" in name:
            reason = "Repeated review cycles hold cases back."
            solution = "Standardise review criteria or review in parallel."
        elif "ticket" in name or "routing" in name or "assign" in name:
            reason = "High-volume assignment work adds overhead and queue time."
            solution = "Route automatically with rule-based classification."
        else:
            reason = "High volume and above-average waiting before this step starts."
            solution = f"Reduce manual touchpoints in '{task}' with templates or scripted automation."

        avg_duration = stats_df.loc[task, "avg_duration_minutes"]
        minutes_saved = avg_duration * assumed_reduction
        opportunities.append({
            "task": task,
            "reason": reason,
            "suggestion": solution,
            "impact_text": (
                f"At an assumed {assumed_reduction:.0%} reduction: "
                f"about {minutes_saved:,.0f} minutes saved per case"
            ),
            "minutes_saved_per_case": round(float(minutes_saved), 1),
            "impact_score": float(count * wait),
        })

    opportunities.sort(key=lambda o: o["impact_score"], reverse=True)
    return opportunities
