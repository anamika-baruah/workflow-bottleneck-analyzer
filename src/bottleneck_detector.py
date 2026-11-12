"""
bottleneck_detector.py

Ranks workflow steps and turns the numbers into plain-language findings.
The bottleneck is the step with the highest average duration; step
statistics also include handoff wait and SLA breach rate for context.
"""

from __future__ import annotations

import pandas as pd

from src.formatting import fmt_minutes


def detect_bottlenecks(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, str, float, float]:
    """Return (task_stats, bottleneck_task, bottleneck_total_minutes, bottleneck_pct).

    ``bottleneck_pct`` is the step's share of all step time (not elapsed time).
    """
    required = {"task", "duration_minutes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns for bottleneck detection: "
            + ", ".join(sorted(missing))
        )

    task_stats = (
        df.groupby("task")["duration_minutes"]
        .agg(
            avg_duration_minutes="mean",
            median_duration_minutes="median",
            count="count",
        )
        .round(2)
        .sort_values("avg_duration_minutes", ascending=False)
    )

    bottleneck_task: str = task_stats["avg_duration_minutes"].idxmax()
    total_bn_time = round(
        float(df.loc[df["task"] == bottleneck_task, "duration_minutes"].sum()), 2
    )
    total_time = float(df["duration_minutes"].sum())
    bottleneck_pct = round(total_bn_time / total_time * 100, 2) if total_time > 0 else 0.0
    return task_stats, bottleneck_task, total_bn_time, bottleneck_pct


def summarize_steps(df: pd.DataFrame, order: list[str] | None = None) -> pd.DataFrame:
    """One row per step: volume, duration, handoff wait and SLA breach rate."""
    summary = df.groupby("task").agg(
        executions=("duration_minutes", "count"),
        avg_duration=("duration_minutes", "mean"),
        median_duration=("duration_minutes", "median"),
        p90_duration=("duration_minutes", lambda s: s.quantile(0.9)),
        avg_wait_before=("waiting_time_minutes", "mean"),
        sla_target=("sla_threshold", "first"),
        sla_breach_pct=("sla_violation", lambda s: s.mean() * 100),
        total_hours=("duration_minutes", lambda s: s.sum() / 60),
    )
    if order:
        summary = summary.reindex([t for t in order if t in summary.index])
    return summary


def generate_bottleneck_insight(
    bottleneck_task: str, avg_duration: float, num_executions: int
) -> tuple[str, str]:
    """Short explanation and suggested direction for the bottleneck step."""
    explanation = (
        f"'{bottleneck_task}' averages {avg_duration:,.1f} minutes per execution "
        f"across {num_executions:,} executions, the longest of any step."
    )
    name = bottleneck_task.lower()
    if "approval" in name or "review" in name:
        improvement = "Add approval rules for low-risk cases or review in parallel."
    elif "create" in name or "generat" in name:
        improvement = "Pre-fill required data or use templates to speed this step up."
    else:
        improvement = "Look for manual touchpoints that can be automated or reassigned."
    return explanation, improvement


def generate_recommendation(
    task_name: str, avg_duration: float, execution_count: int
) -> dict:
    """Suggest an action for a step and a planning target for improvement.

    ``target_reduction_pct`` is a planning assumption, not a prediction: it is
    higher when the step is both slow and frequent. The dashboard's what-if
    control lets the user test other values.
    """
    name = task_name.lower()
    if "approval" in name:
        action = "Auto-approve low-risk cases and add a second approver to cover peaks."
    elif "review" in name:
        action = "Set clear review criteria and run reviews in parallel."
    elif "routing" in name or "assignment" in name:
        action = "Route automatically using business rules."
    else:
        action = f"Reduce manual effort in '{task_name}' by simplifying or automating it."

    signals = 0
    if avg_duration > 60:
        signals += 1
    if execution_count > 100:
        signals += 1
    target = {0: 15, 1: 20, 2: 25}[signals]
    return {"recommendation_text": action, "target_reduction_pct": target}


def find_handoff_delay(steps: pd.DataFrame) -> dict | None:
    """Return the step with an unusually long wait before it starts, or None.

    A wait only counts as a finding when it is at least an hour and at least
    twice the typical wait of the other steps. Without that rule a chart or a
    recommendation would single out a step for a difference of a few minutes.
    """
    waits = steps["avg_wait_before"]
    if len(waits) < 3:
        return None
    task = waits.idxmax()
    wait = float(waits[task])
    typical = float(waits.drop(task).median())
    if wait < 60 or wait < 2 * max(typical, 1.0):
        return None
    work = float(steps.loc[task, "avg_duration"])
    return {
        "task": task,
        "avg_wait": wait,
        "typical_wait": typical,
        "work_ratio": wait / work if work > 0 else float("inf"),
        "sla_target": float(steps.loc[task, "sla_target"]),
    }


def handoff_action(task: str, sla_target: float) -> str:
    """Suggested fix for a long queue in front of a step."""
    name = task.lower()
    if any(word in name for word in ("assign", "dispatch", "rout")):
        return ("Assign automatically on arrival (round-robin or skill-based) and alert "
                "when an item sits unassigned for longer than the SLA target.")
    if any(word in name for word in ("approv", "review", "confirm")):
        return ("Send automatic reminders, name a backup owner, and escalate items still "
                f"waiting after {fmt_minutes(sla_target)}.")
    return ("Make ownership explicit, alert on items waiting longer than the SLA target, "
            "and review capacity at this handoff.")


def has_slow_tail(row) -> bool:
    """True when the slowest 10% of runs take at least 2.5x the median."""
    return bool(row["median_duration"] > 0 and row["p90_duration"] >= 2.5 * row["median_duration"])


def slow_tail_action(task: str, row) -> str:
    """Advice for a step whose problem is a few very slow runs, not every run."""
    text = (
        f"a slow tail ({fmt_minutes(row['p90_duration'])} at the 90th percentile against a "
        f"{fmt_minutes(row['median_duration'])} median). Escalate runs that pass "
        f"{fmt_minutes(row['sla_target'])} and look for what the slowest cases have in common."
    )
    name = task.lower()
    if "approval" in name:
        text += " Auto-approve low-risk cases."
    elif "review" in name:
        text += " Review in parallel."
    return text


def generate_priority_actions(
    task_stats: pd.DataFrame,
    total_bn_time: float = 0.0,
    steps: pd.DataFrame | None = None,
) -> list[str]:
    """Top three steps by total time consumed (average x executions).

    When the per-step summary is supplied, each action names what is actually
    wrong with that step (a queue before it, or a slow tail) instead of a
    generic suggestion.
    """
    if task_stats.empty:
        return []
    ranked = (task_stats["avg_duration_minutes"] * task_stats["count"]).sort_values(
        ascending=False
    )
    actions = []
    for task in ranked.head(3).index:
        name = task.lower()
        row = steps.loc[task] if steps is not None and task in steps.index else None

        if row is not None and row["avg_wait_before"] >= 60 and row["avg_wait_before"] >= row["avg_duration"]:
            actions.append(
                f"{task}: cases wait {fmt_minutes(row['avg_wait_before'])} before it starts, "
                f"more than the {fmt_minutes(row['avg_duration'])} of work. "
                + handoff_action(task, float(row["sla_target"]))
            )
        elif row is not None and has_slow_tail(row):
            actions.append(f"{task}: " + slow_tail_action(task, row))
        elif row is not None:
            actions.append(
                f"{task}: takes {fmt_minutes(row['avg_duration'])} per run with "
                f"{fmt_minutes(row['avg_wait_before'])} of waiting before it, and neither stands "
                "out. Worth a look after the steps above."
            )
        elif "approval" in name:
            actions.append(f"{task}: auto-approve low-risk cases")
        elif "review" in name:
            actions.append(f"{task}: review in parallel")
        else:
            actions.append(f"{task}: standardise and automate repeat work")
    return actions