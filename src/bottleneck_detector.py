"""
bottleneck_detector.py

Figures out which task is slowing down the whole workflow.
Uses average duration as the main signal — simple but effective.
"""

import pandas as pd


def detect_bottlenecks(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, str, float, float]:
    """Detect the primary bottleneck task and measure its impact.

    The bottleneck is the task with the highest average duration — the step
    where work items spend the most time before moving forward.

    Once found, we calculate how much of the total workflow time it eats up.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed workflow log. Needs 'task' and 'duration_minutes'.

    Returns
    -------
    tuple[pd.DataFrame, str, float, float]
        task_stats, bottleneck_task name, total time in bottleneck, bottleneck %.
    """

    required = {"task", "duration_minutes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for bottleneck detection: "
            f"{', '.join(sorted(missing))}"
        )

    # compute per-task stats — avg, median, and count
    task_stats = (
        df.groupby("task")["duration_minutes"]
        .agg(
            avg_duration_minutes="mean",
            median_duration_minutes="median",
            count="count",
        )
    )

    task_stats = task_stats.round(2)
    task_stats = task_stats.sort_values("avg_duration_minutes", ascending=False)

    # highest avg duration = bottleneck
    # median is there for sanity check — high mean with low median often means outliers
    bottleneck_task: str = task_stats["avg_duration_minutes"].idxmax()

    # how much total time does this bottleneck consume across all cases?
    total_bn_time: float = round(
        df.loc[df["task"] == bottleneck_task, "duration_minutes"].sum(), 2
    )

    total_workflow_time: float = df["duration_minutes"].sum()

    # guard against divide-by-zero on empty data
    if total_workflow_time > 0:
        bottleneck_pct: float = round(
            (total_bn_time / total_workflow_time) * 100, 2
        )
    else:
        bottleneck_pct = 0.0

    print("📊  Task Duration Statistics (sorted by avg duration):\n")
    print(task_stats.to_string())
    print(
        f"\n🔴  Detected bottleneck → '{bottleneck_task}' "
        f"(avg {task_stats.loc[bottleneck_task, 'avg_duration_minutes']:.2f} min)"
    )
    print(
        f"📌  Impact: {total_bn_time:,.0f} min total "
        f"({bottleneck_pct:.1f}% of all workflow time)"
    )
    return task_stats, bottleneck_task, total_bn_time, bottleneck_pct


def generate_bottleneck_insight(
    bottleneck_task: str, avg_duration: float, num_executions: int
) -> tuple[str, str]:
    """Write a plain-language explanation of why a task is the bottleneck.

    Returns a short description and a suggested fix.
    """
    explanation = (
        f"The '{bottleneck_task}' step is slowing down the overall process. "
        f"It takes an average of {avg_duration:,.1f} minutes to complete, "
        f"and it has been executed {num_executions:,} times. This high duration "
        f"causes a backlog in the workflow timeline."
    )

    task_lower = bottleneck_task.lower()
    if "approval" in task_lower or "review" in task_lower:
        improvement = "Consider introducing automated approval rules for low-risk cases or adding additional reviewers."
    elif "create" in task_lower or "generat" in task_lower:
        improvement = "Consider pre-fetching required data or using templates to accelerate generation."
    else:
        improvement = "Consider automating parts of this step or reallocating resources to reduce the workload."

    return explanation, improvement


def generate_recommendation(
    task_name: str, avg_duration: float, execution_count: int
) -> dict:
    """Build a context-aware recommendation for the bottleneck task.

    Uses task name + duration/frequency signals to produce a concise suggestion.
    Simple heuristic — not a model, but works well for common patterns.
    """
    task_lower = task_name.lower()
    parts: list[str] = []

    # pick suggestion based on what kind of task this is
    if "approval" in task_lower:
        parts.append(
            "Automate approval for low-risk cases or introduce parallel reviewers."
        )
    elif "review" in task_lower:
        parts.append(
            "Reduce review cycles or introduce clear guidelines to speed up decision making."
        )
    elif "routing" in task_lower or "assignment" in task_lower:
        parts.append(
            "Implement automated routing using business rules or AI-based classification."
        )
    else:
        parts.append(
            f"Consider redesigning or automating the '{task_name}' step to reduce manual effort."
        )

    # layer in data-driven context signals
    signals_triggered = 0

    if avg_duration > 300:  # this is where most delays come from — flag it hard
        parts.append(
            "This step is significantly slower than others and requires immediate intervention."
        )
        signals_triggered += 1
    elif avg_duration > 60:
        parts.append(
            "Duration is above average — reducing manual intervention here will yield measurable gains."
        )
        signals_triggered += 1

    if execution_count > 100:
        # high volume means even small speed-ups compound quickly
        parts.append(
            "This step occurs frequently, so even a small speed-up will have a high compound impact."
        )
        signals_triggered += 1

    recommendation_text = " ".join(parts)

    # estimated savings scale with how many signals fired
    if signals_triggered >= 2:
        estimated_savings = "25–35%"
    elif signals_triggered == 1:
        estimated_savings = "20–30%"
    else:
        estimated_savings = "15–25%"

    return {
        "recommendation_text": recommendation_text,
        "estimated_savings": estimated_savings,
    }


def generate_priority_actions(task_stats: pd.DataFrame, total_bn_time: float) -> list[str]:
    """Return the top 3 action items ranked by aggregate time wasted.

    Priority score = avg_duration * count — basically total time eaten by that task.
    """
    if task_stats.empty:
        return []

    priority_df = task_stats.copy()
    # total time spent = best proxy for where intervention has the most impact
    priority_df["priority_score"] = priority_df["avg_duration_minutes"] * priority_df["count"]
    priority_df = priority_df.sort_values("priority_score", ascending=False)

    actions = []

    for task_name in priority_df.head(3).index:
        task_lower = task_name.lower()

        if "approval" in task_lower:
            actions.append("Automate approval for low-risk cases")
        elif "review" in task_lower:
            actions.append("Introduce parallel reviewers to reduce wait time")
        elif "create" in task_lower or "generat" in task_lower:
            actions.append("Use templates to reduce manual generation steps")
        elif "assignment" in task_lower or "routing" in task_lower:
            actions.append("Implement automated rule-based task routing")
        else:
            actions.append(f"Standardize and automate the '{task_name}' process")

    return actions[:3]
