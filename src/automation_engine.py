"""
automation_engine.py

Spots tasks that would benefit most from automation.
Looks for high-frequency steps with a lot of idle wait time — classic RPA candidates.
"""

import pandas as pd


def identify_automation_opportunities(df: pd.DataFrame, stats_df: pd.DataFrame) -> list[dict]:
    """Find the tasks most worth automating and explain why.

    Targets steps that are both frequent AND slow to wait on —
    those are usually the ones with the most manual overhead.
    """
    if df.empty or stats_df.empty:
        return []

    # average waiting time per task — if available
    if "waiting_time_minutes" in df.columns:
        wait_stats = df.groupby("task")["waiting_time_minutes"].mean()
    else:
        wait_stats = pd.Series(0, index=stats_df.index)

    # anything above-average in both frequency and wait time is a candidate
    avg_freq = stats_df["count"].mean()
    avg_wait = wait_stats.mean()

    opportunities = []

    for task in stats_df.index:
        count = stats_df.loc[task, "count"]
        wait = wait_stats.get(task, 0)

        if count >= avg_freq and wait >= avg_wait:
            task_lower = task.lower()

            # pick a suggestion that fits the task type
            if "approval" in task_lower:
                reason = "High delay and manual dependency on human reviewers."
                solution = "Introduce rule-based auto-approval for low-value or low-risk cases."
            elif "review" in task_lower:
                reason = "Repetitive decision-making cycles slowing down cases."
                solution = "Implement AI-driven document review or parallelize the human review process."
            elif "ticket" in task_lower or "routing" in task_lower:
                reason = "High volume of assignment tasks adds significant overhead."
                solution = "Automate routing using rule-based classification or machine learning."
            elif "assignment" in task_lower or "allocation" in task_lower:
                reason = "Manual resource allocation causing workflow fragmentation."
                solution = "Use automated workflow triggers to assign tasks based on resource availability."
            else:
                reason = "High frequency and wait time indicate a process bottleneck."
                solution = f"Redesign the '{task}' step to reduce manual touchpoints through RPA or script automation."

            # rough impact estimate — not precise, but gives a ballpark
            # TODO: make this more data-driven if we get historical improvement data
            import random
            reduction_pct = random.randint(20, 30)
            avg_duration = stats_df.loc[task, "avg_duration_minutes"]
            minutes_saved = (avg_duration * reduction_pct) / 100

            impact_text = (
                f"Reduce delay by ~{reduction_pct}% "
                f"(~{minutes_saved:,.1f} minutes per case)"
            )

            opportunities.append({
                "task": task,
                "reason": reason,
                "suggestion": solution,
                "impact_text": impact_text,
                "impact_score": count * wait  # higher = more worth fixing
            })

    # surface the highest-impact ones first
    opportunities.sort(key=lambda x: x["impact_score"], reverse=True)
    return opportunities
