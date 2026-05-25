"""
risk_predictor.py

Flags tasks that look like they could become a problem.
Risk = combination of slow avg duration + high variability + frequency.
Nothing fancy — a weighted heuristic that works well in practice.
"""

import pandas as pd


def predict_bottleneck_risk(df: pd.DataFrame, stats_df: pd.DataFrame) -> list[dict]:
    """Score each task from 0-100 based on how risky it looks.

    Higher score = more likely to cause future SLA breaches or bottlenecks.
    Weights: 50% avg duration, 30% variability, 20% volume.
    """
    if "task" not in df.columns or "duration_minutes" not in df.columns:
        return []

    # std dev tells us how unpredictable the task is — high variance = bad
    task_metrics = df.groupby("task")["duration_minutes"].agg(["std", "count"]).fillna(0)

    # baselines to compare each task against
    global_avg_dur = stats_df["avg_duration_minutes"].mean()
    global_avg_std = task_metrics["std"].mean()
    global_avg_count = task_metrics["count"].mean()

    risks = []

    # normalize against max values so scores stay in 0-100 range
    max_avg = stats_df["avg_duration_minutes"].max() if not stats_df.empty else 1
    max_std = task_metrics["std"].max() if not task_metrics.empty else 1

    if max_avg <= 0: max_avg = 1
    if max_std <= 0: max_std = 1

    for task in stats_df.index:
        avg_dur = stats_df.loc[task, "avg_duration_minutes"]
        std_dur = task_metrics.loc[task, "std"]
        count_val = task_metrics.loc[task, "count"]

        # weighted risk score — optimize later if needed
        dur_score = (avg_dur / max_avg) * 50   # slowness is the biggest factor
        var_score = (std_dur / max_std) * 30    # unpredictability is second
        vol_score = (count_val / task_metrics["count"].max()) * 20  # frequency last

        risk_score = round(dur_score + var_score + vol_score, 1)

        # flag which signals contributed to the score
        factors = []
        if avg_dur > global_avg_dur * 1.2:
            factors.append("High average duration")
        if std_dur > global_avg_std * 1.2:
            factors.append("High variability in execution time")
        if count_val > global_avg_count * 1.2:
            factors.append("High frequency of occurrence")

        if risk_score >= 70:
            explanation = "Critical bottleneck risk: Combination of high impact and instability."
        elif risk_score >= 40:
            explanation = "Moderate operational risk: Monitor for process efficiency."
        else:
            explanation = "Low procedural risk: Tasks are performing within normal parameters."

        risks.append({
            "task": task,
            "risk_score": risk_score,
            "explanation": explanation,
            "factors": factors
        })

    risks.sort(key=lambda x: x["risk_score"], reverse=True)
    return risks
