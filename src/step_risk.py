"""
step_risk.py

Ranks steps by how likely they are to cause delays.
A weighted heuristic, not a trained model: 50% average duration,
30% variability, 20% volume, each scaled to the highest step.
"""

import pandas as pd

WEIGHTS = {"duration": 50, "variability": 30, "volume": 20}
FLAG_RATIO = 1.2  # a factor is flagged when a step is 20% above the all-step average


def score_step_risk(df: pd.DataFrame, stats_df: pd.DataFrame) -> list[dict]:
    """Return one dict per step (task, risk_score 0-100, factors), highest risk first."""
    if "task" not in df.columns or "duration_minutes" not in df.columns or stats_df.empty:
        return []

    per_step = df.groupby("task")["duration_minutes"].agg(["std", "count"]).fillna(0)
    avg = stats_df["avg_duration_minutes"]
    std, count = per_step["std"], per_step["count"]

    def scaled(series: pd.Series) -> pd.Series:
        top = series.max()
        return series / top if top > 0 else series * 0

    score = (WEIGHTS["duration"] * scaled(avg)
             + WEIGHTS["variability"] * scaled(std)
             + WEIGHTS["volume"] * scaled(count))

    risks = []
    for task in stats_df.index:
        factors = []
        if avg[task] > FLAG_RATIO * avg.mean():
            factors.append("High average duration")
        if std[task] > FLAG_RATIO * std.mean():
            factors.append("High variability in execution time")
        if count[task] > FLAG_RATIO * count.mean():
            factors.append("High volume")
        risks.append({"task": task, "risk_score": round(float(score[task]), 1), "factors": factors})

    return sorted(risks, key=lambda r: r["risk_score"], reverse=True)