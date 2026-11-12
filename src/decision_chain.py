"""Turns the detected bottleneck into one Finding, Cause, Action, Target chain."""

from __future__ import annotations


def build_decision_chain(
    bottleneck_task: str,
    avg_duration: float,
    wait_pct: float,
    recommendation: dict,
) -> dict:
    """Build the decision chain for the bottleneck step.

    ``wait_pct`` is the share of this step's elapsed time (handoff wait plus
    the step itself) spent waiting for it to start.
    """
    problem = f"{bottleneck_task} averages {avg_duration:,.0f} minutes per execution."

    if wait_pct >= 50:
        cause = (
            f"{wait_pct:.0f}% of the elapsed time around this step is waiting to start, "
            "which points to a queue or handoff problem."
        )
    else:
        cause = (
            f"Only {wait_pct:.0f}% of the elapsed time is waiting to start, so the delay "
            "is in the work inside the step itself."
        )

    action = recommendation.get("recommendation_text", "Reduce manual effort in this step.")
    target = recommendation.get("target_reduction_pct", 20)
    impact = (
        f"Planning target: cut this step's duration by {target}%. "
        "Use the what-if tool to test other values."
    )
    return {"problem": problem, "cause": cause, "action": action, "impact": impact}