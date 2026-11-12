"""
preprocessing.py

Cleans raw workflow event logs and derives the fields every analysis needs.

Definitions used throughout the project
---------------------------------------
duration_minutes
    Time the step itself took (end_time - start_time).
waiting_time_minutes
    Handoff wait: the gap between the previous step finishing and this step
    starting, within the same case. The first step of a case has no wait.
    Both values come straight from the timestamps; nothing is estimated.
"""

from __future__ import annotations

import math

import pandas as pd


def infer_step_order(df: pd.DataFrame) -> list[str]:
    """Infer the usual step order from the data.

    Each task is ranked by the average position of its first appearance in a
    case, so the order adapts to any workflow instead of assuming one.
    """
    ordered = df.sort_values(["case_id", "start_time"]).copy()
    ordered["_pos"] = ordered.groupby("case_id").cumcount()
    first_pos = ordered.groupby(["case_id", "task"])["_pos"].min().reset_index()
    mean_pos = first_pos.groupby("task")["_pos"].mean()
    return sorted(mean_pos.index, key=lambda t: (mean_pos[t], t))


def default_sla_minutes(df: pd.DataFrame, multiplier: float = 2.0) -> dict[str, int]:
    """Suggest an SLA target per step: `multiplier` x its median duration.

    Real SLAs come from the business, so this is only a starting point that
    users can overwrite. Values are rounded up to the next 5 minutes.
    """
    durations = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60
    medians = durations.groupby(df["task"]).median()
    return {
        task: max(5, int(math.ceil(median * multiplier / 5) * 5))
        for task, median in medians.items()
    }


def preprocess_workflow(
    df: pd.DataFrame,
    sla_minutes: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Parse timestamps, drop unusable rows and add duration / wait / SLA fields.

    Parameters
    ----------
    df : pd.DataFrame
        Raw log with case_id, task, start_time, end_time.
    sla_minutes : dict, optional
        SLA target per task in minutes. Tasks not listed fall back to
        ``default_sla_minutes``.

    Returns
    -------
    pd.DataFrame
        Sorted by case and start time. ``df.attrs["quality"]`` holds counts of
        rows removed and ``df.attrs["sla_minutes"]`` the SLA table in use.
    """
    df = df.copy()
    rows_in = len(df)

    for col in ("start_time", "end_time"):
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    unparseable = int(df[["start_time", "end_time"]].isna().any(axis=1).sum())
    df = df.dropna(subset=["start_time", "end_time"])

    # A step that ends before it starts is a data error, not a fast step.
    reversed_mask = df["end_time"] < df["start_time"]
    reversed_rows = int(reversed_mask.sum())
    df = df[~reversed_mask]

    df = df.sort_values(["case_id", "start_time", "end_time"]).reset_index(drop=True)

    df["duration_minutes"] = (
        (df["end_time"] - df["start_time"]).dt.total_seconds() / 60
    )

    previous_end = df.groupby("case_id")["end_time"].shift(1)
    df["waiting_time_minutes"] = (
        ((df["start_time"] - previous_end).dt.total_seconds() / 60)
        .clip(lower=0)
        .fillna(0)
    )
    # kept as an explicit name so charts read "processing vs waiting"
    df["processing_time_minutes"] = df["duration_minutes"]

    sla = default_sla_minutes(df)
    if sla_minutes:
        sla.update({k: float(v) for k, v in sla_minutes.items() if k in sla})
    df["sla_threshold"] = df["task"].map(sla)
    df["sla_violation"] = df["duration_minutes"] > df["sla_threshold"]

    df.attrs["quality"] = {
        "rows_in": rows_in,
        "rows_used": len(df),
        "unparseable_timestamps": unparseable,
        "end_before_start": reversed_rows,
    }
    df.attrs["sla_minutes"] = sla
    return df


def compute_case_durations(df: pd.DataFrame) -> pd.DataFrame:
    """End-to-end time per case: earliest start to latest end."""
    case_aggs = df.groupby("case_id").agg(
        first_start=("start_time", "min"),
        last_end=("end_time", "max"),
    )
    total_mins = (case_aggs["last_end"] - case_aggs["first_start"]).dt.total_seconds() / 60
    return pd.DataFrame(
        {"case_id": case_aggs.index, "total_duration_minutes": total_mins.to_numpy()}
    ).reset_index(drop=True)


def detect_exception_flows(
    df: pd.DataFrame,
    expected_order: list[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Find rework loops and out-of-sequence steps.

    * loop      - a step appears more than once in a case
    * deviation - the case starts on the wrong step, or a step does not follow
                  the expected order (skipped or reordered steps)

    ``expected_order`` defaults to the order inferred from the data.
    """
    if expected_order is None:
        expected_order = infer_step_order(df)
    order_index = {task: i for i, task in enumerate(expected_order)}

    exceptions: list[dict] = []
    ordered = df.sort_values(["case_id", "start_time"])

    for case_id, group in ordered.groupby("case_id"):
        tasks = group["task"].tolist()
        if not tasks:
            continue

        seen: set[str] = set()
        for task in tasks:
            if task in seen:
                exceptions.append(
                    {"case_id": case_id, "issue_type": "loop", "affected_task": task}
                )
            seen.add(task)

        if order_index.get(tasks[0], -1) != 0:
            exceptions.append(
                {"case_id": case_id, "issue_type": "deviation", "affected_task": tasks[0]}
            )

        for prev_task, task in zip(tasks, tasks[1:]):
            if task == prev_task:
                continue
            if order_index.get(task, -1) != order_index.get(prev_task, -1) + 1:
                exceptions.append(
                    {"case_id": case_id, "issue_type": "deviation", "affected_task": task}
                )

    columns = ["case_id", "issue_type", "affected_task"]
    exceptions_df = pd.DataFrame(exceptions, columns=columns)
    return exceptions_df, int(exceptions_df["case_id"].nunique())
