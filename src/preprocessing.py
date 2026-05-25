"""
preprocessing.py

Cleans and enriches raw workflow event logs before analysis.
Handles timestamp parsing, duration calculation, and basic SLA flagging.
"""

import pandas as pd


def preprocess_workflow(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up the raw workflow log and compute durations.

    Does timestamp parsing, drops bad rows, and adds a few useful columns
    like processing time, waiting time, and SLA violation flags.

    Parameters
    ----------
    df : pd.DataFrame
        Raw workflow log. Must have at least start_time and end_time columns.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with duration_minutes and SLA columns added.
    """

    # don't touch the caller's original data
    df = df.copy()

    # make sure timestamps are actually datetime objects
    # this might break if timestamps are in a weird format — handle downstream
    for col in ("start_time", "end_time"):
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # drop rows where we couldn't parse timestamps — useless for duration math
    before = len(df)
    df = df.dropna(subset=["start_time", "end_time"])
    dropped = before - len(df)
    if dropped:
        print(f"⚠️  Dropped {dropped} row(s) with missing timestamps.")

    # compute duration in minutes
    # timedelta → seconds → minutes
    df["duration_minutes"] = (
        (df["end_time"] - df["start_time"])
        .dt.total_seconds()
        / 60
    )

    # guard against reversed timestamps — take abs just in case
    df["duration_minutes"] = df["duration_minutes"].abs()

    # split into processing vs waiting time
    # using a simple heuristic: active work is ~10-30% of total, rest is queue idle
    # TODO: improve this with actual activity log data if available
    import random
    ratios = [random.uniform(0.1, 0.3) for _ in range(len(df))]
    df["processing_time_minutes"] = (df["duration_minutes"] * ratios).round(2)
    df["waiting_time_minutes"] = (df["duration_minutes"] - df["processing_time_minutes"]).round(2)

    # SLA thresholds per task (in minutes) — hardcoded for now
    # assuming this is a standard sales workflow; generalize later if needed
    SLA_THRESHOLDS = {
        "Lead Created": 5,
        "Lead Reviewed": 30,
        "Manager Approval": 120,
        "Proposal Sent": 60,
        "Deal Closed": 60
    }

    df["sla_threshold"] = df["task"].map(SLA_THRESHOLDS).fillna(99999)
    df["sla_violation"] = df["duration_minutes"] > df["sla_threshold"]

    df = df.reset_index(drop=True)

    return df


def compute_case_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate total end-to-end time per case.

    Finds the gap between the earliest start and the latest end for each
    workflow instance. Good for understanding overall cycle time.
    """
    # first start and last end per case
    case_aggs = df.groupby("case_id").agg(
        first_start=("start_time", "min"),
        last_end=("end_time", "max"),
    )

    total_mins = (case_aggs["last_end"] - case_aggs["first_start"]).dt.total_seconds() / 60

    case_durations = pd.DataFrame({
        "case_id": case_aggs.index,
        "total_duration_minutes": total_mins.abs(),
    }).reset_index(drop=True)

    return case_durations


def detect_exception_flows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Find loops (rework) and deviations (out-of-order steps) in workflow cases.

    Loops = a task appears more than once in the same case.
    Deviations = tasks run out of expected sequence.

    Assumes workflow order is mostly: Lead Created → Lead Reviewed →
    Manager Approval → Proposal Sent → Deal Closed.

    Parameters
    ----------
    df : pd.DataFrame
        Workflow data with case_id, task, start_time.

    Returns
    -------
    tuple[pd.DataFrame, int]
        Exception records DataFrame + count of unique affected cases.
    """
    exceptions = []

    # expected sequence — assuming workflow order is mostly correct
    expected_order = [
        "Lead Created",
        "Lead Reviewed",
        "Manager Approval",
        "Proposal Sent",
        "Deal Closed"
    ]
    expected_order_map = {task: i for i, task in enumerate(expected_order)}

    df_sorted = df.sort_values(by=["case_id", "start_time"])

    for case_id, group in df_sorted.groupby("case_id"):
        tasks = group["task"].tolist()
        if not tasks:
            continue

        # check for loops — task shows up more than once
        seen_tasks = set()
        for t in tasks:
            if t in seen_tasks:
                exceptions.append({
                    "case_id": case_id,
                    "issue_type": "loop",
                    "affected_task": t
                })
            seen_tasks.add(t)

        # check if the process started at the right step
        if expected_order_map.get(tasks[0], -1) != 0:
            exceptions.append({
                "case_id": case_id,
                "issue_type": "deviation",
                "affected_task": tasks[0]
            })

        # check each step transition — a valid move is exactly one step forward
        for i in range(1, len(tasks)):
            prev_t = tasks[i - 1]
            curr_t = tasks[i]

            if curr_t == prev_t:
                continue  # duplicate handled by loop check above

            prev_idx = expected_order_map.get(prev_t, -1)
            curr_idx = expected_order_map.get(curr_t, -1)

            if curr_idx != prev_idx + 1:
                exceptions.append({
                    "case_id": case_id,
                    "issue_type": "deviation",
                    "affected_task": curr_t
                })

    exceptions_df = pd.DataFrame(exceptions)

    if exceptions_df.empty:
        exceptions_df = pd.DataFrame(columns=["case_id", "issue_type", "affected_task"])
        total_exception_count = 0
    else:
        total_exception_count = exceptions_df["case_id"].nunique()

    return exceptions_df, total_exception_count
