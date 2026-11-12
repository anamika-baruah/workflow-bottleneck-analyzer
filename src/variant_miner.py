"""
Variant mining

Discovers process variants from workflow event logs, classifies them against
a standard (conformant) path, and scores each variant on cycle-time,
SLA-breach rate, wait-ratio, and drift from the dominant path.
"""

import string

import pandas as pd


def build_case_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a fingerprint string for every case by concatenating its ordered
    task names with a ``|`` separator.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ``[case_id, task, start_time, end_time]``.

    Returns
    -------
    DataFrame
        Columns ``[case_id, fingerprint]``.
    """
    if df.empty:
        return pd.DataFrame(columns=["case_id", "fingerprint"])

    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.dropna(subset=["start_time"])

    if df.empty:
        return pd.DataFrame(columns=["case_id", "fingerprint"])

    df = df.sort_values(["case_id", "start_time"])

    fingerprints = (
        df.groupby("case_id")["task"]
        .apply(lambda tasks: "|".join(tasks.astype(str)))
        .reset_index()
        .rename(columns={"task": "fingerprint"})
    )

    return fingerprints[["case_id", "fingerprint"]]


def mine_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify distinct process variants sorted by frequency (descending) and
    label them Variant A, Variant B and so on.

    Parameters
    ----------
    df : DataFrame
        Raw event log with ``[case_id, task, start_time, end_time]``.

    Returns
    -------
    DataFrame
        Columns ``[variant_label, fingerprint, frequency, cases]``.
    """
    if df.empty:
        return pd.DataFrame(columns=["variant_label", "fingerprint", "frequency", "cases"])

    fp_df = build_case_fingerprints(df)

    if fp_df.empty:
        return pd.DataFrame(columns=["variant_label", "fingerprint", "frequency", "cases"])

    variant_agg = (
        fp_df.groupby("fingerprint")["case_id"]
        .agg(frequency="nunique", cases=list)
        .reset_index()
        .sort_values("frequency", ascending=False)
        .reset_index(drop=True)
    )

    # labels run Variant A to Z, then Variant AA, AB, ...
    labels = []
    for i in range(len(variant_agg)):
        if i < 26:
            labels.append(f"Variant {string.ascii_uppercase[i]}")
        else:
            first = string.ascii_uppercase[(i // 26) - 1]
            second = string.ascii_uppercase[i % 26]
            labels.append(f"Variant {first}{second}")

    variant_agg.insert(0, "variant_label", labels)

    return variant_agg[["variant_label", "fingerprint", "frequency", "cases"]]


def tag_variant_type(fingerprint: str, standard_path: list) -> str:
    """
    Classify a variant fingerprint against a known standard path.

    Priority order (first match wins):

    1. **Conformant**: task list matches the standard path exactly.
    2. **Rework loop**: any task appears more than once.
    3. **Skip**: the path is shorter (some standard tasks are missing).
    4. **Extended**: catch-all for everything else (extra / reordered steps).

    Parameters
    ----------
    fingerprint : str
        Pipe-separated task sequence, e.g. ``"A|B|C"``.
    standard_path : list[str]
        Ordered list of expected task names, e.g. ``["A", "B", "C"]``.

    Returns
    -------
    str
        One of ``"Conformant"``, ``"Rework loop"``, ``"Skip"``, ``"Extended"``.
    """
    tasks = fingerprint.split("|")

    # 1. Conformant
    if tasks == standard_path:
        return "Conformant"

    # 2. Rework loop: any task appears more than once
    if len(tasks) != len(set(tasks)):
        return "Rework loop"

    # 3. Skip: fewer unique tasks than the standard, all present in standard
    if len(tasks) < len(standard_path):
        return "Skip"

    # 4. Extended: everything else
    return "Extended"


def score_variants(
    df: pd.DataFrame,
    variant_df: pd.DataFrame,
    sla_limit_hours: float = 24,
) -> pd.DataFrame:
    """
    Enrich *variant_df* with performance scores and drift indicators.

    Added columns
    --------------
    avg_cycle_hours : float
        Mean case cycle time in hours (rounded to 2 dp).
    sla_breach_rate : float
        Fraction of cases exceeding *sla_limit_hours* (rounded to 4 dp).
    avg_wait_ratio : float
        Mean ratio of wait-time to total-time per case (rounded to 4 dp).
    step_delta : int
        Difference in path length vs. the most-frequent variant.
    drift_label : str
        Human-readable drift tag (``"On track"``, ``"+N extra steps"``,
        ``"-N missing steps"``).

    Parameters
    ----------
    df : DataFrame
        Raw event log with ``[case_id, task, start_time, end_time]``.
    variant_df : DataFrame
        Output of :func:`mine_variants`.
    sla_limit_hours : float, optional
        SLA threshold in hours (default 24).

    Returns
    -------
    DataFrame
        *variant_df* augmented with the five columns listed above.
    """
    if df.empty or variant_df.empty:
        for col in ["avg_cycle_hours", "sla_breach_rate", "avg_wait_ratio",
                     "step_delta", "drift_label"]:
            variant_df[col] = None
        return variant_df

    df = df.copy()
    variant_df = variant_df.copy()

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")

    case_times = df.groupby("case_id").agg(
        min_start=("start_time", "min"),
        max_end=("end_time", "max"),
    )
    case_times["cycle_hours"] = (
        (case_times["max_end"] - case_times["min_start"])
        .dt.total_seconds() / 3600
    )
    case_times["sla_breached"] = case_times["cycle_hours"] > sla_limit_hours

    df_sorted = df.sort_values(["case_id", "start_time"])

    df_sorted["prev_end"] = df_sorted.groupby("case_id")["end_time"].shift(1)
    df_sorted["wait_minutes"] = (
        (df_sorted["start_time"] - df_sorted["prev_end"])
        .dt.total_seconds()
        .div(60)
        .clip(lower=0)
    )
    df_sorted["duration_minutes"] = (
        (df_sorted["end_time"] - df_sorted["start_time"])
        .dt.total_seconds()
        .div(60)
        .clip(lower=0)
    )

    wait_agg = df_sorted.groupby("case_id").agg(
        total_wait=("wait_minutes", "sum"),
        total_processing=("duration_minutes", "sum"),
    )
    wait_agg["wait_ratio"] = (
        wait_agg["total_wait"]
        / (wait_agg["total_wait"] + wait_agg["total_processing"] + 1e-9)
    )

    case_metrics = case_times[["cycle_hours", "sla_breached"]].join(
        wait_agg[["wait_ratio"]]
    )

    exploded = variant_df[["variant_label", "cases"]].explode("cases")
    exploded = exploded.rename(columns={"cases": "case_id"})
    exploded = exploded.merge(case_metrics, on="case_id", how="left")

    variant_scores = exploded.groupby("variant_label").agg(
        avg_cycle_hours=("cycle_hours", "mean"),
        sla_breach_rate=("sla_breached", "mean"),
        avg_wait_ratio=("wait_ratio", "mean"),
    ).reset_index()

    variant_scores["avg_cycle_hours"] = variant_scores["avg_cycle_hours"].round(2)
    variant_scores["sla_breach_rate"] = variant_scores["sla_breach_rate"].round(4)
    variant_scores["avg_wait_ratio"] = variant_scores["avg_wait_ratio"].round(4)

    variant_df = variant_df.merge(variant_scores, on="variant_label", how="left")

    standard_fingerprint = variant_df.iloc[0]["fingerprint"]
    standard_length = len(standard_fingerprint.split("|"))

    variant_df["step_delta"] = variant_df["fingerprint"].apply(
        lambda fp: len(fp.split("|")) - standard_length
    ).astype(int)

    def _drift_label(delta: int) -> str:
        if delta == 0:
            return "On track"
        if delta > 0:
            return f"+{delta} extra steps"
        return f"{delta} missing steps"

    variant_df["drift_label"] = variant_df["step_delta"].apply(_drift_label)

    return variant_df


NO_DATA_MESSAGE = "Not enough variant data to summarise."
MIN_DIFFERENCE_HOURS = 0.5     # ignore cycle-time differences smaller than this
SLA_THRESHOLD_RATE = 0.3       # breach rate worth mentioning
WAIT_THRESHOLD_RATIO = 0.5     # waiting share that is a finding on its own
WAIT_MARGIN = 0.10             # extra waiting share vs the standard path


def generate_variant_recommendations(variant_df: pd.DataFrame) -> pd.DataFrame:
    """
    Write a short recommendation for each process variant.

    Requires columns: ``[variant_label, variant_type, avg_cycle_hours,
    sla_breach_rate, avg_wait_ratio, frequency]``.

    Each text also gives the hours saved if the variant matched the standard path.

    Parameters
    ----------
    variant_df : DataFrame
        Scored variant table (output of :func:`score_variants` after adding
        a ``variant_type`` column via :func:`tag_variant_type`).

    Returns
    -------
    DataFrame
        *variant_df* with an added ``recommendation`` column.
    """
    if variant_df.empty:
        variant_df["recommendation"] = None
        return variant_df

    variant_df = variant_df.copy()

    # Compare against the standard path. The fastest variant is often the one that
    # skips steps, so it is not a fair target for "recoverable" time.
    standard_rows = variant_df[variant_df["variant_type"] == "Conformant"]
    baseline_cycle = (
        _case_weighted_avg_hours(standard_rows)
        if len(standard_rows) else variant_df["avg_cycle_hours"].min()
    )
    if len(standard_rows) and standard_rows["frequency"].sum() > 0:
        baseline_wait = float(
            (standard_rows["avg_wait_ratio"] * standard_rows["frequency"]).sum()
            / standard_rows["frequency"].sum()
        )
    else:
        baseline_wait = float(variant_df["avg_wait_ratio"].mean())

    recommendations = []

    for _, row in variant_df.iterrows():
        vtype = row["variant_type"]

        if vtype == "Rework loop":
            base = "Cases repeat steps, which adds time."
            fix = "Add a check before the repeated step so the work is right the first time."
        elif vtype == "Skip":
            base = "Some standard steps are skipped on this path."
            fix = "Check whether the skipped steps are needed, and if so make them required in the workflow tool."
        elif vtype == "Extended":
            base = "This path includes steps outside the standard sequence."
            fix = "Check whether the extra steps are needed."
        elif vtype == "Conformant":
            base = "This is the most common path."
            fix = "Use it as the baseline and track how many cases leave it."
        else:
            base = "Variant type is unrecognised."
            fix = "Review this path manually."

        # Judged against the standard path, not against a fixed number of hours:
        # a 12 h path is only "slow" if the standard path is faster.
        context = ""
        difference = row["avg_cycle_hours"] - baseline_cycle
        if vtype != "Conformant":
            if difference >= MIN_DIFFERENCE_HOURS:
                context += (
                    f" Takes {difference:.1f}h longer than the standard path"
                    f" ({row['avg_cycle_hours']:.1f}h vs {baseline_cycle:.1f}h)."
                )
            elif difference <= -MIN_DIFFERENCE_HOURS:
                context += (
                    f" Finishes {-difference:.1f}h faster than the standard path, but that"
                    f" speed comes from departing from the standard steps."
                )
        if row["sla_breach_rate"] > SLA_THRESHOLD_RATE:
            context += (
                f" SLA breach rate of {row['sla_breach_rate'] * 100:.0f}%"
                f" is high."
            )
        if vtype == "Conformant":
            if row["avg_wait_ratio"] > WAIT_THRESHOLD_RATIO:
                context += (
                    f" {row['avg_wait_ratio'] * 100:.0f}% of elapsed time is spent"
                    " waiting between steps, which points to a handoff or queue problem."
                )
        elif row["avg_wait_ratio"] > baseline_wait + WAIT_MARGIN:
            context += (
                f" Waiting makes up {row['avg_wait_ratio'] * 100:.0f}% of its elapsed time,"
                f" against {baseline_wait * 100:.0f}% on the standard path."
            )

        extra_time_per_case = max(0, row["avg_cycle_hours"] - baseline_cycle)
        total_time_waste = round(row["frequency"] * extra_time_per_case, 0)

        if total_time_waste > 0:
            recovery_text = (
                f" Matching the standard path would save about"
                f" {total_time_waste:.0f} hours across {row['frequency']} cases."
            )
        else:
            recovery_text = ""

        recommendation = base + " " + fix + context + recovery_text
        recommendations.append(recommendation)

    variant_df["recommendation"] = recommendations

    return variant_df


def _case_weighted_avg_hours(frame: pd.DataFrame) -> float:
    """Average cycle hours across variants, weighted by how many cases each has."""
    weights = frame["frequency"]
    if weights.sum() == 0:
        return float("nan")
    return float((frame["avg_cycle_hours"] * weights).sum() / weights.sum())


def generate_variant_insights(variant_df: pd.DataFrame) -> list:
    """
    Summarise the scored variants in a few plain-English sentences.

    Parameters
    ----------
    variant_df : DataFrame
        Must contain columns ``[variant_label, variant_type, frequency,
        avg_cycle_hours, sla_breach_rate]``.

    Returns
    -------
    list[str]
        One or more insight strings.  Never returns an empty list.
    """
    if variant_df.empty:
        return [NO_DATA_MESSAGE]

    total_cases = variant_df["frequency"].sum()
    if total_cases == 0:
        return [NO_DATA_MESSAGE]

    insights: list[str] = []

    conformant_rows = variant_df[variant_df["variant_type"] == "Conformant"]
    conformant_cases = conformant_rows["frequency"].sum()

    if conformant_cases > 0:
        conformance_pct = conformant_cases / total_cases * 100
        deviation_pct = 100 - conformance_pct

        conformant_avg = _case_weighted_avg_hours(conformant_rows)

        non_conformant_rows = variant_df[variant_df["variant_type"] != "Conformant"]
        if len(non_conformant_rows) > 0:
            deviant_avg = _case_weighted_avg_hours(non_conformant_rows)
        else:
            deviant_avg = conformant_avg

        extra_hours = max(0, deviant_avg - conformant_avg)

        insight = (
            f"{conformance_pct:.0f}% of cases follow the standard workflow. "
            f"The remaining {deviation_pct:.0f}% deviate, averaging "
            f"{extra_hours:.1f} extra hours per case."
        )
        insights.append(insight)

    rework_rows = variant_df[variant_df["variant_type"] == "Rework loop"]

    if len(rework_rows) > 0:
        rework_cases = rework_rows["frequency"].sum()
        rework_avg_hours = _case_weighted_avg_hours(rework_rows)

        if len(conformant_rows) > 0:
            conformant_avg_hours = _case_weighted_avg_hours(conformant_rows)
        else:
            conformant_avg_hours = _case_weighted_avg_hours(variant_df)

        extra_per_case = max(0, rework_avg_hours - conformant_avg_hours)
        total_waste_hours = rework_cases * extra_per_case

        insight = (
            f"Rework loops affect {rework_cases} cases and add "
            f"{extra_per_case:.1f} extra hours per case vs the standard path, "
            f"{total_waste_hours:.0f} hours in total."
        )
        insights.append(insight)

    # weight by cases, otherwise a one-off path counts as much as a path with hundreds of cases
    breached = (variant_df["sla_breach_rate"] * variant_df["frequency"]).groupby(variant_df["variant_type"]).sum()
    type_sla = breached / variant_df.groupby("variant_type")["frequency"].sum()
    worst_type = type_sla.idxmax()
    worst_rate = type_sla.max()

    conformant_rate = type_sla.get("Conformant", 0)

    if worst_type != "Conformant" and worst_rate > 0:
        if conformant_rate > 0:
            multiplier = worst_rate / conformant_rate
            insight = (
                f"{worst_type} variants have a {worst_rate * 100:.0f}% SLA breach rate, "
                f"{multiplier:.1f}x higher than conformant cases."
            )
        else:
            insight = (
                f"{worst_type} variants have a {worst_rate * 100:.0f}% SLA breach rate. "
                f"Conformant cases breach no SLAs."
            )
        insights.append(insight)

    if not insights:
        return [NO_DATA_MESSAGE]

    return insights