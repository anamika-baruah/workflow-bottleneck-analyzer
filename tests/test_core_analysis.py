"""Tests for preprocessing, bottleneck detection, exceptions, health and automation."""

import pandas as pd
import pytest

from src.automation_engine import identify_automation_opportunities
from src.bottleneck_detector import detect_bottlenecks, summarize_steps
from src.health_analyzer import compute_health_score
from src.preprocessing import (
    compute_case_durations,
    default_sla_minutes,
    detect_exception_flows,
    infer_step_order,
    preprocess_workflow,
)


def make_log(rows):
    """rows: (case, task, start, end) with times as 'YYYY-MM-DD HH:MM'."""
    df = pd.DataFrame(rows, columns=["case_id", "task", "start_time", "end_time"])
    df["user"] = "u"
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    return df


@pytest.fixture
def simple_log():
    # Two cases through Intake -> Review -> Approve. Review is the slow step.
    return make_log([
        ("c1", "Intake", "2026-01-05 09:00", "2026-01-05 09:10"),
        ("c1", "Review", "2026-01-05 09:40", "2026-01-05 11:40"),   # waited 30 min, took 120
        ("c1", "Approve", "2026-01-05 11:50", "2026-01-05 12:00"),  # waited 10 min, took 10
        ("c2", "Intake", "2026-01-06 09:00", "2026-01-06 09:10"),
        ("c2", "Review", "2026-01-06 09:10", "2026-01-06 10:10"),   # no wait, took 60
        ("c2", "Approve", "2026-01-06 10:30", "2026-01-06 10:40"),  # waited 20 min, took 10
    ])


class TestPreprocessing:
    def test_duration_and_wait_come_from_timestamps(self, simple_log):
        df = preprocess_workflow(simple_log)
        c1 = df[df["case_id"] == "c1"].set_index("task")
        assert c1.loc["Review", "duration_minutes"] == 120
        assert c1.loc["Review", "waiting_time_minutes"] == 30
        assert c1.loc["Approve", "waiting_time_minutes"] == 10
        assert c1.loc["Intake", "waiting_time_minutes"] == 0

    def test_output_is_deterministic(self, simple_log):
        first = preprocess_workflow(simple_log)
        second = preprocess_workflow(simple_log)
        pd.testing.assert_frame_equal(first, second)

    def test_reversed_rows_are_dropped_and_counted(self, simple_log):
        bad = simple_log.copy()
        start, end = bad.loc[0, "start_time"], bad.loc[0, "end_time"]
        bad.loc[0, "start_time"], bad.loc[0, "end_time"] = end, start
        df = preprocess_workflow(bad)
        assert len(df) == len(simple_log) - 1
        assert df.attrs["quality"]["end_before_start"] == 1
        assert (df["duration_minutes"] >= 0).all()

    def test_unparseable_timestamps_are_counted(self, simple_log):
        raw = simple_log.copy()
        raw["start_time"] = raw["start_time"].astype(str)
        raw.loc[2, "start_time"] = "not a date"
        df = preprocess_workflow(raw)
        assert df.attrs["quality"]["unparseable_timestamps"] == 1
        assert len(df) == len(simple_log) - 1

    def test_default_sla_is_twice_median_rounded_up_to_five(self, simple_log):
        sla = default_sla_minutes(simple_log)
        assert sla["Review"] == 180      # median 90 min -> 180
        assert sla["Intake"] == 20       # median 10 min -> 20

    def test_custom_sla_overrides_defaults(self, simple_log):
        df = preprocess_workflow(simple_log, {"Review": 100})
        review = df[df["task"] == "Review"].set_index("case_id")
        assert bool(review.loc["c1", "sla_violation"]) is True    # 120 > 100
        assert bool(review.loc["c2", "sla_violation"]) is False   # 60 <= 100

    def test_works_for_any_task_names(self):
        df = make_log([
            ("t1", "Opened", "2026-02-01 08:00", "2026-02-01 08:05"),
            ("t1", "Escalated", "2026-02-01 09:00", "2026-02-01 12:00"),
            ("t1", "Closed", "2026-02-01 12:10", "2026-02-01 12:15"),
        ])
        out = preprocess_workflow(df)
        assert out["sla_threshold"].notna().all()
        assert infer_step_order(out) == ["Opened", "Escalated", "Closed"]

    def test_case_durations(self, simple_log):
        durations = compute_case_durations(preprocess_workflow(simple_log)).set_index("case_id")
        assert durations.loc["c1", "total_duration_minutes"] == 180
        assert durations.loc["c2", "total_duration_minutes"] == 100


class TestBottleneck:
    def test_slowest_average_step_is_flagged(self, simple_log):
        df = preprocess_workflow(simple_log)
        stats, task, total, share = detect_bottlenecks(df)
        assert task == "Review"
        assert total == 180
        assert share == pytest.approx(180 / 220 * 100, abs=0.01)

    def test_summary_is_in_process_order(self, simple_log):
        df = preprocess_workflow(simple_log)
        summary = summarize_steps(df, infer_step_order(df))
        assert list(summary.index) == ["Intake", "Review", "Approve"]
        assert summary.loc["Review", "avg_wait_before"] == 15   # (30 + 0) / 2

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError):
            detect_bottlenecks(pd.DataFrame({"task": ["a"]}))


class TestExceptions:
    def _base(self, case, tasks):
        rows, hour = [], 8
        for task in tasks:
            rows.append((case, task, f"2026-03-01 {hour:02d}:00", f"2026-03-01 {hour:02d}:30"))
            hour += 1
        return rows

    def test_clean_process_has_no_exceptions(self):
        rows = self._base("a", ["S1", "S2", "S3"]) + self._base("b", ["S1", "S2", "S3"])
        _, count = detect_exception_flows(preprocess_workflow(make_log(rows)))
        assert count == 0

    def test_rework_loop_and_skip_are_found(self):
        rows = (
            self._base("a", ["S1", "S2", "S3", "S4"])
            + self._base("b", ["S1", "S2", "S3", "S2", "S3", "S4"])   # loop
            + self._base("c", ["S1", "S3", "S4"])                      # skip
            + self._base("d", ["S1", "S2", "S3", "S4"])
        )
        exceptions, count = detect_exception_flows(preprocess_workflow(make_log(rows)),
                                                   ["S1", "S2", "S3", "S4"])
        assert count == 2
        issues = exceptions.groupby("case_id")["issue_type"].apply(set).to_dict()
        assert "loop" in issues["b"]
        assert issues["c"] == {"deviation"}


class TestHealth:
    def test_score_is_bounded_and_explains_itself(self, simple_log):
        df = preprocess_workflow(simple_log)
        _, _, _, share = detect_bottlenecks(df)
        health = compute_health_score(df, share, total_exception_cases=1)
        assert 0 <= health["score"] <= 100
        assert set(health["metrics"]) == {
            "SLA breaches", "Bottleneck concentration", "Rework and deviations", "Handoff waiting"}

    def test_single_step_process_has_no_concentration_penalty(self):
        df = preprocess_workflow(make_log([("a", "Only", "2026-01-01 08:00", "2026-01-01 09:00")]))
        health = compute_health_score(df, bottleneck_pct=100.0, total_exception_cases=0)
        assert health["metrics"]["Bottleneck concentration"] == 0

    def test_empty_input(self):
        assert compute_health_score(pd.DataFrame(), 0, 0)["score"] == 0


class TestAutomation:
    def test_estimates_are_deterministic(self, simple_log):
        df = preprocess_workflow(simple_log)
        stats, *_ = detect_bottlenecks(df)
        first = identify_automation_opportunities(df, stats)
        second = identify_automation_opportunities(df, stats)
        assert first == second

    def test_savings_apply_the_stated_assumption(self, simple_log):
        df = preprocess_workflow(simple_log)
        stats, *_ = detect_bottlenecks(df)
        for item in identify_automation_opportunities(df, stats, assumed_reduction=0.5):
            avg = stats.loc[item["task"], "avg_duration_minutes"]
            assert item["minutes_saved_per_case"] == pytest.approx(avg * 0.5, abs=0.1)

    def test_empty_input(self):
        assert identify_automation_opportunities(pd.DataFrame(), pd.DataFrame()) == []


class TestVariantInsightsWeighting:
    def test_insights_weight_variants_by_case_count(self):
        from src.variant_miner import generate_variant_insights

        variants = pd.DataFrame({
            "variant_label": ["A", "B", "C"],
            "variant_type": ["Conformant", "Rework loop", "Rework loop"],
            "frequency": [100, 99, 1],
            "avg_cycle_hours": [10.0, 20.0, 110.0],
            "sla_breach_rate": [0.1, 0.5, 1.0],
        })
        text = " ".join(generate_variant_insights(variants))
        # weighted rework average = (99*20 + 1*110) / 100 = 20.9 h -> 10.9 h extra per case
        assert "add 10.9 extra hours per case" in text


# ── handoff delay and priority actions ───────────────────────────────────────
from src.bottleneck_detector import (  # noqa: E402
    find_handoff_delay,
    generate_priority_actions,
    handoff_action,
)


def make_steps(rows):
    """rows: (step, avg_wait_before, avg_duration, median_duration, p90_duration)."""
    return pd.DataFrame(
        rows,
        columns=["task", "avg_wait_before", "avg_duration", "median_duration", "p90_duration"],
    ).set_index("task").assign(sla_target=120.0)


class TestHandoffDelay:
    def test_flags_a_wait_far_above_the_others(self):
        steps = make_steps([
            ("Opened", 0, 5, 5, 8), ("Assigned", 300, 15, 15, 20),
            ("Work", 30, 400, 160, 1400), ("Closed", 20, 10, 10, 15),
        ])
        found = find_handoff_delay(steps)
        assert found["task"] == "Assigned"
        assert found["avg_wait"] == 300
        assert found["work_ratio"] == pytest.approx(20)

    def test_similar_waits_are_not_a_finding(self):
        # 137 vs 105 minutes: the largest wait is not meaningfully larger than the rest
        steps = make_steps([
            ("A", 0, 78, 80, 112), ("B", 137, 150, 155, 220), ("C", 105, 973, 320, 2712),
            ("D", 75, 107, 107, 166), ("E", 106, 75, 76, 111),
        ])
        assert find_handoff_delay(steps) is None

    def test_short_waits_are_not_a_finding(self):
        steps = make_steps([("A", 0, 5, 5, 8), ("B", 40, 5, 5, 8), ("C", 5, 5, 5, 8), ("D", 4, 5, 5, 8)])
        assert find_handoff_delay(steps) is None

    def test_needs_at_least_three_steps(self):
        assert find_handoff_delay(make_steps([("A", 0, 5, 5, 8), ("B", 500, 5, 5, 8)])) is None


class TestPriorityActions:
    def _stats(self, names):
        return pd.DataFrame(
            {"avg_duration_minutes": [400, 300, 200][: len(names)], "count": [100] * len(names)},
            index=names,
        )

    def test_queue_dominated_step_gets_a_queue_action(self):
        steps = make_steps([("Agent Assigned", 317, 15, 15, 22)])
        actions = generate_priority_actions(self._stats(["Agent Assigned"]), steps=steps)
        assert "before it starts" in actions[0]
        assert "automatically" in actions[0]          # assignment-specific advice

    def test_slow_tail_gets_a_tail_action(self):
        steps = make_steps([("Investigation", 30, 421, 166, 1394)])
        actions = generate_priority_actions(self._stats(["Investigation"]), steps=steps)
        assert "slow tail" in actions[0]
        assert "90th percentile" in actions[0]

    def test_falls_back_to_keyword_advice_without_step_data(self):
        actions = generate_priority_actions(self._stats(["Manager Approval", "Lead Review", "Packing"]))
        assert actions[0] == "Manager Approval: auto-approve low-risk cases"
        assert actions[1] == "Lead Review: review in parallel"
        assert "standardise" in actions[2]

    def test_actions_differ_when_the_problems_differ(self):
        steps = make_steps([
            ("Agent Assigned", 317, 15, 15, 22),
            ("Investigation", 30, 421, 166, 1394),
            ("Customer Confirmation", 197, 68, 71, 109),
        ])
        stats = self._stats(["Investigation", "Customer Confirmation", "Agent Assigned"])
        actions = generate_priority_actions(stats, steps=steps)
        assert len(set(actions)) == 3

    def test_handoff_action_depends_on_the_step_name(self):
        assert "automatically" in handoff_action("Agent Assigned", 30)
        assert "reminders" in handoff_action("Customer Confirmation", 145)
        assert "ownership" in handoff_action("Packing", 60)


from src.bottleneck_detector import has_slow_tail, slow_tail_action  # noqa: E402


class TestSlowTail:
    def test_detects_tail_only_when_p90_is_far_above_median(self):
        steps = make_steps([("A", 0, 400, 160, 1400), ("B", 0, 100, 100, 150)])
        assert has_slow_tail(steps.loc["A"]) is True
        assert has_slow_tail(steps.loc["B"]) is False

    def test_approval_tail_keeps_the_domain_advice(self):
        steps = make_steps([("Manager Approval", 105, 973, 320, 2712)])
        text = slow_tail_action("Manager Approval", steps.loc["Manager Approval"])
        assert "slow tail" in text and "Auto-approve low-risk cases" in text

    def test_generic_step_gets_no_keyword_suffix(self):
        steps = make_steps([("Investigation", 30, 421, 166, 1394)])
        text = slow_tail_action("Investigation", steps.loc["Investigation"])
        assert "Auto-approve" not in text and "Review in parallel" not in text


class TestNoStandoutProblem:
    def test_step_without_a_problem_gets_an_honest_note_not_boilerplate(self):
        steps = make_steps([("Lead Reviewed", 137, 150, 155, 220)])
        stats = pd.DataFrame({"avg_duration_minutes": [150], "count": [306]}, index=["Lead Reviewed"])
        text = generate_priority_actions(stats, steps=steps)[0]
        assert "neither stands out" in text
        assert "review in parallel" not in text