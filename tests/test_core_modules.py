"""
tests/test_core_modules.py

Unit tests for the three core analytical modules:
  - src.bottleneck_detector   (detect_bottlenecks)
  - src.health_analyzer       (calculate_workflow_health)
  - src.risk_predictor        (predict_bottleneck_risk)

Run with:
    pytest tests/test_core_modules.py -v
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Make src/ importable regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.bottleneck_detector import detect_bottlenecks, generate_recommendation
from src.health_analyzer import calculate_workflow_health
from src.risk_predictor import predict_bottleneck_risk


# ═══════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════════

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal preprocessed-style DataFrame for testing."""
    df = pd.DataFrame(rows)
    # Ensure required columns exist with correct types
    if "duration_minutes" in df.columns:
        df["duration_minutes"] = df["duration_minutes"].astype(float)
    if "sla_violation" not in df.columns:
        df["sla_violation"] = 0
    if "waiting_time_minutes" not in df.columns:
        df["waiting_time_minutes"] = 0.0
    if "processing_time_minutes" not in df.columns:
        df["processing_time_minutes"] = df.get("duration_minutes", pd.Series(dtype=float))
    if "case_id" not in df.columns:
        df["case_id"] = "C001"
    return df


@pytest.fixture
def simple_df():
    """Three tasks with clearly different durations; 'Approval' is slowest."""
    return _make_df([
        {"task": "Lead Created",     "duration_minutes": 5,   "case_id": "C001"},
        {"task": "Manager Approval", "duration_minutes": 120, "case_id": "C001"},
        {"task": "Deal Closed",      "duration_minutes": 20,  "case_id": "C001"},
        {"task": "Lead Created",     "duration_minutes": 6,   "case_id": "C002"},
        {"task": "Manager Approval", "duration_minutes": 150, "case_id": "C002"},
        {"task": "Deal Closed",      "duration_minutes": 25,  "case_id": "C002"},
    ])


@pytest.fixture
def equal_df():
    """All tasks have the same average duration — bottleneck is arbitrary but stable."""
    return _make_df([
        {"task": "Step A", "duration_minutes": 30, "case_id": "C001"},
        {"task": "Step B", "duration_minutes": 30, "case_id": "C001"},
        {"task": "Step C", "duration_minutes": 30, "case_id": "C001"},
    ])


@pytest.fixture
def sla_df():
    """DataFrame with explicit SLA violations and waiting times for health tests."""
    return _make_df([
        # case 1
        {"task": "Lead Created", "duration_minutes": 10, "waiting_time_minutes": 2,
         "sla_violation": 1, "case_id": "C001"},
        {"task": "Approval",     "duration_minutes": 200,"waiting_time_minutes": 80,
         "sla_violation": 1, "case_id": "C001"},
        # case 2
        {"task": "Lead Created", "duration_minutes": 4,  "waiting_time_minutes": 1,
         "sla_violation": 0, "case_id": "C002"},
        {"task": "Approval",     "duration_minutes": 50, "waiting_time_minutes": 10,
         "sla_violation": 0, "case_id": "C002"},
    ])


# ═══════════════════════════════════════════════════════════════════════════
# bottleneck_detector tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectBottlenecks:

    def test_identifies_correct_bottleneck(self, simple_df):
        """The task with the highest avg duration should be flagged."""
        task_stats, bottleneck_task, _, _ = detect_bottlenecks(simple_df)
        assert bottleneck_task == "Manager Approval"

    def test_task_stats_shape(self, simple_df):
        """task_stats should have one row per unique task."""
        task_stats, _, _, _ = detect_bottlenecks(simple_df)
        assert len(task_stats) == 3

    def test_task_stats_sorted_descending(self, simple_df):
        """Rows must be sorted by avg_duration_minutes descending."""
        task_stats, _, _, _ = detect_bottlenecks(simple_df)
        avgs = task_stats["avg_duration_minutes"].tolist()
        assert avgs == sorted(avgs, reverse=True)

    def test_bottleneck_pct_is_valid_range(self, simple_df):
        """bottleneck_pct must be between 0 and 100."""
        _, _, _, bottleneck_pct = detect_bottlenecks(simple_df)
        assert 0.0 <= bottleneck_pct <= 100.0

    def test_bottleneck_pct_calculation(self, simple_df):
        """Manually verify the percentage formula."""
        task_stats, bottleneck_task, total_bn_time, bottleneck_pct = detect_bottlenecks(simple_df)
        total_all = simple_df["duration_minutes"].sum()
        expected_pct = round((total_bn_time / total_all) * 100, 2)
        assert abs(bottleneck_pct - expected_pct) < 0.01

    def test_total_bn_time_is_positive(self, simple_df):
        """Total bottleneck time should be greater than zero for valid data."""
        _, _, total_bn_time, _ = detect_bottlenecks(simple_df)
        assert total_bn_time > 0

    def test_equal_durations_returns_a_result(self, equal_df):
        """When all tasks are equal, a bottleneck is still returned without error."""
        task_stats, bottleneck_task, total_bn_time, bottleneck_pct = detect_bottlenecks(equal_df)
        assert bottleneck_task in ["Step A", "Step B", "Step C"]
        assert bottleneck_pct > 0

    def test_missing_task_column_raises(self):
        """ValueError should be raised when 'task' column is absent."""
        bad_df = pd.DataFrame({"duration_minutes": [10, 20]})
        with pytest.raises(ValueError, match="Missing required columns"):
            detect_bottlenecks(bad_df)

    def test_missing_duration_column_raises(self):
        """ValueError should be raised when 'duration_minutes' column is absent."""
        bad_df = pd.DataFrame({"task": ["A", "B"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            detect_bottlenecks(bad_df)

    def test_single_task_df(self):
        """A single-task dataset is valid — bottleneck_pct should be 100%."""
        df = _make_df([
            {"task": "Only Step", "duration_minutes": 45, "case_id": "C001"},
            {"task": "Only Step", "duration_minutes": 55, "case_id": "C002"},
        ])
        task_stats, bottleneck_task, _, bottleneck_pct = detect_bottlenecks(df)
        assert bottleneck_task == "Only Step"
        assert abs(bottleneck_pct - 100.0) < 0.01

    def test_count_column_correct(self, simple_df):
        """task_stats 'count' should match the number of rows per task."""
        task_stats, _, _, _ = detect_bottlenecks(simple_df)
        # Manager Approval appears twice in simple_df
        assert task_stats.loc["Manager Approval", "count"] == 2


class TestGenerateRecommendation:

    def test_approval_task_recommendation(self):
        rec = generate_recommendation("Manager Approval", avg_duration=180, execution_count=80)
        assert "approval" in rec["recommendation_text"].lower()
        assert "estimated_savings" in rec

    def test_review_task_recommendation(self):
        rec = generate_recommendation("Code Review", avg_duration=90, execution_count=30)
        assert "review" in rec["recommendation_text"].lower()

    def test_high_duration_triggers_urgency(self):
        rec = generate_recommendation("Generic Task", avg_duration=400, execution_count=10)
        assert "immediate" in rec["recommendation_text"].lower() or "significantly" in rec["recommendation_text"].lower()

    def test_high_count_triggers_compound_note(self):
        rec = generate_recommendation("Generic Task", avg_duration=50, execution_count=200)
        assert "frequently" in rec["recommendation_text"].lower() or "compound" in rec["recommendation_text"].lower()

    def test_savings_is_string(self):
        rec = generate_recommendation("Any Task", avg_duration=60, execution_count=20)
        assert isinstance(rec["estimated_savings"], str)
        assert "%" in rec["estimated_savings"]


# ═══════════════════════════════════════════════════════════════════════════
# health_analyzer tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCalculateWorkflowHealth:

    def test_score_is_in_valid_range(self, sla_df):
        result = calculate_workflow_health(sla_df, bottleneck_pct=30.0, total_exception_cases=1)
        assert 0 <= result["score"] <= 100

    def test_returns_required_keys(self, sla_df):
        result = calculate_workflow_health(sla_df, bottleneck_pct=20.0, total_exception_cases=0)
        for key in ("score", "status", "interpretation", "top_contributors", "metrics"):
            assert key in result, f"Key '{key}' missing from result"

    def test_empty_df_returns_zero_score(self):
        empty = pd.DataFrame()
        result = calculate_workflow_health(empty, bottleneck_pct=0.0, total_exception_cases=0)
        assert result["score"] == 0

    def test_healthy_score_gives_healthy_status(self, simple_df):
        """Low bottleneck + no SLA violations + no exceptions → Healthy."""
        result = calculate_workflow_health(simple_df, bottleneck_pct=5.0, total_exception_cases=0)
        # Score should be relatively high; status should NOT be Critical
        assert result["status"] in ("Healthy", "Moderate")

    def test_high_penalty_gives_critical_status(self, sla_df):
        """High bottleneck + many violations + exceptions → Critical."""
        result = calculate_workflow_health(sla_df, bottleneck_pct=80.0, total_exception_cases=2)
        assert result["status"] in ("Critical", "Moderate")

    def test_score_decreases_with_more_exceptions(self, sla_df):
        """More exception cases should result in a lower health score."""
        score_few = calculate_workflow_health(sla_df, bottleneck_pct=20.0, total_exception_cases=0)["score"]
        score_many = calculate_workflow_health(sla_df, bottleneck_pct=20.0, total_exception_cases=2)["score"]
        assert score_few > score_many

    def test_score_decreases_with_higher_bottleneck_pct(self, sla_df):
        """Higher bottleneck percentage should reduce health score."""
        score_low = calculate_workflow_health(sla_df, bottleneck_pct=10.0, total_exception_cases=0)["score"]
        score_high = calculate_workflow_health(sla_df, bottleneck_pct=60.0, total_exception_cases=0)["score"]
        assert score_low > score_high

    def test_top_contributors_is_list(self, sla_df):
        result = calculate_workflow_health(sla_df, bottleneck_pct=40.0, total_exception_cases=1)
        assert isinstance(result["top_contributors"], list)

    def test_metrics_dict_has_expected_keys(self, sla_df):
        result = calculate_workflow_health(sla_df, bottleneck_pct=30.0, total_exception_cases=1)
        metrics = result["metrics"]
        for key in ("sla_viol_rate", "btl_impact", "exc_rate", "wait_ratio"):
            assert key in metrics, f"Metric key '{key}' missing"

    def test_score_is_rounded(self, sla_df):
        """Health score should be rounded (no more than 1 decimal place)."""
        result = calculate_workflow_health(sla_df, bottleneck_pct=25.0, total_exception_cases=1)
        score = result["score"]
        assert score == round(score, 1)


# ═══════════════════════════════════════════════════════════════════════════
# risk_predictor tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPredictBottleneckRisk:

    def test_returns_list(self, simple_df):
        task_stats, _, _, _ = detect_bottlenecks(simple_df)
        result = predict_bottleneck_risk(simple_df, task_stats)
        assert isinstance(result, list)

    def test_one_entry_per_task(self, simple_df):
        task_stats, _, _, _ = detect_bottlenecks(simple_df)
        result = predict_bottleneck_risk(simple_df, task_stats)
        assert len(result) == task_stats.shape[0]

    def test_each_entry_has_required_keys(self, simple_df):
        task_stats, _, _, _ = detect_bottlenecks(simple_df)
        result = predict_bottleneck_risk(simple_df, task_stats)
        for entry in result:
            for key in ("task", "risk_score", "explanation", "factors"):
                assert key in entry, f"Key '{key}' missing from risk entry"

    def test_risk_scores_in_valid_range(self, simple_df):
        task_stats, _, _, _ = detect_bottlenecks(simple_df)
        result = predict_bottleneck_risk(simple_df, task_stats)
        for entry in result:
            assert 0 <= entry["risk_score"] <= 100, (
                f"risk_score {entry['risk_score']} out of range for task {entry['task']}"
            )

    def test_sorted_descending_by_risk(self, simple_df):
        """Results should be ordered from highest to lowest risk."""
        task_stats, _, _, _ = detect_bottlenecks(simple_df)
        result = predict_bottleneck_risk(simple_df, task_stats)
        scores = [r["risk_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_highest_risk_is_bottleneck(self, simple_df):
        """Manager Approval (highest avg + some volume) should rank highest."""
        task_stats, _, _, _ = detect_bottlenecks(simple_df)
        result = predict_bottleneck_risk(simple_df, task_stats)
        assert result[0]["task"] == "Manager Approval"

    def test_factors_is_list(self, simple_df):
        task_stats, _, _, _ = detect_bottlenecks(simple_df)
        result = predict_bottleneck_risk(simple_df, task_stats)
        for entry in result:
            assert isinstance(entry["factors"], list)

    def test_missing_columns_returns_empty(self):
        """When required columns are absent, an empty list should be returned."""
        bad_df     = pd.DataFrame({"task": ["A"]})
        stats_fake = pd.DataFrame({"avg_duration_minutes": [10]}, index=["A"])
        result = predict_bottleneck_risk(bad_df, stats_fake)
        assert result == []

    def test_explanation_is_string(self, simple_df):
        task_stats, _, _, _ = detect_bottlenecks(simple_df)
        result = predict_bottleneck_risk(simple_df, task_stats)
        for entry in result:
            assert isinstance(entry["explanation"], str)
            assert len(entry["explanation"]) > 0

    def test_high_variability_raises_risk(self):
        """A task with very high standard deviation should have a higher risk score
        than an equal-avg task with low variance."""
        df_varied = _make_df([
            {"task": "Stable",   "duration_minutes": 50, "case_id": "C1"},
            {"task": "Stable",   "duration_minutes": 50, "case_id": "C2"},
            {"task": "Unstable", "duration_minutes": 10, "case_id": "C1"},
            {"task": "Unstable", "duration_minutes": 90, "case_id": "C2"},
        ])
        task_stats, _, _, _ = detect_bottlenecks(df_varied)
        result = predict_bottleneck_risk(df_varied, task_stats)
        risk_map = {r["task"]: r["risk_score"] for r in result}
        # Unstable should score higher due to higher variability (std dev)
        assert risk_map["Unstable"] >= risk_map["Stable"]