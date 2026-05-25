"""
Unit tests for src/variant_miner.py
====================================
Covers all five public functions:
  - build_case_fingerprints
  - mine_variants
  - tag_variant_type
  - score_variants  (+ Upgrade 2 drift columns)
  - generate_variant_recommendations  (+ Upgrade 1 time recovery)
"""

import pytest
import pandas as pd
import numpy as np

from src.variant_miner import (
    build_case_fingerprints,
    mine_variants,
    tag_variant_type,
    score_variants,
    generate_variant_recommendations,
)


# ───────────────────────────── fixtures ──────────────────────────────────────

@pytest.fixture
def sample_event_log():
    """A small event log with 3 cases and 2 distinct variants."""
    return pd.DataFrame({
        "case_id": ["C1", "C1", "C1",
                     "C2", "C2", "C2",
                     "C3", "C3", "C3", "C3"],
        "task": ["A", "B", "C",
                 "A", "B", "C",
                 "A", "B", "A", "C"],
        "start_time": [
            "2026-01-01 08:00", "2026-01-01 09:00", "2026-01-01 10:00",
            "2026-01-01 08:00", "2026-01-01 12:00", "2026-01-02 09:00",
            "2026-01-01 08:00", "2026-01-01 09:30", "2026-01-01 11:00",
            "2026-01-01 13:00",
        ],
        "end_time": [
            "2026-01-01 08:30", "2026-01-01 09:30", "2026-01-01 10:30",
            "2026-01-01 09:00", "2026-01-01 13:00", "2026-01-02 10:00",
            "2026-01-01 09:00", "2026-01-01 10:30", "2026-01-01 12:00",
            "2026-01-01 14:00",
        ],
    })


@pytest.fixture
def empty_df():
    return pd.DataFrame()


@pytest.fixture
def standard_path():
    return ["A", "B", "C"]


# ──────────────────── build_case_fingerprints tests ─────────────────────────

class TestBuildCaseFingerprints:

    def test_returns_expected_columns(self, sample_event_log):
        result = build_case_fingerprints(sample_event_log)
        assert list(result.columns) == ["case_id", "fingerprint"]

    def test_correct_fingerprints(self, sample_event_log):
        result = build_case_fingerprints(sample_event_log)
        fp_map = result.set_index("case_id")["fingerprint"].to_dict()
        assert fp_map["C1"] == "A|B|C"
        assert fp_map["C2"] == "A|B|C"
        assert fp_map["C3"] == "A|B|A|C"

    def test_empty_input(self, empty_df):
        result = build_case_fingerprints(empty_df)
        assert result.empty
        assert list(result.columns) == ["case_id", "fingerprint"]

    def test_nat_start_times_dropped(self):
        df = pd.DataFrame({
            "case_id": ["C1", "C1"],
            "task": ["A", "B"],
            "start_time": ["2026-01-01 08:00", "not-a-date"],
            "end_time": ["2026-01-01 09:00", "2026-01-01 10:00"],
        })
        result = build_case_fingerprints(df)
        assert len(result) == 1
        assert result.iloc[0]["fingerprint"] == "A"

    def test_all_nat_returns_empty(self):
        df = pd.DataFrame({
            "case_id": ["C1"],
            "task": ["A"],
            "start_time": ["invalid"],
            "end_time": ["2026-01-01 09:00"],
        })
        result = build_case_fingerprints(df)
        assert result.empty

    def test_sorting_by_start_time(self):
        """Tasks should be ordered by start_time, not by insertion order."""
        df = pd.DataFrame({
            "case_id": ["C1", "C1", "C1"],
            "task": ["C", "A", "B"],
            "start_time": [
                "2026-01-01 12:00",
                "2026-01-01 08:00",
                "2026-01-01 10:00",
            ],
            "end_time": [
                "2026-01-01 13:00",
                "2026-01-01 09:00",
                "2026-01-01 11:00",
            ],
        })
        result = build_case_fingerprints(df)
        assert result.iloc[0]["fingerprint"] == "A|B|C"


# ──────────────────────── mine_variants tests ───────────────────────────────

class TestMineVariants:

    def test_returns_expected_columns(self, sample_event_log):
        result = mine_variants(sample_event_log)
        assert list(result.columns) == [
            "variant_label", "fingerprint", "frequency", "cases"
        ]

    def test_variant_count(self, sample_event_log):
        result = mine_variants(sample_event_log)
        assert len(result) == 2  # A|B|C and A|B|A|C

    def test_sorted_by_frequency_desc(self, sample_event_log):
        result = mine_variants(sample_event_log)
        freqs = result["frequency"].tolist()
        assert freqs == sorted(freqs, reverse=True)

    def test_labels_sequential(self, sample_event_log):
        result = mine_variants(sample_event_log)
        assert result.iloc[0]["variant_label"] == "Variant A"
        assert result.iloc[1]["variant_label"] == "Variant B"

    def test_most_frequent_variant_first(self, sample_event_log):
        result = mine_variants(sample_event_log)
        assert result.iloc[0]["frequency"] == 2
        assert result.iloc[0]["fingerprint"] == "A|B|C"

    def test_cases_list_content(self, sample_event_log):
        result = mine_variants(sample_event_log)
        top_cases = result.iloc[0]["cases"]
        assert set(top_cases) == {"C1", "C2"}

    def test_empty_input(self, empty_df):
        result = mine_variants(empty_df)
        assert result.empty
        assert list(result.columns) == [
            "variant_label", "fingerprint", "frequency", "cases"
        ]


# ──────────────────── tag_variant_type tests ────────────────────────────────

class TestTagVariantType:

    def test_conformant(self, standard_path):
        assert tag_variant_type("A|B|C", standard_path) == "Conformant"

    def test_rework_loop(self, standard_path):
        assert tag_variant_type("A|B|A|C", standard_path) == "Rework loop"

    def test_skip(self, standard_path):
        assert tag_variant_type("A|C", standard_path) == "Skip"

    def test_extended(self, standard_path):
        assert tag_variant_type("A|B|C|D", standard_path) == "Extended"

    def test_priority_rework_over_extended(self, standard_path):
        """A path with duplicates AND more tasks should be 'Rework loop', not 'Extended'."""
        assert tag_variant_type("A|B|C|A|D", standard_path) == "Rework loop"

    def test_reordered_same_length_is_extended(self, standard_path):
        """Same tasks in different order, no duplicates, same length → Extended."""
        assert tag_variant_type("C|B|A", standard_path) == "Extended"

    def test_single_task_skip(self, standard_path):
        assert tag_variant_type("B", standard_path) == "Skip"


# ──────────────────── score_variants tests ──────────────────────────────────

class TestScoreVariants:

    def test_returns_score_columns(self, sample_event_log):
        vdf = mine_variants(sample_event_log)
        scored = score_variants(sample_event_log, vdf, sla_limit_hours=4)
        for col in ["avg_cycle_hours", "sla_breach_rate", "avg_wait_ratio",
                     "step_delta", "drift_label"]:
            assert col in scored.columns

    def test_numeric_types(self, sample_event_log):
        vdf = mine_variants(sample_event_log)
        scored = score_variants(sample_event_log, vdf)
        assert scored["avg_cycle_hours"].dtype == np.float64
        assert scored["avg_wait_ratio"].dtype == np.float64

    def test_sla_breach_rate_range(self, sample_event_log):
        vdf = mine_variants(sample_event_log)
        scored = score_variants(sample_event_log, vdf)
        assert (scored["sla_breach_rate"] >= 0).all()
        assert (scored["sla_breach_rate"] <= 1).all()

    def test_wait_ratio_range(self, sample_event_log):
        vdf = mine_variants(sample_event_log)
        scored = score_variants(sample_event_log, vdf)
        assert (scored["avg_wait_ratio"] >= 0).all()
        assert (scored["avg_wait_ratio"] <= 1).all()

    def test_step_delta_integer(self, sample_event_log):
        vdf = mine_variants(sample_event_log)
        scored = score_variants(sample_event_log, vdf)
        assert scored["step_delta"].dtype in (np.int64, np.int32, int)

    def test_drift_label_on_track_for_baseline(self, sample_event_log):
        vdf = mine_variants(sample_event_log)
        scored = score_variants(sample_event_log, vdf)
        baseline_row = scored.iloc[0]
        assert baseline_row["drift_label"] == "On track"

    def test_drift_label_extra_steps(self, sample_event_log):
        vdf = mine_variants(sample_event_log)
        scored = score_variants(sample_event_log, vdf)
        rework_row = scored[scored["fingerprint"] == "A|B|A|C"].iloc[0]
        assert "extra steps" in rework_row["drift_label"]
        assert rework_row["step_delta"] > 0

    def test_empty_df_guard(self, empty_df):
        vdf = pd.DataFrame(columns=[
            "variant_label", "fingerprint", "frequency", "cases"
        ])
        result = score_variants(empty_df, vdf)
        for col in ["avg_cycle_hours", "sla_breach_rate", "avg_wait_ratio",
                     "step_delta", "drift_label"]:
            assert col in result.columns

    def test_rounding_precision(self, sample_event_log):
        vdf = mine_variants(sample_event_log)
        scored = score_variants(sample_event_log, vdf, sla_limit_hours=4)
        for _, row in scored.iterrows():
            # avg_cycle_hours rounded to 2 decimals
            assert round(row["avg_cycle_hours"], 2) == row["avg_cycle_hours"]
            # sla_breach_rate rounded to 4 decimals
            assert round(row["sla_breach_rate"], 4) == row["sla_breach_rate"]
            # avg_wait_ratio rounded to 4 decimals
            assert round(row["avg_wait_ratio"], 4) == row["avg_wait_ratio"]


# ────────────── generate_variant_recommendations tests ──────────────────────

class TestGenerateVariantRecommendations:

    def _make_variant_df(self, variant_type, avg_cycle=8.0, sla_rate=0.1,
                         wait_ratio=0.2, frequency=10):
        return pd.DataFrame({
            "variant_label": ["Variant A"],
            "variant_type": [variant_type],
            "avg_cycle_hours": [avg_cycle],
            "sla_breach_rate": [sla_rate],
            "avg_wait_ratio": [wait_ratio],
            "frequency": [frequency],
        })

    def test_returns_recommendation_column(self):
        vdf = self._make_variant_df("Conformant")
        result = generate_variant_recommendations(vdf)
        assert "recommendation" in result.columns

    def test_conformant_text(self):
        vdf = self._make_variant_df("Conformant")
        result = generate_variant_recommendations(vdf)
        rec = result.iloc[0]["recommendation"]
        assert "optimal workflow path" in rec

    def test_rework_loop_text(self):
        vdf = self._make_variant_df("Rework loop")
        result = generate_variant_recommendations(vdf)
        rec = result.iloc[0]["recommendation"]
        assert "Repeated steps" in rec
        assert "validation gates" in rec

    def test_skip_text(self):
        vdf = self._make_variant_df("Skip")
        result = generate_variant_recommendations(vdf)
        rec = result.iloc[0]["recommendation"]
        assert "bypassed" in rec

    def test_extended_text(self):
        vdf = self._make_variant_df("Extended")
        result = generate_variant_recommendations(vdf)
        rec = result.iloc[0]["recommendation"]
        assert "Extra steps" in rec

    def test_cycle_time_context_triggered(self):
        vdf = self._make_variant_df("Conformant", avg_cycle=15.0)
        result = generate_variant_recommendations(vdf)
        rec = result.iloc[0]["recommendation"]
        assert "15.0h" in rec
        assert "significant delay" in rec

    def test_sla_breach_context_triggered(self):
        vdf = self._make_variant_df("Conformant", sla_rate=0.55)
        result = generate_variant_recommendations(vdf)
        rec = result.iloc[0]["recommendation"]
        assert "55%" in rec
        assert "reliability risk" in rec

    def test_wait_ratio_context_triggered(self):
        vdf = self._make_variant_df("Conformant", wait_ratio=0.65)
        result = generate_variant_recommendations(vdf)
        rec = result.iloc[0]["recommendation"]
        assert "idle waiting" in rec

    def test_no_context_below_thresholds(self):
        vdf = self._make_variant_df("Conformant", avg_cycle=5.0, sla_rate=0.1,
                                     wait_ratio=0.2)
        result = generate_variant_recommendations(vdf)
        rec = result.iloc[0]["recommendation"]
        assert "significant delay" not in rec
        assert "reliability risk" not in rec
        assert "idle waiting" not in rec

    def test_recovery_text_when_above_baseline(self):
        """Upgrade 1: non-baseline variant should show recovery hours."""
        vdf = pd.DataFrame({
            "variant_label": ["Variant A", "Variant B"],
            "variant_type": ["Conformant", "Rework loop"],
            "avg_cycle_hours": [5.0, 15.0],
            "sla_breach_rate": [0.0, 0.5],
            "avg_wait_ratio": [0.1, 0.3],
            "frequency": [20, 10],
        })
        result = generate_variant_recommendations(vdf)
        # Variant B has 10h extra * 10 cases = 100h recovery
        rec_b = result[result["variant_label"] == "Variant B"].iloc[0]["recommendation"]
        assert "recover approximately" in rec_b
        assert "100" in rec_b

    def test_no_recovery_for_baseline(self):
        """Upgrade 1: the fastest variant should NOT show recovery text."""
        vdf = pd.DataFrame({
            "variant_label": ["Variant A", "Variant B"],
            "variant_type": ["Conformant", "Rework loop"],
            "avg_cycle_hours": [5.0, 15.0],
            "sla_breach_rate": [0.0, 0.5],
            "avg_wait_ratio": [0.1, 0.3],
            "frequency": [20, 10],
        })
        result = generate_variant_recommendations(vdf)
        rec_a = result[result["variant_label"] == "Variant A"].iloc[0]["recommendation"]
        assert "recover approximately" not in rec_a

    def test_empty_input(self, empty_df):
        result = generate_variant_recommendations(empty_df)
        assert "recommendation" in result.columns
        assert result.empty
