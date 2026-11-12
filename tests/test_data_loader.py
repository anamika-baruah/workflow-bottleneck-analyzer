import pandas as pd
import pytest

from src.data_loader import apply_mapping, read_csv_bytes, suggest_mapping


class TestSuggestMapping:
    def test_recognises_our_own_names(self):
        m = suggest_mapping(["case_id", "task", "start_time", "end_time", "user"])
        assert m == {"case_id": "case_id", "task": "task", "start_time": "start_time",
                     "end_time": "end_time", "user": "user"}

    def test_recognises_common_export_names(self):
        m = suggest_mapping(["Ticket ID", "Status", "Started At", "Completed At", "Assignee"])
        assert m["case_id"] == "Ticket ID"
        assert m["task"] == "Status"
        assert m["start_time"] == "Started At"
        assert m["end_time"] == "Completed At"
        assert m["user"] == "Assignee"

    def test_unknown_columns_give_no_guess_and_no_column_is_used_twice(self):
        m = suggest_mapping(["foo", "bar"])
        assert all(v is None for v in m.values())
        picked = [v for v in suggest_mapping(["id", "step", "start", "end"]).values() if v]
        assert len(picked) == len(set(picked))


class TestApplyMapping:
    def _df(self):
        return pd.DataFrame({
            "Ticket": [" T1 ", "T1"], "Stage": ["Open", "Close"],
            "From": ["2026-01-01 09:00", "2026-01-01 10:00"],
            "To": ["2026-01-01 09:30", "2026-01-01 10:20"],
        })

    def _map(self, **kw):
        base = {"case_id": "Ticket", "task": "Stage", "start_time": "From", "end_time": "To", "user": None}
        base.update(kw)
        return base

    def test_renames_parses_and_trims(self):
        out = apply_mapping(self._df(), self._map())
        assert list(out.columns[:4]) == ["case_id", "task", "start_time", "end_time"]
        assert out["case_id"].iloc[0] == "T1"
        assert pd.api.types.is_datetime64_any_dtype(out["start_time"])

    def test_user_is_optional(self):
        assert (apply_mapping(self._df(), self._map())["user"] == "Unknown").all()

    def test_missing_required_field_is_a_clear_error(self):
        with pytest.raises(ValueError, match="Choose a column for: end_time"):
            apply_mapping(self._df(), self._map(end_time=None))

    def test_same_column_twice_is_rejected(self):
        with pytest.raises(ValueError, match="own column"):
            apply_mapping(self._df(), self._map(end_time="From"))

    def test_text_column_chosen_as_time_is_rejected(self):
        df = self._df()
        df["Note"] = ["hello", "world"]
        with pytest.raises(ValueError, match="could not be read as dates"):
            apply_mapping(df, self._map(start_time="Note"))

    def test_some_bad_timestamps_become_missing_not_an_error(self):
        df = self._df()
        df.loc[1, "To"] = "not a date"
        assert apply_mapping(df, self._map())["end_time"].isna().sum() == 1


class TestReadCsv:
    def test_handles_byte_order_mark(self):
        df = read_csv_bytes("﻿a,b\n1,2\n".encode("utf-8"))
        assert list(df.columns) == ["a", "b"]

    def test_empty_and_header_only_files_fail_clearly(self):
        with pytest.raises(ValueError):
            read_csv_bytes(b"")
        with pytest.raises(ValueError):
            read_csv_bytes(b"a,b\n")


class TestDayFirstDates:
    def test_day_first_is_detected_from_the_data(self):
        from src.data_loader import parse_times
        out = parse_times(pd.Series(["13/02/2026 09:00", "03/02/2026 10:00"]))
        assert out.iloc[0] == pd.Timestamp("2026-02-13 09:00")
        assert out.iloc[1] == pd.Timestamp("2026-02-03 10:00")   # 3 Feb, not 2 Mar

    def test_iso_dates_are_untouched(self):
        from src.data_loader import parse_times
        out = parse_times(pd.Series(["2026-02-03 10:00:00"]))
        assert out.iloc[0] == pd.Timestamp("2026-02-03 10:00")

    def test_month_first_stays_month_first(self):
        from src.data_loader import parse_times
        out = parse_times(pd.Series(["02/13/2026 09:00", "02/03/2026 10:00"]))
        assert out.iloc[1] == pd.Timestamp("2026-02-03 10:00")