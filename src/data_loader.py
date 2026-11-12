"""
data_loader.py

Loads workflow event logs and validates them before anything else runs.
Real client exports rarely use our column names, so a log can be loaded
with a column mapping: {"case_id": "Ticket ID", "task": "Status", ...}.
"""

from __future__ import annotations

import io
import os

import pandas as pd

# every valid workflow log needs these; "user" is optional and unused by the analysis
REQUIRED_COLUMNS = ("case_id", "task", "start_time", "end_time")
OPTIONAL_COLUMNS = ("user",)

# lower-case, punctuation-free names that usually mean each field
_SYNONYMS = {
    "case_id": ["caseid", "case", "id", "ticketid", "ticket", "orderid", "order", "requestid",
                "instanceid", "processid", "leadid", "incidentid", "workitem"],
    "task": ["task", "step", "activity", "stage", "status", "action", "event", "taskname",
             "stepname", "activityname", "state"],
    "start_time": ["starttime", "start", "startdate", "started", "startedat", "begin", "begintime",
                   "timestampstart", "from", "opened", "createdat"],
    "end_time": ["endtime", "end", "enddate", "ended", "endedat", "finish", "finished",
                 "completetime", "completed", "completedat", "timestampend", "to", "closed"],
    "user": ["user", "owner", "assignee", "agent", "resource", "performer", "employee",
             "handler", "assignedto"],
}


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


_SLASH_DATE = r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}"


def parse_times(series: pd.Series) -> pd.Series:
    """Parse timestamps, reading 13/02/2026 as day-first when the data shows it.

    Plain pandas reads 03/02/2026 as 2 March. If any value in the column has a
    first number above 12 (so it cannot be a month), the whole column is day-first.
    """
    parts = series.astype(str).str.extract(_SLASH_DATE).astype(float)
    dayfirst = bool(parts[0].gt(12).any()) and not bool(parts[1].gt(12).any())
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, format="mixed")


def suggest_mapping(columns: list[str]) -> dict[str, str | None]:
    """Best guess of which file column holds each field (None when no guess)."""
    by_norm = {}
    for col in columns:
        by_norm.setdefault(_norm(col), col)
    used: set[str] = set()
    mapping: dict[str, str | None] = {}
    for field in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS):
        pick = None
        # an exact match on our own name wins, then synonyms in priority order
        for candidate in [_norm(field), *_SYNONYMS[field]]:
            col = by_norm.get(candidate)
            if col is not None and col not in used:
                pick = col
                break
        if pick is not None:
            used.add(pick)
        mapping[field] = pick
    return mapping


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Rename mapped columns to our names, parse timestamps and tidy text fields."""
    missing = [f for f in REQUIRED_COLUMNS if not mapping.get(f)]
    if missing:
        raise ValueError("Choose a column for: " + ", ".join(missing))

    chosen = [mapping[f] for f in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS) if mapping.get(f)]
    if len(set(chosen)) != len(chosen):
        raise ValueError("Each field needs its own column; one column is selected twice.")
    absent = [c for c in chosen if c not in df.columns]
    if absent:
        raise ValueError("Column not found in the file: " + ", ".join(absent))

    out = pd.DataFrame({f: df[mapping[f]] for f in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS) if mapping.get(f)})
    if "user" not in out.columns:
        out["user"] = "Unknown"

    for col in ("start_time", "end_time"):
        parsed = parse_times(out[col])
        if parsed.notna().sum() == 0:
            raise ValueError(
                f"Column '{mapping[col]}' could not be read as dates or times. "
                "Pick the column that holds timestamps."
            )
        out[col] = parsed

    for col in ("case_id", "task", "user"):
        out[col] = out[col].astype(str).str.strip()
    return out


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    """Read an uploaded CSV, tolerating a UTF-8 byte order mark."""
    try:
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The file is empty.") from exc
    if df.empty:
        raise ValueError("The file has a header but no rows.")
    return df


def load_workflow_data(file_path: str, mapping: dict[str, str | None] | None = None) -> pd.DataFrame:
    """Load a workflow event-log CSV, optionally with a column mapping.

    Without a mapping the file must already use our column names.
    Unparseable timestamps become NaT and are counted and dropped later in
    preprocessing, so they show up in the data-quality note.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Workflow log file not found: {file_path}")

    df = pd.read_csv(file_path, encoding="utf-8-sig")
    if mapping is None:
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"The following required columns are missing from '{file_path}': "
                + ", ".join(sorted(missing))
            )
        mapping = {f: f for f in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS) if f in df.columns}
    return apply_mapping(df, mapping)