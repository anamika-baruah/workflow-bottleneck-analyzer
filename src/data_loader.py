"""
data_loader.py

Loads workflow CSV files and validates them before anything else runs.
Raises early if columns are missing or timestamps can't be parsed.
"""

import os
import pandas as pd


# every valid workflow log needs these
REQUIRED_COLUMNS = {"case_id", "task", "start_time", "end_time", "user"}


def load_workflow_data(file_path: str) -> pd.DataFrame:
    """Load and validate a workflow event-log CSV.

    Checks file existence, validates columns, parses timestamps, and
    strips whitespace from string fields. Fails loudly if anything's wrong —
    better to catch bad data here than get silent errors downstream.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with proper datetime columns.

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.
    ValueError
        If required columns are missing or timestamps can't be parsed.
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Workflow log file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    # quick check — fail immediately if something's missing
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"The following required columns are missing from "
            f"'{file_path}': {', '.join(sorted(missing))}"
        )

    # parse timestamps — raise if it fails, don't silently coerce to NaT
    # this might break if the format is unusual, but that's intentional
    for col in ("start_time", "end_time"):
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception as exc:
            raise ValueError(
                f"Could not parse column '{col}' as datetime: {exc}"
            ) from exc

    # strip extra whitespace from string columns
    for col in ("case_id", "task", "user"):
        df[col] = df[col].astype(str).str.strip()

    return df
