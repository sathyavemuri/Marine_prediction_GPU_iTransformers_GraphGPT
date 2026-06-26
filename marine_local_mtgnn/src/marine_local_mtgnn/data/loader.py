"""Data loading and parsing for marine local MTGNN."""

from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from ..constants import RAW_CSV_COLUMNS

logger = logging.getLogger(__name__)


def load_raw_csv(
    csv_path: str,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """
    Load raw CSV with 1-minute timestamp resolution.

    Parameters
    ----------
    csv_path : str
        Path to raw CSV file
    timezone : str
        Timezone to localize naive timestamps

    Returns
    -------
    pd.DataFrame
        Loaded data with parsed timestamp index

    Raises
    ------
    FileNotFoundError
        If CSV file does not exist
    ValueError
        If required columns are missing
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    # Load CSV
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Localize to specified timezone if data came in naive
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(timezone)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(timezone)

    # Set index
    df = df.set_index("timestamp")

    return df


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> Tuple[bool, list[str]]:
    """
    Validate that all required columns exist.

    Parameters
    ----------
    df : pd.DataFrame
        Data to validate
    required_columns : list[str]
        Expected column names (excluding timestamp)

    Returns
    -------
    Tuple[bool, list[str]]
        (is_valid, missing_columns)
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return False, missing
    return True, []


def compute_cadence(timestamps: pd.DatetimeIndex) -> Tuple[pd.Timedelta, list[dict]]:
    """
    Compute time cadence and detect missing intervals.

    Parameters
    ----------
    timestamps : pd.DatetimeIndex
        Sorted timestamps

    Returns
    -------
    Tuple[pd.Timedelta, list[dict]]
        (nominal_cadence, missing_intervals)

        missing_intervals = [{"start": ts, "end": ts, "gap_minutes": float, "missing_rows": int}]
    """
    if len(timestamps) < 2:
        return pd.Timedelta(minutes=1), []

    # Compute differences
    diffs = timestamps.to_series().diff()[1:]
    nominal_cadence = diffs.mode()[0] if len(diffs.mode()) > 0 else diffs.min()

    # Find gaps
    tolerance = pd.Timedelta(seconds=30)  # Allow small variations
    missing_intervals = []

    for i, (ts, diff) in enumerate(zip(timestamps[1:], diffs, strict=False), 1):
        if diff > nominal_cadence + tolerance:
            gap_duration = diff - nominal_cadence
            expected_rows = int(gap_duration / nominal_cadence)
            missing_intervals.append({
                "start_timestamp": timestamps[i - 1].isoformat(),
                "end_timestamp": ts.isoformat(),
                "gap_minutes": float(gap_duration.total_seconds() / 60),
                "missing_rows": expected_rows,
            })

    return nominal_cadence, missing_intervals


def check_monotonicity(timestamps: pd.DatetimeIndex) -> Tuple[bool, list[dict]]:
    """
    Check for monotonic increasing timestamps and duplicates.

    Parameters
    ----------
    timestamps : pd.DatetimeIndex
        Timestamps to check

    Returns
    -------
    Tuple[bool, list[dict]]
        (is_monotonic, duplicates)

        duplicates = [{"timestamp": ts, "count": int}]
    """
    # Check for duplicates
    duplicates = timestamps[timestamps.duplicated(keep=False)]
    dup_info = []
    if len(duplicates) > 0:
        for ts, count in duplicates.value_counts().items():
            dup_info.append({
                "timestamp": ts.isoformat(),
                "count": int(count),
            })

    is_monotonic = timestamps.is_monotonic_increasing

    return is_monotonic and len(dup_info) == 0, dup_info


def check_value_ranges(
    df: pd.DataFrame,
    ranges: dict[str, Tuple[float, float]]
) -> list[dict]:
    """
    Check for values outside expected ranges.

    Parameters
    ----------
    df : pd.DataFrame
        Data to validate
    ranges : dict[str, Tuple[float, float]]
        Expected (min, max) for each parameter

    Returns
    -------
    list[dict]
        Warnings = [{"parameter": str, "out_of_range_count": int, "min": float, "max": float, "violations": [...]}]
    """
    warnings = []

    for col, (min_val, max_val) in ranges.items():
        if col not in df.columns:
            continue

        data = df[col]
        below_min = data < min_val
        above_max = data > max_val
        out_of_range = below_min | above_max

        if out_of_range.any():
            indices = df.index[out_of_range].tolist()
            warnings.append({
                "parameter": col,
                "out_of_range_count": int(out_of_range.sum()),
                "expected_min": float(min_val),
                "expected_max": float(max_val),
                "actual_min": float(data.min()),
                "actual_max": float(data.max()),
                "first_violation": {
                    "timestamp": indices[0].isoformat(),
                    "value": float(data[indices[0]]),
                } if indices else None,
            })

    return warnings


def summarize_data(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for data quality report.

    Parameters
    ----------
    df : pd.DataFrame
        Data to summarize

    Returns
    -------
    dict
        Summary statistics
    """
    return {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "time_range": {
            "start": df.index.min().isoformat(),
            "end": df.index.max().isoformat(),
            "duration_days": float((df.index.max() - df.index.min()).days),
        },
        "missing_values": {
            col: int(df[col].isna().sum())
            for col in df.columns
        },
    }
