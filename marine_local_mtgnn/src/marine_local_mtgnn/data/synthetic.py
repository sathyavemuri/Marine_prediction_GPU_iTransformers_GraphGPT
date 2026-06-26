"""Generate synthetic marine data for testing."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from ..constants import RAW_CSV_COLUMNS


def generate_synthetic_data(
    start_date: str = "2026-02-23",
    num_days: int = 120,
    cadence_minutes: int = 1,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic marine data for testing/development.

    Parameters
    ----------
    start_date : str
        Start date (ISO 8601 format)
    num_days : int
        Number of days of data to generate
    cadence_minutes : int
        Time cadence in minutes (default 1 for raw data)
    output_path : str | Path | None
        If provided, save to CSV

    Returns
    -------
    pd.DataFrame
        Generated data with RAW_CSV_COLUMNS
    """
    # Generate timestamps
    start = pd.Timestamp(start_date, tz="UTC")
    periods = int(num_days * 24 * 60 / cadence_minutes)
    timestamps = pd.date_range(start=start, periods=periods, freq=f"{cadence_minutes}min")

    # Generate realistic synthetic data
    np.random.seed(42)
    t = np.arange(periods) / periods * 2 * np.pi  # normalized time

    data = {
        "timestamp": timestamps,
        # Scalar parameters with diurnal/seasonal variation
        "air_temp_c": 15 + 5 * np.sin(t) + np.random.normal(0, 0.5, periods),
        "air_pressure_hpa": 1010 + 5 * np.sin(t * 0.5) + np.random.normal(0, 1, periods),
        "water_temp_c": 18 + 3 * np.sin(t * 0.5) + np.random.normal(0, 0.3, periods),
        "dew_point_c": 10 + 4 * np.sin(t) + np.random.normal(0, 0.5, periods),
        "salinity_psu": 35 + 0.5 * np.sin(t * 0.1) + np.random.normal(0, 0.1, periods),
        "tidal_level_m": 1.5 * np.sin(t * 24 / 12.42) + np.random.normal(0, 0.1, periods),  # ~12.42h tidal period
        # Direction parameters (0-360)
        "wind_direction_deg": (180 + 90 * np.sin(t) + np.random.normal(0, 10, periods)) % 360,
        "current_direction_deg": (180 + 45 * np.cos(t * 0.5) + np.random.normal(0, 5, periods)) % 360,
        # Speed parameters (non-negative)
        "wind_speed_ms": np.maximum(5 + 3 * np.sin(t) + np.random.normal(0, 0.5, periods), 0),
        "current_speed_ms": np.maximum(0.3 + 0.2 * np.sin(t * 0.5) + np.random.normal(0, 0.05, periods), 0),
        # Wave parameters (non-negative)
        "significant_wave_height_m": np.maximum(1.5 + 0.5 * np.sin(t) + np.random.normal(0, 0.2, periods), 0),
        "significant_wave_period_s": np.maximum(8 + 2 * np.sin(t * 0.5) + np.random.normal(0, 0.5, periods), 0),
        "zero_crossing_period_s": np.maximum(6 + 1.5 * np.sin(t * 0.5) + np.random.normal(0, 0.4, periods), 0),
        "peak_wave_period_s": np.maximum(10 + 2 * np.sin(t * 0.5) + np.random.normal(0, 0.6, periods), 0),
        # Radiation (non-negative)
        "global_radiation_wm2": np.maximum(
            200 + 400 * np.maximum(np.sin(t), 0) + np.random.normal(0, 20, periods),
            0,
        ),
        # Input-only parameters
        "relative_humidity_pct": np.clip(60 + 20 * np.sin(t) + np.random.normal(0, 2, periods), 0, 100),
        "conductivity_mscm": np.maximum(50 + 5 * np.sin(t * 0.1) + np.random.normal(0, 1, periods), 0),
        "compass_deg": (180 + 90 * np.sin(t) + np.random.normal(0, 10, periods)) % 360,
    }

    df = pd.DataFrame(data)

    # Ensure column order matches RAW_CSV_COLUMNS
    df = df[RAW_CSV_COLUMNS]

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated synthetic data: {output_path}")

    return df
