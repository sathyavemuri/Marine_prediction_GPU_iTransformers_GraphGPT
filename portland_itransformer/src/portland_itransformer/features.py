"""Feature transformation utilities."""

import numpy as np
import pandas as pd
from typing import Tuple


def speed_dir_to_uv(
    speed_ms: np.ndarray,
    direction_deg: np.ndarray,
    convention: str = "from"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert speed and direction to u/v components.

    Parameters
    ----------
    speed_ms : array
        Wind/current speed in m/s
    direction_deg : array
        Direction in degrees (0-360)
    convention : str
        'from' (meteorological) or 'to' (oceanographic)

    Returns
    -------
    u_east, v_north : arrays
        East and north components in m/s
    """
    theta = np.deg2rad(np.mod(direction_deg, 360.0))

    if convention == "from":
        # Meteorological: wind is FROM this direction
        u_east = -speed_ms * np.sin(theta)
        v_north = -speed_ms * np.cos(theta)
    elif convention == "to":
        # Oceanographic: current is TO this direction
        u_east = speed_ms * np.sin(theta)
        v_north = speed_ms * np.cos(theta)
    else:
        raise ValueError("convention must be 'from' or 'to'")

    return u_east, v_north


def uv_to_speed_dir(
    u_east: np.ndarray,
    v_north: np.ndarray,
    convention: str = "from"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert u/v components to speed and direction.

    Parameters
    ----------
    u_east : array
        East component in m/s
    v_north : array
        North component in m/s
    convention : str
        'from' (meteorological) or 'to' (oceanographic)

    Returns
    -------
    speed_ms, direction_deg : arrays
        Speed in m/s and direction in degrees [0, 360)
    """
    speed = np.hypot(u_east, v_north)
    direction_to = (np.rad2deg(np.arctan2(u_east, v_north)) + 360.0) % 360.0

    if convention == "from":
        # Meteorological: wind is FROM this direction
        direction = (direction_to + 180.0) % 360.0
    elif convention == "to":
        # Oceanographic: current is TO this direction
        direction = direction_to
    else:
        raise ValueError("convention must be 'from' or 'to'")

    return speed, direction


def relative_humidity_pct(
    temp_c: np.ndarray,
    dewpoint_c: np.ndarray
) -> np.ndarray:
    """
    Compute relative humidity from temperature and dew point (Magnus formula).

    Parameters
    ----------
    temp_c : array
        Temperature in Celsius
    dewpoint_c : array
        Dew point temperature in Celsius

    Returns
    -------
    rh_pct : array
        Relative humidity as percentage (may be outside [0, 100] before clipping)
    """
    rh = 100.0 * np.exp(
        (17.625 * dewpoint_c) / (243.04 + dewpoint_c)
        - (17.625 * temp_c) / (243.04 + temp_c)
    )
    return rh


def apply_log_transform(
    values: np.ndarray,
    eps: float = 1e-4
) -> np.ndarray:
    """
    Apply log transform to positive values.

    Parameters
    ----------
    values : array
        Input values (assumed positive)
    eps : float
        Small constant for numerical stability

    Returns
    -------
    log_values : array
        log(values + eps)
    """
    return np.log(values + eps)


def inverse_log_transform(
    log_values: np.ndarray,
    eps: float = 1e-4
) -> np.ndarray:
    """
    Invert log transform.

    Parameters
    ----------
    log_values : array
        Log-transformed values
    eps : float
        Small constant used in forward transform

    Returns
    -------
    values : array
        exp(log_values) - eps, clipped to >= 0
    """
    return np.maximum(np.exp(log_values) - eps, 0.0)


def apply_log1p_transform(values: np.ndarray) -> np.ndarray:
    """
    Apply log1p transform: log(1 + x).

    Parameters
    ----------
    values : array
        Input values

    Returns
    -------
    log_values : array
        log1p(values)
    """
    return np.log1p(values)


def inverse_log1p_transform(log_values: np.ndarray) -> np.ndarray:
    """
    Invert log1p transform: expm1(x).

    Parameters
    ----------
    log_values : array
        Log1p-transformed values

    Returns
    -------
    values : array
        expm1(log_values), clipped to >= 0
    """
    return np.maximum(np.expm1(log_values), 0.0)


def normalize_degrees(degrees: np.ndarray) -> np.ndarray:
    """Normalize degree values to [0, 360)."""
    return np.mod(degrees, 360.0)


def clip_positive(values: np.ndarray) -> np.ndarray:
    """Clip values to be >= 0."""
    return np.maximum(values, 0.0)


def create_cyclical_features(
    timestamps: pd.DatetimeIndex
) -> dict:
    """
    Create cyclical hour and day-of-year features.

    Parameters
    ----------
    timestamps : DatetimeIndex
        UTC timestamps

    Returns
    -------
    features : dict
        'hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos'
    """
    # Ensure it's a DatetimeIndex, not a Series
    if isinstance(timestamps, pd.Series):
        timestamps = pd.DatetimeIndex(timestamps)

    # Hour of day (0-23)
    hour = timestamps.hour
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    # Day of year (1-366)
    dayofyear = timestamps.dayofyear
    dayofyear_sin = np.sin(2 * np.pi * dayofyear / 366.0)
    dayofyear_cos = np.cos(2 * np.pi * dayofyear / 366.0)

    return {
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'dayofyear_sin': dayofyear_sin,
        'dayofyear_cos': dayofyear_cos,
    }
