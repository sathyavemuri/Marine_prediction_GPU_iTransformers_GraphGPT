"""Data transformations for marine local MTGNN."""

import numpy as np
import pandas as pd
import logging

from ..constants import LOG_PARAMS, LOG1P_PARAMS, POSITIVE_ONLY, LOG_TRANSFORM_RAW

logger = logging.getLogger(__name__)


def degrees_to_components(
    direction_deg: np.ndarray,
    speed_ms: np.ndarray,
    convention: str = "from",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert direction/speed to u/v components.

    Parameters
    ----------
    direction_deg : np.ndarray
        Direction in degrees (0-360)
    speed_ms : np.ndarray
        Speed in m/s
    convention : str
        "from" (meteorological) or "to" (oceanographic)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (u_east_ms, v_north_ms)

    Notes
    -----
    Meteorological convention: wind/current FROM the given direction
    Oceanographic convention: wind/current TO the given direction

    In both cases, we compute u/v as:
        u = -speed * sin(direction_rad) if convention=="from" else +speed * sin(direction_rad)
        v = -speed * cos(direction_rad) if convention=="from" else +speed * cos(direction_rad)

    This ensures u/v represent the direction the vector points towards.
    """
    # Convert to radians
    direction_rad = np.radians(direction_deg)

    # Trigonometric components
    sin_dir = np.sin(direction_rad)
    cos_dir = np.cos(direction_rad)

    if convention.lower() == "from":
        # Meteorological: FROM direction means vector points opposite
        u = -speed_ms * sin_dir
        v = -speed_ms * cos_dir
    elif convention.lower() == "to":
        # Oceanographic: TO direction means vector points in that direction
        u = speed_ms * sin_dir
        v = speed_ms * cos_dir
    else:
        raise ValueError(f"Unknown convention: {convention}")

    return u, v


def components_to_degrees(
    u: np.ndarray,
    v: np.ndarray,
    convention: str = "from",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct direction/speed from u/v components.

    Parameters
    ----------
    u : np.ndarray
        Eastward component (m/s)
    v : np.ndarray
        Northward component (m/s)
    convention : str
        "from" or "to"

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (direction_deg, speed_ms)
    """
    speed_ms = np.sqrt(u**2 + v**2)

    # Compute direction from components
    # atan2 returns angle from positive x-axis (east), counterclockwise
    # We want meteorological angle: 0° = north, 90° = east, 180° = south, 270° = west
    direction_rad = np.arctan2(u, v)
    direction_deg = np.degrees(direction_rad)

    # Convert to 0-360 range
    direction_deg = (direction_deg + 360) % 360

    # Adjust for convention if needed
    if convention.lower() == "from":
        # Reverse to get "from" direction
        direction_deg = (direction_deg + 180) % 360

    return direction_deg, speed_ms


def apply_log_transform(data: np.ndarray, param_name: str) -> np.ndarray:
    """
    Apply log or log1p transform to parameter.

    Parameters
    ----------
    data : np.ndarray
        Input data
    param_name : str
        Parameter name (raw or log-prefixed, to determine transform type)

    Returns
    -------
    np.ndarray
        Transformed data
    """
    # Check if this is a raw parameter that needs transformation
    if param_name in LOG_TRANSFORM_RAW:
        transform_type = LOG_TRANSFORM_RAW[param_name]
        if transform_type == "log":
            transformed = np.log(np.maximum(data, 1e-6))
            return transformed
        elif transform_type == "log1p":
            transformed = np.log1p(np.maximum(data, 0))
            return transformed

    # Otherwise return unchanged
    return data


def reverse_log_transform(data: np.ndarray, param_name: str) -> np.ndarray:
    """
    Reverse log or log1p transform.

    Parameters
    ----------
    data : np.ndarray
        Transformed data
    param_name : str
        Parameter name (raw or log-prefixed)

    Returns
    -------
    np.ndarray
        Original-scale data
    """
    # Check if this is a raw parameter name
    if param_name in LOG_TRANSFORM_RAW:
        transform_type = LOG_TRANSFORM_RAW[param_name]
        if transform_type == "log":
            return np.exp(data)
        elif transform_type == "log1p":
            return np.expm1(data)

    # Check if this is a log-prefixed name and extract the raw name
    # log_X -> check if we have X in LOG_TRANSFORM_RAW
    if param_name in LOG_PARAMS:
        return np.exp(data)
    elif param_name in LOG1P_PARAMS:
        return np.expm1(data)

    # Otherwise return unchanged
    return data


def clip_positive(data: np.ndarray, param_name: str) -> np.ndarray:
    """
    Clip parameters to non-negative range if required.

    Parameters
    ----------
    data : np.ndarray
        Data to clip
    param_name : str
        Parameter name

    Returns
    -------
    np.ndarray
        Clipped data
    """
    if param_name in POSITIVE_ONLY:
        return np.maximum(data, 0)
    return data


class TransformPipeline:
    """Pipeline for transforming raw data to model input."""

    def __init__(
        self,
        wind_direction_convention: str = "from",
        current_direction_convention: str = "to",
    ):
        """
        Initialize pipeline.

        Parameters
        ----------
        wind_direction_convention : str
            "from" (meteorological) or "to"
        current_direction_convention : str
            "from" or "to" (oceanographic)
        """
        self.wind_convention = wind_direction_convention.lower()
        self.current_convention = current_direction_convention.lower()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw DataFrame to input-ready format.

        Raw columns:
            - wind_direction_deg, wind_speed_ms
            - current_direction_deg, current_speed_ms
            - wave parameters (height, periods)
            - radiation
            - scalars (temperature, pressure, etc.)

        Output columns:
            - wind_u_east_ms, wind_v_north_ms (replacing direction/speed)
            - current_u_east_ms, current_v_north_ms
            - log-transformed wave/radiation
            - compass_sin, compass_cos (from wind_direction)
            - all scalar parameters

        Parameters
        ----------
        df : pd.DataFrame
            Raw data with timestamp index

        Returns
        -------
        pd.DataFrame
            Transformed data
        """
        result = df.copy()

        # Transform wind direction/speed to components
        if "wind_direction_deg" in result.columns and "wind_speed_ms" in result.columns:
            wind_u, wind_v = degrees_to_components(
                result["wind_direction_deg"].values,
                result["wind_speed_ms"].values,
                convention=self.wind_convention,
            )
            result["wind_u_east_ms"] = wind_u
            result["wind_v_north_ms"] = wind_v
            result = result.drop(columns=["wind_direction_deg", "wind_speed_ms"])
            logger.debug("Transformed wind direction/speed to u/v components")

        # Transform current direction/speed to components
        if "current_direction_deg" in result.columns and "current_speed_ms" in result.columns:
            current_u, current_v = degrees_to_components(
                result["current_direction_deg"].values,
                result["current_speed_ms"].values,
                convention=self.current_convention,
            )
            result["current_u_east_ms"] = current_u
            result["current_v_north_ms"] = current_v
            result = result.drop(columns=["current_direction_deg", "current_speed_ms"])
            logger.debug("Transformed current direction/speed to u/v components")

        # Create circular encoding of compass (0-360 wind direction)
        if "wind_u_east_ms" in result.columns and "wind_v_north_ms" in result.columns:
            wind_dir_rad = np.arctan2(result["wind_u_east_ms"].values, result["wind_v_north_ms"].values)
            result["compass_sin"] = np.sin(wind_dir_rad)
            result["compass_cos"] = np.cos(wind_dir_rad)
            # Drop the raw compass_deg if it exists (no longer needed)
            if "compass_deg" in result.columns:
                result = result.drop(columns=["compass_deg"])
            logger.debug("Created circular encoding for compass direction")

        # Apply log transforms to wave and radiation parameters (raw names only)
        for raw_param, transform_type in LOG_TRANSFORM_RAW.items():
            if raw_param in result.columns:
                if transform_type == "log":
                    result[f"log_{raw_param}"] = apply_log_transform(result[raw_param].values, raw_param)
                    result = result.drop(columns=[raw_param])
                    logger.debug(f"Applied log transform to {raw_param}")
                elif transform_type == "log1p":
                    result[f"log1p_{raw_param}"] = apply_log_transform(result[raw_param].values, raw_param)
                    result = result.drop(columns=[raw_param])
                    logger.debug(f"Applied log1p transform to {raw_param}")

        return result

    def reverse_transform(self, df: pd.DataFrame, wind_direction_col: bool = True) -> pd.DataFrame:
        """
        Reverse transformations to get back to original scale.

        Parameters
        ----------
        df : pd.DataFrame
            Transformed data
        wind_direction_col : bool
            If True, reconstruct wind_direction_deg from compass_sin/cos

        Returns
        -------
        pd.DataFrame
            Data in original scale
        """
        result = df.copy()

        # Reverse log transforms
        for param in LOG_PARAMS:
            log_col = f"log_{param}"
            if log_col in result.columns:
                result[param] = reverse_log_transform(result[log_col].values, param)
                result = result.drop(columns=[log_col])

        for param in LOG1P_PARAMS:
            log_col = f"log1p_{param}"
            if log_col in result.columns:
                result[param] = reverse_log_transform(result[log_col].values, param)
                result = result.drop(columns=[log_col])

        # Reconstruct wind direction/speed from components
        if "wind_u_east_ms" in result.columns and "wind_v_north_ms" in result.columns:
            wind_dir, wind_speed = components_to_degrees(
                result["wind_u_east_ms"].values,
                result["wind_v_north_ms"].values,
                convention=self.wind_convention,
            )
            result["wind_direction_deg"] = wind_dir
            result["wind_speed_ms"] = wind_speed
            result = result.drop(columns=["wind_u_east_ms", "wind_v_north_ms"])

        # Reconstruct current direction/speed from components
        if "current_u_east_ms" in result.columns and "current_v_north_ms" in result.columns:
            current_dir, current_speed = components_to_degrees(
                result["current_u_east_ms"].values,
                result["current_v_north_ms"].values,
                convention=self.current_convention,
            )
            result["current_direction_deg"] = current_dir
            result["current_speed_ms"] = current_speed
            result = result.drop(columns=["current_u_east_ms", "current_v_north_ms"])

        # Drop circular encoding
        if "compass_sin" in result.columns:
            result = result.drop(columns=["compass_sin"])
        if "compass_cos" in result.columns:
            result = result.drop(columns=["compass_cos"])

        return result
