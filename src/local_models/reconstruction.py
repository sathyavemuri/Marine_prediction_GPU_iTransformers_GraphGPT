"""Reconstruction: convert model outputs to physical units with constraint enforcement."""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class PhysicalReconstruction:
    """Convert scaled/transformed outputs back to physical units and enforce constraints."""

    def __init__(self):
        self.scalers = {}  # Store scalers for each variable
        self.baselines = {}  # Store baseline functions

    def register_scaler(self, variable_name: str, scaler):
        """Register a scaler for a variable."""
        self.scalers[variable_name] = scaler

    def register_baseline(self, variable_name: str, baseline):
        """Register a baseline function for a variable."""
        self.baselines[variable_name] = baseline

    def inverse_scale(self, variable_name: str, scaled_values: np.ndarray) -> np.ndarray:
        """
        Inverse scale values.

        Parameters
        ----------
        variable_name : str
        scaled_values : np.ndarray

        Returns
        -------
        np.ndarray
            Unscaled values
        """
        if variable_name not in self.scalers:
            logger.warning(f"No scaler registered for {variable_name}, returning as-is")
            return scaled_values

        scaler = self.scalers[variable_name]
        if hasattr(scaler, 'inverse_transform'):
            return scaler.inverse_transform(scaled_values.reshape(-1, 1)).flatten()
        return scaled_values

    def reconstruct_temperature(self, air_temp: np.ndarray, dew_point: np.ndarray) -> tuple:
        """
        Enforce dew_point <= air_temp.

        Parameters
        ----------
        air_temp : np.ndarray
        dew_point : np.ndarray

        Returns
        -------
        tuple
            (air_temp, dew_point) with constraints enforced
        """
        dew_point = np.minimum(dew_point, air_temp)
        return air_temp, dew_point

    def reconstruct_wind(self, u: np.ndarray, v: np.ndarray) -> tuple:
        """
        Convert u/v to speed and direction.

        Parameters
        ----------
        u : np.ndarray
            Eastward component
        v : np.ndarray
            Northward component

        Returns
        -------
        tuple
            (speed, direction_deg) where direction in [0, 360)
        """
        from .wind_vector_model import WindDerivation

        speed, direction_deg = WindDerivation.uv_to_speed_direction(u, v)

        # Enforce: speed >= 0
        speed = np.maximum(speed, 0.0)

        # Enforce: direction in [0, 360)
        direction_deg = np.where(direction_deg < 0, direction_deg + 360, direction_deg)
        direction_deg = np.where(direction_deg >= 360, direction_deg - 360, direction_deg)

        return speed, direction_deg

    def reconstruct_humidity(self, air_temp: np.ndarray, dew_point: np.ndarray) -> np.ndarray:
        """
        Compute relative humidity from air_temp and dew_point using Magnus equation.

        RH = 100 * [exp((b*T_d)/(c+T_d)) / exp((b*T)/(c+T))]

        Parameters
        ----------
        air_temp : np.ndarray
            Air temperature (°C)
        dew_point : np.ndarray
            Dew point (°C)

        Returns
        -------
        np.ndarray
            Relative humidity (%), clipped to [0, 100]
        """
        # Magnus formula coefficients
        b = 17.27
        c = 237.7

        numerator = np.exp((b * dew_point) / (c + dew_point))
        denominator = np.exp((b * air_temp) / (c + air_temp))

        rh = 100.0 * (numerator / denominator)

        # Enforce: RH in [0, 100]
        rh = np.clip(rh, 0.0, 100.0)

        return rh

    def reconstruct_radiation(self, log_radiation: np.ndarray) -> np.ndarray:
        """
        Inverse log transform radiation.

        Parameters
        ----------
        log_radiation : np.ndarray
            log1p transformed radiation

        Returns
        -------
        np.ndarray
            Radiation (W/m²), clipped to [0, max_realistic]
        """
        radiation = np.expm1(log_radiation)
        radiation = np.maximum(radiation, 0.0)

        # Clip at physically realistic max (clear-sky max ~1000 W/m²)
        radiation = np.minimum(radiation, 1200.0)

        return radiation

    def reconstruct_waves(self, log_height: np.ndarray, log_period: np.ndarray) -> tuple:
        """
        Inverse log transform wave height and period.

        Parameters
        ----------
        log_height : np.ndarray
            log1p transformed significant wave height
        log_period : np.ndarray
            log1p transformed zero-crossing period

        Returns
        -------
        tuple
            (height, period) both clipped to [0, max]
        """
        height = np.expm1(log_height)
        height = np.maximum(height, 0.0)
        height = np.minimum(height, 15.0)  # Practical max ~15m

        period = np.expm1(log_period)
        period = np.maximum(period, 0.0)
        period = np.minimum(period, 30.0)  # Practical max ~30s

        return height, period

    def reconstruct_salinity(self, salinity: np.ndarray) -> np.ndarray:
        """
        Enforce physical bounds on salinity.

        Parameters
        ----------
        salinity : np.ndarray
            Salinity (PSU)

        Returns
        -------
        np.ndarray
            Salinity clipped to [0, 40] PSU
        """
        salinity = np.clip(salinity, 0.0, 40.0)
        return salinity

    def reconstruct_tides(self, tidal_residual: np.ndarray) -> np.ndarray:
        """
        Validate tidal residual (no hard constraint, but warn if extreme).

        Parameters
        ----------
        tidal_residual : np.ndarray
            Tidal residual (m)

        Returns
        -------
        np.ndarray
            Tidal residual
        """
        # Warn if very large
        if np.any(np.abs(tidal_residual) > 2.0):
            logger.warning("Tidal residual exceeded ±2.0 m, may indicate instability")

        return tidal_residual
