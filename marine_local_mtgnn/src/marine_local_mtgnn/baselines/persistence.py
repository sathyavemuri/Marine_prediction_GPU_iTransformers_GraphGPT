"""Persistence and seasonal persistence baselines."""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class PersistenceBaseline:
    """Persistence baseline: repeat last observation."""

    def __init__(self):
        """Initialize persistence baseline (stateless)."""
        pass

    def fit(self, df_train: pd.DataFrame):
        """
        Fit persistence baseline (no-op, stateless).

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data (unused)
        """
        pass

    def forecast(
        self,
        df_history: pd.DataFrame,
        horizon_steps: int,
    ) -> np.ndarray:
        """
        Generate persistence forecast.

        Parameters
        ----------
        df_history : pd.DataFrame
            History (last observation will be repeated)
        horizon_steps : int
            Number of future steps

        Returns
        -------
        np.ndarray
            Forecast of shape (horizon_steps, n_targets)
        """
        last_obs = df_history.iloc[-1].values
        forecast = np.tile(last_obs, (horizon_steps, 1))
        return forecast


class SeasonalBaseline:
    """Seasonal persistence baseline: repeat observation from same time period."""

    def __init__(self, lag_steps: int = 672):
        """
        Initialize seasonal baseline.

        Parameters
        ----------
        lag_steps : int
            Lag to use for seasonal pattern (default 672 = 7 days at 15-minute cadence)
        """
        self.lag_steps = lag_steps

    def fit(self, df_train: pd.DataFrame):
        """
        Fit seasonal baseline (no-op, only stores lag).

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data (unused)
        """
        pass

    def forecast(
        self,
        df_history: pd.DataFrame,
        horizon_steps: int,
    ) -> np.ndarray:
        """
        Generate seasonal persistence forecast.

        Parameters
        ----------
        df_history : pd.DataFrame
            History (must have at least lag_steps rows)
        horizon_steps : int
            Number of future steps

        Returns
        -------
        np.ndarray
            Forecast of shape (horizon_steps, n_targets)
        """
        if len(df_history) < self.lag_steps:
            # Fallback to persistence if not enough history
            last_obs = df_history.iloc[-1].values
            forecast = np.tile(last_obs, (horizon_steps, 1))
        else:
            # Use seasonal lag
            seasonal_start = len(df_history) - self.lag_steps
            seasonal_data = df_history.iloc[seasonal_start:].values

            # Repeat seasonal pattern cyclically
            n_targets = seasonal_data.shape[1]
            forecast = np.zeros((horizon_steps, n_targets))

            for i in range(horizon_steps):
                idx = i % len(seasonal_data)
                forecast[i] = seasonal_data[idx]

        return forecast
