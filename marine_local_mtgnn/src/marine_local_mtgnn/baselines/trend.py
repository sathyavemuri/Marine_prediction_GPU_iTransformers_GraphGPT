"""Local trend baseline using Theil-Sen regression."""

import numpy as np
import pandas as pd
from sklearn.linear_model import TheilSenRegressor
import logging

logger = logging.getLogger(__name__)


class TrendBaseline:
    """Local trend baseline: Theil-Sen regression on recent history."""

    def __init__(self, window_steps: int = 12):
        """
        Initialize trend baseline.

        Parameters
        ----------
        window_steps : int
            Number of recent steps to use for trend (default 12 = 3 hours at 15-min cadence)
        """
        self.window_steps = window_steps
        self.regressors = {}

    def fit(self, df_train: pd.DataFrame):
        """
        Fit trend baseline (no-op, only stores window).

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
        Generate trend-based forecast.

        Parameters
        ----------
        df_history : pd.DataFrame
            History (uses last window_steps rows)
        horizon_steps : int
            Number of future steps

        Returns
        -------
        np.ndarray
            Forecast of shape (horizon_steps, n_targets)
        """
        n_targets = df_history.shape[1]
        window_start = max(0, len(df_history) - self.window_steps)
        window = df_history.iloc[window_start:].values

        forecast = np.zeros((horizon_steps, n_targets))

        if len(window) < 2:
            # Not enough data for trend, use last observation
            forecast[:] = df_history.iloc[-1].values
            return forecast

        # Fit Theil-Sen trend for each target
        X = np.arange(len(window)).reshape(-1, 1)

        for j in range(n_targets):
            y = window[:, j]

            # Skip if all NaN
            if np.all(np.isnan(y)):
                forecast[:, j] = df_history.iloc[-1, j]
                continue

            # Handle NaN by linear interpolation
            mask = ~np.isnan(y)
            if not mask.any():
                forecast[:, j] = df_history.iloc[-1, j]
                continue

            if mask.sum() < 2:
                # Not enough valid values for trend
                forecast[:, j] = df_history.iloc[-1, j]
                continue

            try:
                ts = TheilSenRegressor(random_state=42, max_subpopulation=100)
                ts.fit(X[mask], y[mask])

                # Predict future steps
                X_future = np.arange(len(window), len(window) + horizon_steps).reshape(-1, 1)
                forecast[:, j] = ts.predict(X_future)
            except Exception as e:
                logger.debug(f"Theil-Sen fit failed for target {j}: {e}")
                forecast[:, j] = df_history.iloc[-1, j]

        return forecast
