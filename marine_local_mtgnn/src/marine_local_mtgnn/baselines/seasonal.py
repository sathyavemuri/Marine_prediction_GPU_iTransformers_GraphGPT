"""Seasonal baseline with daily and weekly patterns."""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DailySeasonalBaseline:
    """Daily seasonal baseline: average of same time-of-day."""

    def __init__(self, minutes_per_day: int = 24 * 60, cadence_minutes: int = 15):
        """
        Initialize daily seasonal baseline.

        Parameters
        ----------
        minutes_per_day : int
            Minutes per day (default 1440)
        cadence_minutes : int
            Time cadence in minutes (default 15)
        """
        self.minutes_per_day = minutes_per_day
        self.cadence_minutes = cadence_minutes
        self.period = minutes_per_day // cadence_minutes
        self.daily_mean = None

    def fit(self, df_train: pd.DataFrame):
        """
        Fit daily seasonal baseline.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data with datetime index
        """
        # Extract time of day in minutes
        hours = df_train.index.hour
        minutes = df_train.index.minute
        time_of_day_minutes = hours * 60 + minutes

        # Convert to time period index (0 to period-1)
        time_of_day = (time_of_day_minutes // self.cadence_minutes) % self.period

        # Compute mean for each time-of-day
        self.daily_mean = []
        for t in range(self.period):
            mask = time_of_day == t
            if mask.any():
                mean_val = df_train.loc[mask].mean()
            else:
                # If no data for this time slot, create NaN row
                mean_val = pd.Series(np.full(df_train.shape[1], np.nan), index=df_train.columns)
            self.daily_mean.append(mean_val.values)

        self.daily_mean = np.array(self.daily_mean)
        logger.debug(f"Fitted daily seasonal baseline with period={self.period}")

    def forecast(
        self,
        df_history: pd.DataFrame,
        horizon_steps: int,
    ) -> np.ndarray:
        """
        Generate daily seasonal forecast.

        Parameters
        ----------
        df_history : pd.DataFrame
            History (only used for start time)
        horizon_steps : int
            Number of future steps

        Returns
        -------
        np.ndarray
            Forecast of shape (horizon_steps, n_targets)
        """
        if self.daily_mean is None:
            raise ValueError("Must fit baseline before forecasting")

        # Get start time
        start_time = df_history.index[-1]
        start_hour = start_time.hour
        start_minute = start_time.minute
        start_t = (start_hour * 60 + start_minute) // self.cadence_minutes

        # Generate forecast
        forecast = []
        for i in range(horizon_steps):
            t = (start_t + i + 1) % self.period
            forecast.append(self.daily_mean[t])

        return np.array(forecast)


class WeeklySeasonalBaseline:
    """Weekly seasonal baseline: average of same day-of-week and time-of-day."""

    def __init__(self, cadence_minutes: int = 15):
        """
        Initialize weekly seasonal baseline.

        Parameters
        ----------
        cadence_minutes : int
            Time cadence in minutes (default 15)
        """
        self.cadence_minutes = cadence_minutes
        self.period = 7 * 24 * 60 // cadence_minutes  # steps per week
        self.weekly_mean = None

    def fit(self, df_train: pd.DataFrame):
        """
        Fit weekly seasonal baseline.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data with datetime index
        """
        # Group by day-of-week and time-of-day
        day_of_week = df_train.index.dayofweek
        hours = df_train.index.hour
        minutes_mod = df_train.index.minute

        # Time within week (minutes)
        time_of_week = (
            day_of_week * 24 * 60 +
            hours * 60 +
            minutes_mod
        ) // self.cadence_minutes

        # Compute mean for each time-of-week
        self.weekly_mean = []
        for t in range(self.period):
            mask = time_of_week == t
            if mask.any():
                mean_val = df_train.loc[mask].mean()
            else:
                mean_val = df_train.iloc[0] * np.nan
            self.weekly_mean.append(mean_val)

        self.weekly_mean = np.array([m.values for m in self.weekly_mean])
        logger.debug(f"Fitted weekly seasonal baseline with period={self.period}")

    def forecast(
        self,
        df_history: pd.DataFrame,
        horizon_steps: int,
    ) -> np.ndarray:
        """
        Generate weekly seasonal forecast.

        Parameters
        ----------
        df_history : pd.DataFrame
            History (only used for start time)
        horizon_steps : int
            Number of future steps

        Returns
        -------
        np.ndarray
            Forecast of shape (horizon_steps, n_targets)
        """
        if self.weekly_mean is None:
            raise ValueError("Must fit baseline before forecasting")

        # Get start time
        start_time = df_history.index[-1]
        start_dow = start_time.dayofweek
        start_hour = start_time.hour
        start_minute = start_time.minute
        start_t = (
            (start_dow * 24 * 60 + start_hour * 60 + start_minute) // self.cadence_minutes
        )

        # Generate forecast
        forecast = []
        for i in range(horizon_steps):
            t = (start_t + i + 1) % self.period
            forecast.append(self.weekly_mean[t])

        return np.array(forecast)
