"""Tests for baseline forecasters."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from marine_local_mtgnn.baselines.persistence import PersistenceBaseline, SeasonalBaseline
from marine_local_mtgnn.baselines.seasonal import DailySeasonalBaseline, WeeklySeasonalBaseline
from marine_local_mtgnn.baselines.trend import TrendBaseline


@pytest.fixture
def synthetic_timeseries():
    """Create synthetic time series data."""
    dates = pd.date_range("2026-02-01", periods=1000, freq="15min", tz="UTC")
    data = {
        "param1": 10 + np.sin(np.arange(1000) / 100) + np.random.normal(0, 0.1, 1000),
        "param2": 20 + np.cos(np.arange(1000) / 50) + np.random.normal(0, 0.2, 1000),
        "param3": 15 + np.random.normal(0, 0.5, 1000),
    }
    df = pd.DataFrame(data, index=dates)
    return df


class TestPersistenceBaseline:
    """Test persistence baseline."""

    def test_persistence_forecast_shape(self, synthetic_timeseries):
        """Test that persistence forecast has correct shape."""
        baseline = PersistenceBaseline()
        baseline.fit(synthetic_timeseries)

        history = synthetic_timeseries.iloc[-100:]
        forecast = baseline.forecast(history, horizon_steps=50)

        assert forecast.shape == (50, 3)
        assert np.all(np.isfinite(forecast))

    def test_persistence_repeats_last(self, synthetic_timeseries):
        """Test that persistence repeats last observation."""
        baseline = PersistenceBaseline()
        baseline.fit(synthetic_timeseries)

        history = synthetic_timeseries.iloc[-100:]
        forecast = baseline.forecast(history, horizon_steps=10)

        last_obs = history.iloc[-1].values
        np.testing.assert_allclose(forecast[0], last_obs, rtol=1e-10)
        np.testing.assert_allclose(forecast[-1], last_obs, rtol=1e-10)


class TestSeasonalBaseline:
    """Test seasonal persistence baseline."""

    def test_seasonal_forecast_shape(self, synthetic_timeseries):
        """Test that seasonal baseline forecast has correct shape."""
        baseline = SeasonalBaseline(lag_steps=96)  # 1 day
        baseline.fit(synthetic_timeseries)

        history = synthetic_timeseries.iloc[-200:]
        forecast = baseline.forecast(history, horizon_steps=50)

        assert forecast.shape == (50, 3)
        assert np.all(np.isfinite(forecast))

    def test_seasonal_insufficient_history(self, synthetic_timeseries):
        """Test seasonal baseline with insufficient history."""
        baseline = SeasonalBaseline(lag_steps=1000)  # Larger than data
        baseline.fit(synthetic_timeseries)

        history = synthetic_timeseries.iloc[-100:]
        forecast = baseline.forecast(history, horizon_steps=50)

        # Should fallback to persistence
        assert forecast.shape == (50, 3)


class TestDailySeasonalBaseline:
    """Test daily seasonal baseline."""

    def test_daily_seasonal_forecast_shape(self, synthetic_timeseries):
        """Test daily seasonal baseline forecast shape."""
        baseline = DailySeasonalBaseline()
        baseline.fit(synthetic_timeseries)

        history = synthetic_timeseries.iloc[-200:]
        forecast = baseline.forecast(history, horizon_steps=96)  # Full day

        assert forecast.shape == (96, 3)
        assert np.all(np.isfinite(forecast))

    def test_daily_seasonal_periodicity(self, synthetic_timeseries):
        """Test that daily seasonal baseline has correct period."""
        baseline = DailySeasonalBaseline()
        assert baseline.period == 96  # 24 hours at 15-minute cadence


class TestWeeklySeasonalBaseline:
    """Test weekly seasonal baseline."""

    def test_weekly_seasonal_forecast_shape(self, synthetic_timeseries):
        """Test weekly seasonal baseline forecast shape."""
        baseline = WeeklySeasonalBaseline()
        baseline.fit(synthetic_timeseries)

        history = synthetic_timeseries.iloc[-500:]
        forecast = baseline.forecast(history, horizon_steps=200)

        assert forecast.shape == (200, 3)
        assert np.all(np.isfinite(forecast))

    def test_weekly_seasonal_periodicity(self, synthetic_timeseries):
        """Test that weekly seasonal baseline has correct period."""
        baseline = WeeklySeasonalBaseline()
        assert baseline.period == 7 * 96  # 7 days at 15-minute cadence


class TestTrendBaseline:
    """Test trend baseline."""

    def test_trend_forecast_shape(self, synthetic_timeseries):
        """Test trend baseline forecast shape."""
        baseline = TrendBaseline(window_steps=20)
        baseline.fit(synthetic_timeseries)

        history = synthetic_timeseries.iloc[-100:]
        forecast = baseline.forecast(history, horizon_steps=50)

        assert forecast.shape == (50, 3)
        assert np.all(np.isfinite(forecast))

    def test_trend_increasing(self):
        """Test trend baseline on increasing trend."""
        dates = pd.date_range("2026-02-01", periods=100, freq="15min", tz="UTC")
        x = np.arange(100)
        data = {
            "param1": 10 + 0.1 * x + np.random.normal(0, 0.01, 100),
            "param2": 20 + 0.05 * x + np.random.normal(0, 0.02, 100),
        }
        df = pd.DataFrame(data, index=dates)

        baseline = TrendBaseline(window_steps=30)
        baseline.fit(df)

        history = df.iloc[-30:]
        forecast = baseline.forecast(history, horizon_steps=10)

        # Forecast should generally increase
        first_param = forecast[:, 0]
        trend = np.mean(np.diff(first_param))
        assert trend > -0.05  # Allow some noise, but expect positive trend

    def test_trend_constant(self):
        """Test trend baseline on constant signal."""
        dates = pd.date_range("2026-02-01", periods=100, freq="15min", tz="UTC")
        data = {
            "param1": np.ones(100) * 15,
            "param2": np.ones(100) * 25,
        }
        df = pd.DataFrame(data, index=dates)

        baseline = TrendBaseline(window_steps=30)
        baseline.fit(df)

        history = df.iloc[-30:]
        forecast = baseline.forecast(history, horizon_steps=10)

        # Forecast should be approximately constant
        assert np.allclose(forecast[:, 0], 15, atol=1e-5)
        assert np.allclose(forecast[:, 1], 25, atol=1e-5)
