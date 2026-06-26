"""State-space (UnobservedComponents) models for atmospheric variables."""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import joblib
from statsmodels.tsa.statespace.structural import UnobservedComponents

logger = logging.getLogger(__name__)


class AirTemperatureModel:
    """Air temperature: harmonic baseline + UnobservedComponents anomaly."""

    def __init__(self):
        self.harmonic_baseline = None
        self.ucm_results = None
        self.fitted = False

    def fit(self, timestamps: pd.DatetimeIndex, values: np.ndarray, train_mask: np.ndarray):
        """
        Fit harmonic baseline and UnobservedComponents to anomaly.

        Parameters
        ----------
        timestamps : pd.DatetimeIndex
        values : np.ndarray
            Air temperature values
        train_mask : np.ndarray
            Boolean mask for training data
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from calendar_features import HarmonicBaseline

        # Fit harmonic baseline on train data
        self.harmonic_baseline = HarmonicBaseline()
        self.harmonic_baseline.fit(timestamps, values, train_mask)

        # Compute anomaly on train data
        baseline_pred = self.harmonic_baseline.predict(timestamps[train_mask])
        anomaly_train = values[train_mask] - baseline_pred

        # Fit UnobservedComponents to anomaly
        try:
            ucm_model = UnobservedComponents(
                anomaly_train,
                level=True,
                stochastic_level=True,
                autoregressive=1,
            )
            self.ucm_results = ucm_model.fit(disp=False)
            logger.info(f"Air temperature UCM fitted, AIC: {self.ucm_results.aic:.2f}")
            self.fitted = True
        except Exception as e:
            logger.warning(f"UCM fit failed: {e}, falling back to harmonic baseline only")
            self.ucm_results = None
            self.fitted = True  # Still consider it fitted (baseline-only mode)

    def predict(self, timestamps: pd.DatetimeIndex, steps: int) -> np.ndarray:
        """
        Predict air temperature: baseline + UCM anomaly.

        Parameters
        ----------
        timestamps : pd.DatetimeIndex
            Timestamps up to forecast start
        steps : int
            Number of steps to forecast

        Returns
        -------
        np.ndarray
            Forecasted temperature values
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")

        baseline_future = self.harmonic_baseline.predict(timestamps[-1:])  # Next baseline value

        if self.ucm_results is None:
            # Baseline only
            return np.full(steps, baseline_future[0])

        # Get UCM forecast
        try:
            forecast_result = self.ucm_results.get_forecast(steps=steps)
            anomaly_forecast = forecast_result.predicted_mean
            if hasattr(anomaly_forecast, 'values'):
                anomaly_forecast = anomaly_forecast.values
            anomaly_forecast = np.asarray(anomaly_forecast).flatten()
        except Exception as e:
            logger.warning(f"UCM forecast failed: {e}, returning baseline")
            return np.full(steps, baseline_future[0])

        # Reconstruct
        return baseline_future[0] + anomaly_forecast


class AirPressureModel:
    """Air pressure: damped persistence baseline."""

    def __init__(self, decay_time_hours: float = 48.0, cadence_minutes: float = 15.0):
        """
        Initialize damped persistence model.

        Parameters
        ----------
        decay_time_hours : float
            Time constant for decay in hours
        cadence_minutes : float
            Data cadence in minutes (default 15)
        """
        self.decay_time_hours = decay_time_hours
        self.cadence_minutes = cadence_minutes
        self.long_term_mean = None
        self.fitted = False

    def fit(self, values: np.ndarray, train_mask: np.ndarray):
        """Fit long-term mean on training data."""
        self.long_term_mean = np.mean(values[train_mask])
        self.fitted = True
        logger.info(f"Pressure damped persistence fitted, LTM: {self.long_term_mean:.2f}")

    def predict(self, latest_value: float, steps: int) -> np.ndarray:
        """
        Predict pressure: P(h) = LTM + exp(-h/tau) * (P0 - LTM).

        Parameters
        ----------
        latest_value : float
            Most recent observed value
        steps : int
            Number of steps to forecast

        Returns
        -------
        np.ndarray
            Forecasted pressure values
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")

        # Convert decay time to steps
        tau_steps = (self.decay_time_hours * 60) / self.cadence_minutes

        # Compute forecast for each step
        horizons = np.arange(1, steps + 1)
        decay_factors = np.exp(-horizons / tau_steps)

        return self.long_term_mean + decay_factors * (latest_value - self.long_term_mean)
