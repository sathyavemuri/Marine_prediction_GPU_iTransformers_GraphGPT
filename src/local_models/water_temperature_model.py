"""Water temperature model using ExponentialSmoothing on anomaly."""

import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.holtwinters import ExponentialSmoothing

logger = logging.getLogger(__name__)


class WaterTemperatureModel:
    """Water temperature: harmonic baseline + ExponentialSmoothing on anomaly."""

    def __init__(self):
        self.harmonic_baseline = None
        self.es_results = None
        self.fitted = False

    def fit(self, timestamps: pd.DatetimeIndex, values: np.ndarray, train_mask: np.ndarray):
        """
        Fit harmonic baseline and ExponentialSmoothing to anomaly.

        Parameters
        ----------
        timestamps : pd.DatetimeIndex
        values : np.ndarray
            Water temperature values (°C)
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

        # Fit ExponentialSmoothing (Holt-Winters)
        try:
            es_model = ExponentialSmoothing(
                anomaly_train,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )
            self.es_results = es_model.fit(optimized=True)
            logger.info(f"Water temperature ExponentialSmoothing fitted, AIC: {self.es_results.aic:.2f}")
            self.fitted = True
        except Exception as e:
            logger.warning(f"ExponentialSmoothing fit failed: {e}, baseline-only mode")
            self.es_results = None
            self.fitted = True

    def predict(self, timestamps: pd.DatetimeIndex, steps: int) -> np.ndarray:
        """
        Predict water temperature: baseline + ES anomaly.

        Parameters
        ----------
        timestamps : pd.DatetimeIndex
            Recent timestamps
        steps : int
            Number of steps to forecast

        Returns
        -------
        np.ndarray
            Forecasted temperature values
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")

        baseline_future = self.harmonic_baseline.predict(timestamps[-1:])

        if self.es_results is None:
            # Baseline only
            return np.full(steps, baseline_future[0])

        # Get ES forecast
        try:
            forecast_result = self.es_results.get_forecast(steps=steps)
            anomaly_forecast = forecast_result.predicted_mean
            if hasattr(anomaly_forecast, 'values'):
                anomaly_forecast = anomaly_forecast.values
            anomaly_forecast = np.asarray(anomaly_forecast).flatten()
        except Exception as e:
            logger.warning(f"ExponentialSmoothing forecast failed: {e}")
            return np.full(steps, baseline_future[0])

        # Reconstruct
        return baseline_future[0] + anomaly_forecast
