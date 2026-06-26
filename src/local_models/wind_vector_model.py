"""Wind vector model using u/v components with damped persistence and climatology."""

import numpy as np
import pandas as pd
import logging
from scipy.stats import circmean, circstd

logger = logging.getLogger(__name__)


class WindVectorModel:
    """Wind speed and direction via u/v components with damped persistence + climatology."""

    def __init__(self, decay_time_hours: float = 24.0, cadence_minutes: float = 15.0):
        """
        Initialize wind vector model.

        Parameters
        ----------
        decay_time_hours : float
            Time constant for decay in hours (default 24)
        cadence_minutes : float
            Data cadence in minutes (default 15)
        """
        self.decay_time_hours = decay_time_hours
        self.cadence_minutes = cadence_minutes
        self.mean_u = None
        self.mean_v = None
        self.std_u = None
        self.std_v = None
        self.fitted = False

    def fit(self, u_values: np.ndarray, v_values: np.ndarray, train_mask: np.ndarray):
        """
        Fit climatological mean and std of u/v components.

        Parameters
        ----------
        u_values : np.ndarray
            Eastward wind component (m/s)
        v_values : np.ndarray
            Northward wind component (m/s)
        train_mask : np.ndarray
            Boolean mask for training data
        """
        self.mean_u = np.mean(u_values[train_mask])
        self.mean_v = np.mean(v_values[train_mask])
        self.std_u = np.std(u_values[train_mask])
        self.std_v = np.std(v_values[train_mask])

        # Ensure non-zero std
        self.std_u = max(self.std_u, 0.1)
        self.std_v = max(self.std_v, 0.1)

        self.fitted = True
        logger.info(f"Wind vector fitted: mean_u={self.mean_u:.2f}, mean_v={self.mean_v:.2f}, "
                   f"std_u={self.std_u:.2f}, std_v={self.std_v:.2f}")

    def predict(self, latest_u: float, latest_v: float, steps: int) -> tuple:
        """
        Predict u/v components with damped persistence.

        Parameters
        ----------
        latest_u : float
            Most recent u component
        latest_v : float
            Most recent v component
        steps : int
            Number of steps to forecast

        Returns
        -------
        tuple
            (u_forecast, v_forecast) as np.ndarray
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")

        # Convert decay time to steps
        tau_steps = (self.decay_time_hours * 60) / self.cadence_minutes

        # Compute decay factors for each horizon
        horizons = np.arange(1, steps + 1)
        decay_factors = np.exp(-horizons / tau_steps)

        # Damped persistence: trend toward climatological mean
        u_forecast = self.mean_u + decay_factors * (latest_u - self.mean_u)
        v_forecast = self.mean_v + decay_factors * (latest_v - self.mean_v)

        return u_forecast, v_forecast


class WindDerivation:
    """Convert u/v components to speed and direction."""

    @staticmethod
    def uv_to_speed_direction(u: np.ndarray, v: np.ndarray, convention: str = 'from') -> tuple:
        """
        Convert u/v to speed and direction.

        Parameters
        ----------
        u : np.ndarray
            Eastward component (m/s)
        v : np.ndarray
            Northward component (m/s)
        convention : str
            'from' (meteorological) or 'to' (oceanographic)

        Returns
        -------
        tuple
            (speed, direction_deg) where direction is in [0, 360) degrees
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'portland_itransformer' / 'src' / 'portland_itransformer'))
        from features import uv_to_speed_dir

        return uv_to_speed_dir(u, v, convention=convention)

    @staticmethod
    def speed_direction_to_uv(speed: np.ndarray, direction_deg: np.ndarray, convention: str = 'from') -> tuple:
        """
        Convert speed and direction to u/v components.

        Parameters
        ----------
        speed : np.ndarray
            Wind speed (m/s)
        direction_deg : np.ndarray
            Direction in degrees [0, 360)
        convention : str
            'to' = direction points toward (oceanographic)
            'from' = direction vector points from (meteorological, default)

        Returns
        -------
        tuple
            (u, v) components (m/s)
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'portland_itransformer' / 'src' / 'portland_itransformer'))
        from features import speed_dir_to_uv

        return speed_dir_to_uv(speed, direction_deg, convention=convention)


class DewPointModel:
    """Dew point: UnobservedComponents on depression (dew_point_depression = air_temp - dew_point)."""

    def __init__(self):
        self.harmonic_baseline = None
        self.ucm_results = None
        self.fitted = False

    def fit(self, timestamps: pd.DatetimeIndex, air_temp: np.ndarray, dew_point: np.ndarray, train_mask: np.ndarray):
        """
        Fit harmonic baseline and UnobservedComponents to dew point depression.

        Parameters
        ----------
        timestamps : pd.DatetimeIndex
        air_temp : np.ndarray
            Air temperature (°C)
        dew_point : np.ndarray
            Dew point (°C)
        train_mask : np.ndarray
            Boolean mask for training data
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from calendar_features import HarmonicBaseline
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        # Compute depression
        depression = air_temp - dew_point
        depression = np.maximum(depression, 0.01)  # Ensure positive

        # Log transform for stability
        log_depression = np.log1p(depression)

        # Fit harmonic baseline
        self.harmonic_baseline = HarmonicBaseline()
        self.harmonic_baseline.fit(timestamps, log_depression, train_mask)

        # Compute anomaly on train data
        baseline_pred = self.harmonic_baseline.predict(timestamps[train_mask])
        anomaly_train = log_depression[train_mask] - baseline_pred

        # Fit UnobservedComponents
        try:
            ucm_model = UnobservedComponents(
                anomaly_train,
                level=True,
                stochastic_level=True,
                autoregressive=1,
            )
            self.ucm_results = ucm_model.fit(disp=False)
            logger.info(f"Dew point UCM fitted, AIC: {self.ucm_results.aic:.2f}")
            self.fitted = True
        except Exception as e:
            logger.warning(f"Dew point UCM fit failed: {e}, baseline-only mode")
            self.ucm_results = None
            self.fitted = True

    def predict(self, timestamps: pd.DatetimeIndex, air_temp_forecast: np.ndarray, steps: int) -> np.ndarray:
        """
        Predict dew point: compute depression, then clamp to air_temp.

        Parameters
        ----------
        timestamps : pd.DatetimeIndex
            Recent timestamps for baseline computation
        air_temp_forecast : np.ndarray
            Forecasted air temperature
        steps : int
            Number of steps

        Returns
        -------
        np.ndarray
            Forecasted dew point (°C), clamped to air_temp_forecast
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")

        # Get baseline for next step
        baseline_future = self.harmonic_baseline.predict(timestamps[-1:])

        if self.ucm_results is None:
            # Baseline only
            depression_log = np.full(steps, baseline_future[0])
        else:
            # Get UCM forecast
            try:
                forecast_result = self.ucm_results.get_forecast(steps=steps)
                anomaly_forecast = forecast_result.predicted_mean
                if hasattr(anomaly_forecast, 'values'):
                    anomaly_forecast = anomaly_forecast.values
                anomaly_forecast = np.asarray(anomaly_forecast).flatten()
                depression_log = baseline_future[0] + anomaly_forecast
            except Exception as e:
                logger.warning(f"Dew point UCM forecast failed: {e}")
                depression_log = np.full(steps, baseline_future[0])

        # Invert log transform
        depression = np.expm1(depression_log)
        depression = np.maximum(depression, 0.0)

        # Compute dew point
        dew_point = air_temp_forecast - depression

        # Enforce constraint: dew_point <= air_temp
        dew_point = np.minimum(dew_point, air_temp_forecast)

        return dew_point
