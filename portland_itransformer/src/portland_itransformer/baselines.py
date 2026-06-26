"""Deterministic baselines: UTide, pvlib, seasonal."""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class UTideBaseline:
    """Harmonic tide model using UTide."""

    def __init__(self, config):
        """Initialize UTide baseline."""
        self.config = config
        self.coef = None

    def fit(self, timestamps: pd.DatetimeIndex, tidal_level_m: np.ndarray):
        """
        Fit UTide model using first 60 days only.

        Parameters
        ----------
        timestamps : DatetimeIndex
            UTC timestamps
        tidal_level_m : array
            Observed tidal level in meters
        """
        try:
            import utide
        except ImportError:
            raise ImportError("utide not installed. pip install utide")

        # Extract first 60 days for fitting
        baseline_fit_days = self.config.data.baseline_fit_days
        baseline_end = timestamps[0] + pd.Timedelta(days=baseline_fit_days)

        mask = timestamps <= baseline_end
        fit_timestamps = timestamps[mask]
        fit_levels = tidal_level_m[mask]

        if len(fit_timestamps) < 100:
            logger.warning("Less than 100 samples for UTide fit")
            return

        # Convert timestamps to matplotlib days format
        if isinstance(fit_timestamps, np.ndarray):
            # Already numpy array
            fit_ts_index = pd.DatetimeIndex(fit_timestamps)
        else:
            # Pandas Series/Index
            fit_ts_index = fit_timestamps

        time_utc = fit_ts_index.values.astype('datetime64[D]').astype(float)
        # Add fractional day from time of day
        time_utc = time_utc + fit_ts_index.hour / 24.0 + fit_ts_index.minute / 1440.0

        logger.info(f"Fitting UTide on {len(fit_timestamps)} samples")

        self.coef = utide.solve(
            time_utc,
            fit_levels,
            lat=self.config.site.latitude,
            method="ols",
            trend=False,
        )

        logger.info("✓ UTide model fitted")

    def reconstruct(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Reconstruct tide baseline at given timestamps.

        Parameters
        ----------
        timestamps : DatetimeIndex
            UTC timestamps

        Returns
        -------
        tide_baseline_m : array
            Reconstructed harmonic tide in meters
        """
        if self.coef is None:
            logger.warning("UTide model not fitted, returning zeros")
            return np.zeros(len(timestamps))

        try:
            import utide
        except ImportError:
            raise ImportError("utide not installed")

        # Convert to matplotlib days
        if isinstance(timestamps, np.ndarray):
            ts_index = pd.DatetimeIndex(timestamps)
        else:
            ts_index = timestamps

        time_utc = ts_index.values.astype('datetime64[D]').astype(float)
        time_utc = time_utc + ts_index.hour / 24.0 + ts_index.minute / 1440.0

        tide_pred = utide.reconstruct(time_utc, self.coef)
        return tide_pred.h

    def save(self, artifacts_dir: Path):
        """Save UTide coefficients."""
        if self.coef is None:
            return

        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        path = artifacts_dir / 'utide_coefficients.pkl'
        joblib.dump(self.coef, path)
        logger.info(f"Saved UTide model: {path}")

    def load(self, artifacts_dir: Path):
        """Load UTide coefficients."""
        artifacts_dir = Path(artifacts_dir)
        path = artifacts_dir / 'utide_coefficients.pkl'

        if path.exists():
            self.coef = joblib.load(path)
            logger.info(f"Loaded UTide model: {path}")
        else:
            logger.warning(f"UTide model not found: {path}")


class ClearSkyBaseline:
    """Clear-sky solar radiation using pvlib."""

    def __init__(self, config):
        """Initialize clear-sky baseline."""
        self.config = config
        self.location = None

    def fit(self, timestamps: pd.DatetimeIndex):
        """
        Initialize location for clear-sky calculations.

        Parameters
        ----------
        timestamps : DatetimeIndex
            UTC timestamps (used for validation only)
        """
        try:
            from pvlib import location
        except ImportError:
            raise ImportError("pvlib not installed. pip install pvlib")

        self.location = location.Location(
            latitude=self.config.site.latitude,
            longitude=self.config.site.longitude,
            tz=self.config.site.timezone,
        )

        logger.info(f"✓ Clear-sky location initialized: "
                   f"lat={self.config.site.latitude}, lon={self.config.site.longitude}")

    def get_clear_sky(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Get clear-sky GHI for given timestamps.

        Parameters
        ----------
        timestamps : DatetimeIndex
            UTC timestamps

        Returns
        -------
        clear_sky_wm2 : array
            Clear-sky global horizontal irradiance in W/m²
        """
        if self.location is None:
            logger.warning("Clear-sky location not initialized")
            return np.zeros(len(timestamps))

        # Ensure timestamps is a DatetimeIndex
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex(timestamps)

        # Get clear-sky model
        clearsky = self.location.get_clearsky(
            times=timestamps,
            model='ineichen',
        )

        return clearsky['ghi'].values

    def save(self, artifacts_dir: Path):
        """Save metadata (location is deterministic, no need to persist)."""
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            'latitude': self.config.site.latitude,
            'longitude': self.config.site.longitude,
            'timezone': self.config.site.timezone,
        }

        path = artifacts_dir / 'clearsky_metadata.json'
        import json
        with open(path, 'w') as f:
            json.dump(metadata, f)
        logger.info(f"Saved clear-sky metadata: {path}")


class HarmonicAnomalyBaseline:
    """Fit daily + annual harmonic baseline on training data only."""

    def __init__(self):
        """Initialize."""
        self.coeffs = None  # [beta0, beta1, beta2, beta3, beta4]

    def fit(self, timestamps: pd.DatetimeIndex, values: np.ndarray, train_mask: np.ndarray):
        """
        Fit 4-harmonic model on training data only.

        Parameters
        ----------
        timestamps : DatetimeIndex
            UTC timestamps
        values : array
            [num_samples] target values
        train_mask : array
            Boolean mask for training rows only
        """
        if isinstance(timestamps, np.ndarray):
            timestamps = pd.DatetimeIndex(timestamps)

        # Extract only training data
        train_ts = timestamps[train_mask]
        train_vals = values[train_mask]

        # Compute harmonic features
        hour_frac = train_ts.hour / 24.0 + train_ts.minute / 1440.0
        day_of_year = train_ts.dayofyear

        X = np.column_stack([
            np.ones(len(train_ts)),
            np.sin(2 * np.pi * hour_frac),
            np.cos(2 * np.pi * hour_frac),
            np.sin(2 * np.pi * day_of_year / 365.25),
            np.cos(2 * np.pi * day_of_year / 365.25),
        ])

        # Fit via OLS
        self.coeffs = np.linalg.lstsq(X, train_vals, rcond=None)[0]
        logger.info(f"Fitted harmonic baseline with coefficients: {np.round(self.coeffs, 4)}")

    def predict(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Get baseline for any timestamp.

        Parameters
        ----------
        timestamps : DatetimeIndex
            UTC timestamps

        Returns
        -------
        baseline : array
            [len(timestamps)] baseline values
        """
        if self.coeffs is None:
            logger.warning("Harmonic baseline not fitted, returning zeros")
            return np.zeros(len(timestamps))

        if isinstance(timestamps, np.ndarray):
            timestamps = pd.DatetimeIndex(timestamps)

        hour_frac = timestamps.hour / 24.0 + timestamps.minute / 1440.0
        day_of_year = timestamps.dayofyear

        X = np.column_stack([
            np.ones(len(timestamps)),
            np.sin(2 * np.pi * hour_frac),
            np.cos(2 * np.pi * hour_frac),
            np.sin(2 * np.pi * day_of_year / 365.25),
            np.cos(2 * np.pi * day_of_year / 365.25),
        ])

        return X @ self.coeffs

    def save(self, artifacts_dir: Path):
        """Save harmonic coefficients."""
        if self.coeffs is None:
            return
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = artifacts_dir / 'harmonic_baseline.joblib'
        joblib.dump(self.coeffs, path)
        logger.info(f"Saved harmonic baseline: {path}")

    def load(self, artifacts_dir: Path):
        """Load harmonic coefficients."""
        artifacts_dir = Path(artifacts_dir)
        path = artifacts_dir / 'harmonic_baseline.joblib'
        if path.exists():
            self.coeffs = joblib.load(path)
            logger.info(f"Loaded harmonic baseline: {path}")
        else:
            logger.warning(f"Harmonic baseline not found: {path}")


class DailySeasonalBaseline:
    """Daily seasonal baseline: same time 1 day ago."""

    def __init__(self):
        """Initialize."""
        pass

    def forecast(
        self,
        df_past: pd.DataFrame,
        timestamps_future: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Get seasonal forecast (shift by 1 day).

        Parameters
        ----------
        df_past : DataFrame
            Past observations (with timestamp index)
        timestamps_future : DatetimeIndex
            Future timestamps to forecast

        Returns
        -------
        forecast : array
            [len(timestamps_future), num_features]
        """
        # Look for same time, one day ago
        shift_timestamps = timestamps_future - pd.Timedelta(days=1)

        forecast_list = []
        for ts in shift_timestamps:
            if ts in df_past.index:
                forecast_list.append(df_past.loc[ts].values)
            else:
                # No match, use persistence from end of past data
                forecast_list.append(df_past.iloc[-1].values)

        return np.array(forecast_list)


def get_baseline_forecasts(
    df_train: pd.DataFrame,
    timestamps_test: pd.DatetimeIndex,
    mode: str = "persistence"
) -> np.ndarray:
    """
    Generate baseline forecast.

    Parameters
    ----------
    df_train : DataFrame
        Training data (for seasonal offset)
    timestamps_test : DatetimeIndex
        Test timestamps
    mode : str
        'persistence' or 'seasonal'

    Returns
    -------
    forecast : array
        [len(timestamps_test), num_features]
    """
    if mode == "persistence":
        # Last value repeated
        last_vals = df_train.iloc[-1].values
        return np.tile(last_vals, (len(timestamps_test), 1))

    elif mode == "seasonal":
        # Same time 1 day ago
        baseline = DailySeasonalBaseline()
        df_past = df_train.set_index('timestamp')
        return baseline.forecast(df_past, timestamps_test)

    else:
        raise ValueError(f"Unknown baseline mode: {mode}")
