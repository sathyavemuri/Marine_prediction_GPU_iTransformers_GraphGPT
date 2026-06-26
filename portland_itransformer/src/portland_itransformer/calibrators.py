"""Derived output calibrators: conductivity and wave periods."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import RidgeCV
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class DerivedCalibrators:
    """
    Calibrate derived outputs using training data only.

    Three calibrators:
    1. conductivity_mscm = f(salinity_psu, water_temp_c)
    2. significant_wave_period_s = f(log_Tz, log_Hs)
    3. peak_wave_period_s = f(log_Tz, log_Hs)

    All fit on training data only (no test leakage).
    """

    def __init__(self):
        """Initialize calibrators."""
        self.conductivity_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
        self.sig_wave_period_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
        self.peak_wave_period_model = RidgeCV(alphas=[0.1, 1.0, 10.0])

        self.is_fitted = False
        self.feature_ranges = {}

    def fit(
        self,
        df_train: pd.DataFrame,
    ):
        """
        Fit calibrators on training data only.

        Parameters
        ----------
        df_train : DataFrame
            Training data (must include raw features for calibration)
        """
        logger.info("Fitting derived output calibrators on training data...")

        # 1. Conductivity calibrator
        # Features: [salinity_psu, water_temp_c]
        # Target: conductivity_mscm
        if 'salinity_psu' in df_train.columns and \
           'water_temp_c' in df_train.columns and \
           'conductivity_mscm' in df_train.columns:

            X_cond = df_train[['salinity_psu', 'water_temp_c']].values
            y_cond = df_train['conductivity_mscm'].values

            # Remove NaNs
            mask = ~(np.isnan(X_cond).any(axis=1) | np.isnan(y_cond))
            X_cond = X_cond[mask]
            y_cond = y_cond[mask]

            if len(X_cond) > 10:
                self.conductivity_model.fit(X_cond, y_cond)
                logger.info(f"  ✓ Conductivity calibrator fitted (alpha={self.conductivity_model.alpha_:.4f})")
                logger.info(f"    Samples: {len(X_cond)}, R²: {self.conductivity_model.score(X_cond, y_cond):.4f}")
            else:
                logger.warning("  ⚠ Not enough samples for conductivity calibrator")

        # 2. Significant wave period calibrator
        # Features: [log_zero_crossing_period_s, log_significant_wave_height_m]
        # Target: log_significant_wave_period_s
        if 'log_zero_crossing_period_s' in df_train.columns and \
           'log_significant_wave_height_m' in df_train.columns and \
           'significant_wave_period_s' in df_train.columns:

            X_sig = df_train[
                ['log_zero_crossing_period_s', 'log_significant_wave_height_m']
            ].values
            y_sig = df_train['significant_wave_period_s'].values

            # Log transform target
            EPS = 1e-4
            y_sig_log = np.log(y_sig + EPS)

            # Remove NaNs/infs
            mask = ~(np.isnan(X_sig).any(axis=1) | np.isnan(y_sig_log) | np.isinf(y_sig_log))
            X_sig = X_sig[mask]
            y_sig_log = y_sig_log[mask]

            if len(X_sig) > 10:
                self.sig_wave_period_model.fit(X_sig, y_sig_log)
                logger.info(f"  ✓ Sig wave period calibrator fitted (alpha={self.sig_wave_period_model.alpha_:.4f})")
                logger.info(f"    Samples: {len(X_sig)}, R²: {self.sig_wave_period_model.score(X_sig, y_sig_log):.4f}")
            else:
                logger.warning("  ⚠ Not enough samples for sig wave period calibrator")

        # 3. Peak wave period calibrator
        # Features: [log_zero_crossing_period_s, log_significant_wave_height_m]
        # Target: log_peak_wave_period_s
        if 'log_zero_crossing_period_s' in df_train.columns and \
           'log_significant_wave_height_m' in df_train.columns and \
           'peak_wave_period_s' in df_train.columns:

            X_peak = df_train[
                ['log_zero_crossing_period_s', 'log_significant_wave_height_m']
            ].values
            y_peak = df_train['peak_wave_period_s'].values

            # Log transform target
            EPS = 1e-4
            y_peak_log = np.log(y_peak + EPS)

            # Remove NaNs/infs
            mask = ~(np.isnan(X_peak).any(axis=1) | np.isnan(y_peak_log) | np.isinf(y_peak_log))
            X_peak = X_peak[mask]
            y_peak_log = y_peak_log[mask]

            if len(X_peak) > 10:
                self.peak_wave_period_model.fit(X_peak, y_peak_log)
                logger.info(f"  ✓ Peak wave period calibrator fitted (alpha={self.peak_wave_period_model.alpha_:.4f})")
                logger.info(f"    Samples: {len(X_peak)}, R²: {self.peak_wave_period_model.score(X_peak, y_peak_log):.4f}")
            else:
                logger.warning("  ⚠ Not enough samples for peak wave period calibrator")

        self.is_fitted = True
        logger.info("✓ All calibrators fitted")

    def predict_conductivity(
        self,
        salinity: np.ndarray,
        water_temp: np.ndarray,
    ) -> np.ndarray:
        """
        Predict conductivity from salinity and water temperature.

        Parameters
        ----------
        salinity : array
            Salinity in PSU
        water_temp : array
            Water temperature in °C

        Returns
        -------
        conductivity : array
            Conductivity in mS/cm
        """
        if not self.is_fitted:
            logger.warning("Calibrators not fitted, returning zeros")
            return np.zeros_like(salinity)

        X = np.column_stack([salinity, water_temp])
        return self.conductivity_model.predict(X)

    def predict_significant_wave_period(
        self,
        log_zero_crossing_period: np.ndarray,
        log_wave_height: np.ndarray,
    ) -> np.ndarray:
        """
        Predict significant wave period from log Tz and log Hs.

        Parameters
        ----------
        log_zero_crossing_period : array
            log(zero_crossing_period_s)
        log_wave_height : array
            log(significant_wave_height_m)

        Returns
        -------
        significant_wave_period : array
            Significant wave period in seconds
        """
        if not self.is_fitted:
            logger.warning("Calibrators not fitted, returning zeros")
            return np.zeros_like(log_zero_crossing_period)

        X = np.column_stack([log_zero_crossing_period, log_wave_height])
        log_period = self.sig_wave_period_model.predict(X)

        # Inverse log transform
        EPS = 1e-4
        period = np.exp(log_period) - EPS
        return np.maximum(period, 0.0)

    def predict_peak_wave_period(
        self,
        log_zero_crossing_period: np.ndarray,
        log_wave_height: np.ndarray,
    ) -> np.ndarray:
        """
        Predict peak wave period from log Tz and log Hs.

        Parameters
        ----------
        log_zero_crossing_period : array
            log(zero_crossing_period_s)
        log_wave_height : array
            log(significant_wave_height_m)

        Returns
        -------
        peak_wave_period : array
            Peak wave period in seconds
        """
        if not self.is_fitted:
            logger.warning("Calibrators not fitted, returning zeros")
            return np.zeros_like(log_zero_crossing_period)

        X = np.column_stack([log_zero_crossing_period, log_wave_height])
        log_period = self.peak_wave_period_model.predict(X)

        # Inverse log transform
        EPS = 1e-4
        period = np.exp(log_period) - EPS
        return np.maximum(period, 0.0)

    def save(self, artifacts_dir: Path):
        """
        Save calibrators to disk.

        Parameters
        ----------
        artifacts_dir : Path
            Directory to save artifacts
        """
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        path = artifacts_dir / 'derived_calibrators.joblib'
        joblib.dump({
            'conductivity': self.conductivity_model,
            'sig_wave_period': self.sig_wave_period_model,
            'peak_wave_period': self.peak_wave_period_model,
            'is_fitted': self.is_fitted,
        }, path)

        logger.info(f"Saved derived calibrators: {path}")

    def load(self, artifacts_dir: Path):
        """
        Load calibrators from disk.

        Parameters
        ----------
        artifacts_dir : Path
            Directory containing saved artifacts
        """
        artifacts_dir = Path(artifacts_dir)
        path = artifacts_dir / 'derived_calibrators.joblib'

        if path.exists():
            data = joblib.load(path)
            self.conductivity_model = data.get('conductivity', self.conductivity_model)
            self.sig_wave_period_model = data.get('sig_wave_period', self.sig_wave_period_model)
            self.peak_wave_period_model = data.get('peak_wave_period', self.peak_wave_period_model)
            self.is_fitted = data.get('is_fitted', False)
            logger.info(f"Loaded derived calibrators: {path}")
        else:
            logger.warning(f"Calibrators not found: {path}")
