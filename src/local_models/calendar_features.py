"""Calendar and cyclical feature engineering for local statistical models."""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CalendarFeatures:
    """Create cyclical features from timestamps."""

    @staticmethod
    def create_features(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Create calendar and cyclical features.

        Parameters
        ----------
        timestamps : pd.DatetimeIndex
            Timestamps to extract features from

        Returns
        -------
        pd.DataFrame
            Features: hour_fraction, day_of_year, sin_hour, cos_hour, sin_year, cos_year
        """
        df = pd.DataFrame(index=timestamps)

        # Hour fraction
        df['hour'] = timestamps.hour
        df['minute'] = timestamps.minute
        df['hour_fraction'] = df['hour'] + df['minute'] / 60.0

        # Day of year
        df['day_of_year'] = timestamps.dayofyear

        # Cyclical encoding
        df['sin_hour'] = np.sin(2 * np.pi * df['hour_fraction'] / 24.0)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour_fraction'] / 24.0)

        df['sin_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['cos_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

        return df[['hour_fraction', 'day_of_year', 'sin_hour', 'cos_hour', 'sin_year', 'cos_year']]

    @staticmethod
    def create_chronological_split(
        timestamps: pd.DatetimeIndex,
        train_fraction: float = 0.7,
        val_fraction: float = 0.15,
    ) -> np.ndarray:
        """
        Create chronological train/val/test split.

        Parameters
        ----------
        timestamps : pd.DatetimeIndex
        train_fraction : float
            Fraction for training (default 0.7)
        val_fraction : float
            Fraction for validation (default 0.15)
            Test = 1 - train - val

        Returns
        -------
        np.ndarray
            Split labels: 0=train, 1=val, 2=test
        """
        n = len(timestamps)
        train_end = int(n * train_fraction)
        val_end = int(n * (train_fraction + val_fraction))

        split_labels = np.zeros(n, dtype=int)
        split_labels[train_end:val_end] = 1
        split_labels[val_end:] = 2

        logger.info(f"Chronological split: {train_end} train, {val_end - train_end} val, {n - val_end} test")
        return split_labels


class HarmonicBaseline:
    """Fit and predict harmonic baseline using calendar features."""

    def __init__(self):
        self.coefficients = None
        self.fitted = False

    def fit(self, timestamps: pd.DatetimeIndex, values: np.ndarray, train_mask: np.ndarray = None):
        """
        Fit harmonic baseline: beta0 + beta1*sin_hour + beta2*cos_hour + beta3*sin_year + beta4*cos_year.

        Parameters
        ----------
        timestamps : pd.DatetimeIndex
        values : np.ndarray
            Target values
        train_mask : np.ndarray, optional
            Boolean mask for training data only
        """
        cal = CalendarFeatures.create_features(timestamps)

        if train_mask is not None:
            X_train = cal[train_mask].values
            y_train = values[train_mask]
        else:
            X_train = cal.values
            y_train = values

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X_train)), X_train[:, 2:6]])  # sin_hour, cos_hour, sin_year, cos_year

        # Fit OLS: beta = (X^T X)^-1 X^T y
        self.coefficients = np.linalg.lstsq(X_with_intercept, y_train, rcond=None)[0]
        self.fitted = True

        logger.info(f"Harmonic baseline fitted: coefficients {self.coefficients}")

    def predict(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Predict harmonic baseline for given timestamps."""
        if not self.fitted:
            raise ValueError("Baseline not fitted yet")

        cal = CalendarFeatures.create_features(timestamps)
        X_with_intercept = np.column_stack([np.ones(len(cal)), cal[['sin_hour', 'cos_hour', 'sin_year', 'cos_year']].values])

        return X_with_intercept @ self.coefficients
