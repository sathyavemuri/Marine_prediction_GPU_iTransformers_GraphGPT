"""Atmospheric TimeXer preprocessing: anomaly-based 5-target model."""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
import joblib

from .validate import DataValidator
from .features import speed_dir_to_uv
from .baselines import HarmonicAnomalyBaseline

logger = logging.getLogger(__name__)

# 5 Atmospheric targets (anomaly-based)
TARGET_ATMOSPHERE_TIMEXER = [
    'air_temp_anomaly_c',
    'log1p_dewpoint_depression_c',
    'air_pressure_anomaly_hpa',
    'wind_u_anomaly_ms',
    'wind_v_anomaly_ms',
]

KNOWN_FEATURES_ATMOSPHERE = [
    'hour_sin',
    'hour_cos',
    'dayofyear_sin',
    'dayofyear_cos',
]


class AtmosphereTimeXerPreprocessor:
    """Atmospheric anomaly-based preprocessor for TimeXer or compact iTransformer."""

    def __init__(self, config):
        self.config = config
        self.df_raw = None
        self.harmonic_temp = None
        self.harmonic_pressure = None
        self.wind_climatology = None
        self.scaler_targets = None
        self.scaler_known = None

    def preprocess(self):
        """Run full preprocessing pipeline."""
        logger.info("=" * 80)
        logger.info("ATMOSPHERIC TIMEXER PREPROCESSING PIPELINE START")
        logger.info("=" * 80)

        # 1. Validation
        logger.info("\n1. VALIDATION")
        validator = DataValidator(self.config)
        validator.validate()

        # Load raw data
        self.df_raw = pd.read_csv(self.config.paths.raw_csv)
        if self.df_raw.columns[0].startswith("Unnamed"):
            self.df_raw = self.df_raw.iloc[:, 1:]
        self.df_raw['timestamp'] = pd.to_datetime(self.df_raw['timestamp'], utc=True)
        self.df_raw = self.df_raw.sort_values('timestamp').reset_index(drop=True)
        logger.info(f"✓ Loaded {len(self.df_raw)} rows from {self.config.paths.raw_csv}")

        # 2. Direction transform (wind only)
        logger.info("\n2. DIRECTION TRANSFORMS (Atmosphere)")
        self._transform_wind_direction()

        # 3. Chronological split
        logger.info("\n3. CHRONOLOGICAL SPLIT")
        split_labels = self._chronological_split()

        # 4. Harmonic baselines
        logger.info("\n4. HARMONIC BASELINES (Train-Only Fit)")
        self._fit_harmonic_baselines(split_labels)

        # 5. Wind climatology
        logger.info("\n5. WIND CLIMATOLOGY (Train-Only Fit)")
        self._fit_wind_climatology(split_labels)

        # 6. Cyclical features
        logger.info("\n6. CYCLICAL FEATURES")
        self._create_cyclical_features()

        # 7. Derive anomalies
        logger.info("\n7. ATMOSPHERE ANOMALY TARGETS")
        self._derive_anomaly_targets()

        # 8. Scaling
        logger.info("\n8. SCALING (TRAIN-ONLY FIT)")
        self._fit_scalers(split_labels)

        # 9. Save
        logger.info("\n9. SAVING ARTIFACTS")
        self._save_artifacts(split_labels)

        logger.info("\n" + "=" * 80)
        logger.info("ATMOSPHERIC TIMEXER PREPROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  ✓ Processed: {len(self.df_raw)} rows")
        logger.info(f"  ✓ Targets: {len(TARGET_ATMOSPHERE_TIMEXER)} (anomaly-based)")
        logger.info(f"  ✓ Split: {(split_labels == 0).sum()} train, "
                    f"{(split_labels == 1).sum()} val, {(split_labels == 2).sum()} test")

        return {
            'num_samples': len(self.df_raw),
            'split_counts': {
                'train': (split_labels == 0).sum(),
                'validation': (split_labels == 1).sum(),
                'test': (split_labels == 2).sum(),
            }
        }

    def _transform_wind_direction(self):
        """Convert wind direction to u/v components."""
        # Meteorological "from" direction
        self.df_raw['wind_u_east_ms'], self.df_raw['wind_v_north_ms'] = speed_dir_to_uv(
            self.df_raw['wind_speed_ms'].values,
            self.df_raw['wind_direction_deg'].values,
            convention='from'
        )
        logger.info("✓ Wind converted to u/v components")

    def _chronological_split(self):
        """Create train/val/test split."""
        n = len(self.df_raw)
        train_end = int(0.5 * n)
        val_end = int(0.75 * n)

        split_labels = np.zeros(n, dtype=int)
        split_labels[train_end:val_end] = 1
        split_labels[val_end:] = 2

        logger.info(f"  Train: {train_end} rows")
        logger.info(f"  Valid: {val_end - train_end} rows")
        logger.info(f"  Test:  {n - val_end} rows")

        np.save(Path(self.config.paths.processed_dir) / "atmosphere_timexer_split_labels.npy", split_labels)
        return split_labels

    def _fit_harmonic_baselines(self, split_labels):
        """Fit harmonic baselines on training data only."""
        train_mask = split_labels == 0

        # Temperature harmonic baseline
        self.harmonic_temp = HarmonicAnomalyBaseline()
        self.harmonic_temp.fit(
            pd.DatetimeIndex(self.df_raw['timestamp']),
            self.df_raw['air_temp_c'].values,
            train_mask
        )

        # Pressure harmonic baseline
        self.harmonic_pressure = HarmonicAnomalyBaseline()
        self.harmonic_pressure.fit(
            pd.DatetimeIndex(self.df_raw['timestamp']),
            self.df_raw['air_pressure_hpa'].values,
            train_mask
        )

        logger.info("✓ Harmonic baselines fitted (temp, pressure)")

    def _fit_wind_climatology(self, split_labels):
        """Fit wind climatology (hourly means) on training data."""
        train_mask = split_labels == 0

        self.wind_climatology = {}
        for hour in range(24):
            hour_mask = (pd.DatetimeIndex(self.df_raw['timestamp']).hour == hour) & train_mask
            if hour_mask.sum() > 0:
                self.wind_climatology[hour] = {
                    'u': self.df_raw.loc[hour_mask, 'wind_u_east_ms'].mean(),
                    'v': self.df_raw.loc[hour_mask, 'wind_v_north_ms'].mean(),
                }

        logger.info("✓ Wind climatology fitted (hourly means)")

    def _create_cyclical_features(self):
        """Create hourly and daily cyclical features."""
        timestamps = pd.DatetimeIndex(self.df_raw['timestamp'])

        # Hourly cycle
        hour = timestamps.hour + timestamps.minute / 60.0
        self.df_raw['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        self.df_raw['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)

        # Daily cycle (day of year)
        day_of_year = timestamps.dayofyear
        self.df_raw['dayofyear_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        self.df_raw['dayofyear_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

        logger.info("✓ Cyclical features created")

    def _derive_anomaly_targets(self):
        """Derive 5 anomaly-based targets."""
        timestamps = pd.DatetimeIndex(self.df_raw['timestamp'])

        # Temperature anomaly
        temp_baseline = self.harmonic_temp.predict(timestamps)
        self.df_raw['air_temp_anomaly_c'] = self.df_raw['air_temp_c'] - temp_baseline

        # Dew point depression (anomaly-based)
        self.df_raw['log1p_dewpoint_depression_c'] = np.log1p(
            np.maximum(self.df_raw['air_temp_c'] - self.df_raw['dew_point_c'], 0.01)
        )

        # Pressure anomaly
        pressure_baseline = self.harmonic_pressure.predict(timestamps)
        self.df_raw['air_pressure_anomaly_hpa'] = self.df_raw['air_pressure_hpa'] - pressure_baseline

        # Wind anomalies (residuals from climatology)
        wind_u_clim = np.array([
            self.wind_climatology.get(h, {}).get('u', 0.0)
            for h in timestamps.hour
        ])
        wind_v_clim = np.array([
            self.wind_climatology.get(h, {}).get('v', 0.0)
            for h in timestamps.hour
        ])
        self.df_raw['wind_u_anomaly_ms'] = self.df_raw['wind_u_east_ms'] - wind_u_clim
        self.df_raw['wind_v_anomaly_ms'] = self.df_raw['wind_v_north_ms'] - wind_v_clim

        logger.info("✓ Atmosphere anomaly targets derived (5 targets)")

    def _fit_scalers(self, split_labels):
        """Fit scalers on training data only."""
        train_mask = split_labels == 0

        # Target scaler
        target_data = self.df_raw.loc[train_mask, TARGET_ATMOSPHERE_TIMEXER].values
        self.scaler_targets = StandardScaler()
        self.scaler_targets.fit(target_data)

        # Known features scaler
        known_data = self.df_raw.loc[train_mask, KNOWN_FEATURES_ATMOSPHERE].values
        self.scaler_known = StandardScaler()
        self.scaler_known.fit(known_data)

        # Scale all data
        self.df_raw[TARGET_ATMOSPHERE_TIMEXER] = self.scaler_targets.transform(
            self.df_raw[TARGET_ATMOSPHERE_TIMEXER].values
        )
        self.df_raw[KNOWN_FEATURES_ATMOSPHERE] = self.scaler_known.transform(
            self.df_raw[KNOWN_FEATURES_ATMOSPHERE].values
        )

        logger.info(f"✓ Target scaler fit on {train_mask.sum()} training samples")
        logger.info(f"✓ Known scaler fit on {train_mask.sum()} training samples")

    def _save_artifacts(self, split_labels):
        """Save processed data and artifacts."""
        processed_dir = Path(self.config.paths.processed_dir)
        artifacts_dir = Path(self.config.paths.artifacts_dir)

        processed_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save processed data
        self.df_raw.to_parquet(processed_dir / "atmosphere_timexer_preprocessed.parquet")
        np.save(processed_dir / "atmosphere_timexer_split_labels.npy", split_labels)

        # Save scalers
        joblib.dump(self.scaler_targets, artifacts_dir / "atmosphere_timexer_scaler_targets.joblib")
        joblib.dump(self.scaler_known, artifacts_dir / "atmosphere_timexer_scaler_known.joblib")

        # Save baselines
        joblib.dump(self.harmonic_temp, artifacts_dir / "atmosphere_timexer_harmonic_temp.joblib")
        joblib.dump(self.harmonic_pressure, artifacts_dir / "atmosphere_timexer_harmonic_pressure.joblib")
        joblib.dump(self.wind_climatology, artifacts_dir / "atmosphere_timexer_wind_climatology.joblib")

        logger.info(f"  Saved: {processed_dir / 'atmosphere_timexer_preprocessed.parquet'}")
        logger.info(f"  Saved: {processed_dir / 'atmosphere_timexer_split_labels.npy'}")
        logger.info(f"  Saved scalers and baselines to {artifacts_dir}")
