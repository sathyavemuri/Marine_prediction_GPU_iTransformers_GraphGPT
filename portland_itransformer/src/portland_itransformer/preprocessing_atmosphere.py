"""Preprocessing pipeline for Atmospheric iTransformer (5 targets, anomaly-based)."""

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

from .validate import DataValidator
from .features import speed_dir_to_uv, apply_log_transform
from .baselines import HarmonicAnomalyBaseline

logger = logging.getLogger(__name__)

# Canonical target order (NEVER change this)
TARGET_ATMOSPHERE = [
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


class AtmospherePreprocessor:
    """Atmospheric data pipeline: temp, pressure, wind (anomaly-based)."""

    def __init__(self, config):
        """Initialize."""
        self.config = config
        self.df_raw = None
        self.df_processed = None
        self.harmonic_temp = None
        self.harmonic_pressure = None
        self.wind_u_climatology = None
        self.wind_v_climatology = None
        self.scaler_targets = None
        self.scaler_known = None

    def validate(self):
        """Validate raw data."""
        logger.info("\n1. VALIDATION")
        validator = DataValidator(self.config)
        validator.validate()  # Loads and validates from config.paths.raw_csv

        raw_path = self.config.paths.raw_csv
        self.df_raw = pd.read_csv(raw_path)

        logger.info(f"✓ Loaded {len(self.df_raw)} rows from {raw_path}")

        return self.df_raw

    def transform_directions(self):
        """Convert wind direction to u/v components."""
        logger.info("\n2. DIRECTION TRANSFORMS (Atmosphere)")

        # Wind: meteorological "from" direction
        self.df_raw['wind_u_east_ms'], self.df_raw['wind_v_north_ms'] = speed_dir_to_uv(
            self.df_raw['wind_speed_ms'].values,
            self.df_raw['wind_direction_deg'].values,
            convention='from'
        )

        logger.info("✓ Wind converted to u/v components")

    def create_cyclical_features(self):
        """Create hour and day-of-year cyclical features."""
        logger.info("\n3. CYCLICAL FEATURES")

        # Ensure timestamps are DatetimeIndex
        timestamps = pd.DatetimeIndex(self.df_raw['timestamp'])

        hour = timestamps.hour + timestamps.minute / 60.0
        day_of_year = timestamps.dayofyear

        self.df_raw['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        self.df_raw['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        self.df_raw['dayofyear_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        self.df_raw['dayofyear_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

        logger.info("✓ Cyclical features created")

    def chronological_split(self):
        """Split into train/val/test chronologically."""
        logger.info("\n4. CHRONOLOGICAL SPLIT")

        n = len(self.df_raw)
        train_end = int(0.5 * n)
        val_end = int(0.75 * n)

        split_labels = np.zeros(n, dtype=int)
        split_labels[train_end:val_end] = 1
        split_labels[val_end:] = 2

        logger.info(f"  Train: {train_end} rows")
        logger.info(f"  Valid: {val_end - train_end} rows")
        logger.info(f"  Test:  {n - val_end} rows")

        return split_labels

    def fit_harmonic_baselines(self, train_mask):
        """Fit daily + annual harmonic baselines on training data only."""
        logger.info("\n5. HARMONIC BASELINES (Train-Only Fit)")

        timestamps = pd.DatetimeIndex(self.df_raw['timestamp'])

        # Air temperature baseline
        self.harmonic_temp = HarmonicAnomalyBaseline()
        self.harmonic_temp.fit(
            timestamps,
            self.df_raw['air_temp_c'].values,
            train_mask
        )
        air_temp_baseline = self.harmonic_temp.predict(timestamps)
        self.df_raw['air_temp_anomaly_c'] = self.df_raw['air_temp_c'] - air_temp_baseline

        # Air pressure baseline
        self.harmonic_pressure = HarmonicAnomalyBaseline()
        self.harmonic_pressure.fit(
            timestamps,
            self.df_raw['air_pressure_hpa'].values,
            train_mask
        )
        air_pressure_baseline = self.harmonic_pressure.predict(timestamps)
        self.df_raw['air_pressure_anomaly_hpa'] = (
            self.df_raw['air_pressure_hpa'] - air_pressure_baseline
        )

        logger.info("✓ Harmonic baselines fitted (air_temp, air_pressure)")

    def fit_wind_climatology(self, train_mask):
        """Fit hourly wind climatology on training data only."""
        logger.info("\n6. WIND CLIMATOLOGY (Train-Only Fit)")

        timestamps = pd.DatetimeIndex(self.df_raw['timestamp'])
        train_data = self.df_raw[train_mask].copy()
        train_data['hour'] = pd.DatetimeIndex(train_data['timestamp']).hour

        # Hour-of-day climatology
        self.wind_u_climatology = train_data.groupby('hour')['wind_u_east_ms'].mean().to_dict()
        self.wind_v_climatology = train_data.groupby('hour')['wind_v_north_ms'].mean().to_dict()

        # Apply to all data
        hour_of_day = timestamps.hour
        self.df_raw['wind_u_anomaly_ms'] = (
            self.df_raw['wind_u_east_ms'] - hour_of_day.map(self.wind_u_climatology)
        )
        self.df_raw['wind_v_anomaly_ms'] = (
            self.df_raw['wind_v_north_ms'] - hour_of_day.map(self.wind_v_climatology)
        )

        logger.info("✓ Wind climatology fitted (hourly means)")

    def compute_dewpoint_depression(self):
        """Compute dew point depression as log(air_temp - dew_point)."""
        logger.info("\n7. DEW POINT DEPRESSION")

        depression = np.maximum(
            self.df_raw['air_temp_c'].values - self.df_raw['dew_point_c'].values,
            0.001  # Avoid log(0)
        )
        self.df_raw['log1p_dewpoint_depression_c'] = np.log1p(depression)

        logger.info("✓ Dew point depression computed")

    def fit_scalers(self, train_mask):
        """Fit scalers on training data only."""
        logger.info("\n8. SCALING (TRAIN-ONLY FIT)")

        # Fit target scaler on train data
        train_data = self.df_raw[train_mask][TARGET_ATMOSPHERE]
        self.scaler_targets = StandardScaler()
        self.scaler_targets.fit(train_data)

        logger.info(f"✓ Target scaler fit on {train_mask.sum()} training samples")
        logger.info(f"  Targets: {TARGET_ATMOSPHERE}")

        # Fit known features scaler on train data
        train_known = self.df_raw[train_mask][KNOWN_FEATURES_ATMOSPHERE]
        self.scaler_known = StandardScaler()
        self.scaler_known.fit(train_known)

        logger.info(f"✓ Known scaler fit on {train_mask.sum()} training samples")
        logger.info(f"  Known features: {KNOWN_FEATURES_ATMOSPHERE}")

        # Scale all data
        self.df_raw[TARGET_ATMOSPHERE] = self.scaler_targets.transform(
            self.df_raw[TARGET_ATMOSPHERE]
        )
        self.df_raw[KNOWN_FEATURES_ATMOSPHERE] = self.scaler_known.transform(
            self.df_raw[KNOWN_FEATURES_ATMOSPHERE]
        )

        logger.info(f"✓ Scaled to {len(TARGET_ATMOSPHERE) + len(KNOWN_FEATURES_ATMOSPHERE)} features")

    def save_artifacts(self, split_labels):
        """Save preprocessed data and artifacts."""
        logger.info("\n9. SAVING ARTIFACTS")

        artifacts_dir = Path(self.config.paths.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        processed_dir = Path(self.config.paths.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Save processed data
        self.df_raw.to_parquet(processed_dir / "atmosphere_preprocessed.parquet")
        np.save(processed_dir / "atmosphere_split_labels.npy", split_labels)

        # Save scalers
        joblib.dump(self.scaler_targets, artifacts_dir / "atmosphere_scaler_targets.joblib")
        joblib.dump(self.scaler_known, artifacts_dir / "atmosphere_scaler_known.joblib")

        # Save baselines
        self.harmonic_temp.save(artifacts_dir / "harmonic_temp")
        self.harmonic_pressure.save(artifacts_dir / "harmonic_pressure")

        # Save climatologies
        joblib.dump(self.wind_u_climatology, artifacts_dir / "wind_u_climatology.joblib")
        joblib.dump(self.wind_v_climatology, artifacts_dir / "wind_v_climatology.joblib")

        # Save target order (CRITICAL for inference)
        with open(artifacts_dir / "target_columns_atmosphere.json", "w") as f:
            json.dump(TARGET_ATMOSPHERE, f, indent=2)

        with open(artifacts_dir / "known_columns_atmosphere.json", "w") as f:
            json.dump(KNOWN_FEATURES_ATMOSPHERE, f, indent=2)

        logger.info(f"  Saved: {processed_dir}/atmosphere_preprocessed.parquet")
        logger.info(f"  Saved: {processed_dir}/atmosphere_split_labels.npy")
        logger.info(f"  Saved scalers and baselines to {artifacts_dir}")

    def preprocess(self):
        """Execute full pipeline."""
        logger.info("=" * 80)
        logger.info("ATMOSPHERE PREPROCESSING PIPELINE START")
        logger.info("=" * 80)

        # Step 1: Validate
        self.validate()

        # Step 2: Transform directions
        self.transform_directions()

        # Step 3: Cyclical features
        self.create_cyclical_features()

        # Step 4: Chronological split
        split_labels = self.chronological_split()
        train_mask = split_labels == 0

        # Step 5: Fit anomaly baselines on train data
        self.fit_harmonic_baselines(train_mask)

        # Step 6: Fit wind climatology on train data
        self.fit_wind_climatology(train_mask)

        # Step 7: Dew point depression
        self.compute_dewpoint_depression()

        # Step 8: Scale
        self.fit_scalers(train_mask)

        # Step 9: Save
        self.save_artifacts(split_labels)

        logger.info("\n" + "=" * 80)
        logger.info("ATMOSPHERE PREPROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  ✓ Processed: {len(self.df_raw)} rows × {len(TARGET_ATMOSPHERE) + len(KNOWN_FEATURES_ATMOSPHERE)} features")
        logger.info(f"  ✓ Targets: {len(TARGET_ATMOSPHERE)} (anomaly-based)")
        logger.info(f"  ✓ Known features: {len(KNOWN_FEATURES_ATMOSPHERE)}")

        return {
            'num_samples': len(self.df_raw),
            'num_features': len(TARGET_ATMOSPHERE) + len(KNOWN_FEATURES_ATMOSPHERE),
            'num_targets': len(TARGET_ATMOSPHERE),
            'split_counts': {
                'train': train_mask.sum(),
                'validation': (split_labels == 1).sum(),
                'test': (split_labels == 2).sum(),
            }
        }
