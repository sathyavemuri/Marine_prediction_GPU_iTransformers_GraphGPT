"""Water Temperature anomaly-based preprocessing for TiDE/NHITS/TCN models."""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
import joblib

from .validate import DataValidator
from .baselines import HarmonicAnomalyBaseline

logger = logging.getLogger(__name__)

# 1 Water temperature target (anomaly-based)
TARGET_WATER_TEMP = ['water_temp_anomaly_c']

# Multi-variate inputs: water temp history + exogenous features
EXOGENOUS_FEATURES_WATER_TEMP = [
    'air_temp_c',
    'tidal_residual_m',
    'current_u_east_ms',
    'current_v_north_ms',
    'salinity_psu',
    'log1p_global_radiation_wm2',
]

KNOWN_FEATURES_WATER_TEMP = [
    'hour_sin',
    'hour_cos',
    'dayofyear_sin',
    'dayofyear_cos',
]


class WaterTempPreprocessor:
    """Water temperature anomaly-based preprocessor for TiDE/NHITS/TCN."""

    def __init__(self, config):
        self.config = config
        self.df_raw = None
        self.harmonic_temp = None
        self.scaler_target = None
        self.scaler_exogenous = None
        self.scaler_known = None

    def preprocess(self):
        """Run full preprocessing pipeline."""
        logger.info("=" * 80)
        logger.info("WATER TEMPERATURE ANOMALY PREPROCESSING PIPELINE START")
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

        # 2. Chronological split
        logger.info("\n2. CHRONOLOGICAL SPLIT")
        split_labels = self._chronological_split()

        # 3. Water temperature harmonic baseline
        logger.info("\n3. WATER TEMPERATURE HARMONIC BASELINE (Train-Only Fit)")
        self._fit_harmonic_baseline(split_labels)

        # 4. Cyclical features
        logger.info("\n4. CYCLICAL FEATURES")
        self._create_cyclical_features()

        # 5. Derive water temperature anomaly
        logger.info("\n5. WATER TEMPERATURE ANOMALY TARGET")
        self._derive_water_temp_anomaly()

        # 6. Prepare marine features (for exogenous use)
        logger.info("\n6. PREPARE MARINE EXOGENOUS FEATURES")
        self._prepare_exogenous_features(split_labels)

        # 7. Scaling
        logger.info("\n7. SCALING (TRAIN-ONLY FIT)")
        self._fit_scalers(split_labels)

        # 8. Save
        logger.info("\n8. SAVING ARTIFACTS")
        self._save_artifacts(split_labels)

        logger.info("\n" + "=" * 80)
        logger.info("WATER TEMPERATURE PREPROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  ✓ Processed: {len(self.df_raw)} rows")
        logger.info(f"  ✓ Target: water_temp_anomaly_c (1 variable)")
        logger.info(f"  ✓ Exogenous: {len(EXOGENOUS_FEATURES_WATER_TEMP)} marine features")
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

        np.save(Path(self.config.paths.processed_dir) / "water_temp_split_labels.npy", split_labels)
        return split_labels

    def _fit_harmonic_baseline(self, split_labels):
        """Fit water temperature harmonic baseline on training data."""
        from .baselines import HarmonicAnomalyBaseline

        train_mask = split_labels == 0

        self.harmonic_temp = HarmonicAnomalyBaseline()
        self.harmonic_temp.fit(
            pd.DatetimeIndex(self.df_raw['timestamp']),
            self.df_raw['water_temp_c'].values,
            train_mask
        )

        logger.info("✓ Water temperature harmonic baseline fitted")

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

    def _derive_water_temp_anomaly(self):
        """Derive water temperature anomaly target."""
        timestamps = pd.DatetimeIndex(self.df_raw['timestamp'])

        temp_baseline = self.harmonic_temp.predict(timestamps)
        self.df_raw['water_temp_anomaly_c'] = self.df_raw['water_temp_c'] - temp_baseline

        logger.info("✓ Water temperature anomaly derived")

    def _prepare_exogenous_features(self, split_labels):
        """Prepare exogenous marine features (simplified)."""
        # These come from marine model predictions or raw data
        # For training: use raw data; for inference: use marine model outputs

        # Log transform radiation
        self.df_raw['log1p_global_radiation_wm2'] = np.log1p(
            np.maximum(self.df_raw['global_radiation_wm2'], 0.0)
        )

        # Convert current direction to u/v (if needed)
        from .features import speed_dir_to_uv
        self.df_raw['current_u_east_ms'], self.df_raw['current_v_north_ms'] = speed_dir_to_uv(
            self.df_raw['current_speed_ms'].values,
            self.df_raw['current_direction_deg'].values,
            convention='to'
        )

        logger.info("✓ Exogenous marine features prepared")

    def _fit_scalers(self, split_labels):
        """Fit scalers on training data only."""
        train_mask = split_labels == 0

        # Target scaler (water temp anomaly)
        target_data = self.df_raw.loc[train_mask, TARGET_WATER_TEMP].values
        self.scaler_target = StandardScaler()
        self.scaler_target.fit(target_data)

        # Exogenous scaler (marine features)
        exogenous_data = self.df_raw.loc[train_mask, EXOGENOUS_FEATURES_WATER_TEMP].values
        self.scaler_exogenous = StandardScaler()
        self.scaler_exogenous.fit(exogenous_data)

        # Known features scaler (calendar)
        known_data = self.df_raw.loc[train_mask, KNOWN_FEATURES_WATER_TEMP].values
        self.scaler_known = StandardScaler()
        self.scaler_known.fit(known_data)

        # Scale all data
        self.df_raw[TARGET_WATER_TEMP] = self.scaler_target.transform(
            self.df_raw[TARGET_WATER_TEMP].values
        )
        self.df_raw[EXOGENOUS_FEATURES_WATER_TEMP] = self.scaler_exogenous.transform(
            self.df_raw[EXOGENOUS_FEATURES_WATER_TEMP].values
        )
        self.df_raw[KNOWN_FEATURES_WATER_TEMP] = self.scaler_known.transform(
            self.df_raw[KNOWN_FEATURES_WATER_TEMP].values
        )

        logger.info(f"✓ Target scaler fit on {train_mask.sum()} training samples")
        logger.info(f"✓ Exogenous scaler fit on {train_mask.sum()} training samples")
        logger.info(f"✓ Known scaler fit on {train_mask.sum()} training samples")

    def _save_artifacts(self, split_labels):
        """Save processed data and artifacts."""
        processed_dir = Path(self.config.paths.processed_dir)
        artifacts_dir = Path(self.config.paths.artifacts_dir)

        processed_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save processed data
        self.df_raw.to_parquet(processed_dir / "water_temp_preprocessed.parquet")
        np.save(processed_dir / "water_temp_split_labels.npy", split_labels)

        # Save scalers
        joblib.dump(self.scaler_target, artifacts_dir / "water_temp_scaler_target.joblib")
        joblib.dump(self.scaler_exogenous, artifacts_dir / "water_temp_scaler_exogenous.joblib")
        joblib.dump(self.scaler_known, artifacts_dir / "water_temp_scaler_known.joblib")

        # Save baseline
        joblib.dump(self.harmonic_temp, artifacts_dir / "water_temp_harmonic_baseline.joblib")

        logger.info(f"  Saved: {processed_dir / 'water_temp_preprocessed.parquet'}")
        logger.info(f"  Saved: {processed_dir / 'water_temp_split_labels.npy'}")
        logger.info(f"  Saved scalers and baseline to {artifacts_dir}")
