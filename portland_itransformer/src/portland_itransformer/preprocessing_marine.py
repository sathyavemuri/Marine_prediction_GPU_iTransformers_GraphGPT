"""Preprocessing pipeline for Marine iTransformer (8 targets, deterministic)."""

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

from .validate import DataValidator
from .features import speed_dir_to_uv, apply_log_transform
from .baselines import UTideBaseline, ClearSkyBaseline
from .scaling import ScalingPipeline

logger = logging.getLogger(__name__)

# Canonical target order (NEVER change this)
TARGET_MARINE = [
    'tidal_residual_m',
    'current_u_east_ms',
    'current_v_north_ms',
    'salinity_psu',
    'water_temp_c',
    'log1p_global_radiation_wm2',
    'log_significant_wave_height_m',
    'log_zero_crossing_period_s',
]

KNOWN_FEATURES_MARINE = [
    'hour_sin',
    'hour_cos',
    'dayofyear_sin',
    'dayofyear_cos',
]


class MarinePreprocessor:
    """Marine data pipeline: tides, currents, waves, radiation."""

    def __init__(self, config):
        """Initialize."""
        self.config = config
        self.df_raw = None
        self.df_processed = None
        self.utide_baseline = None
        self.clearsky_baseline = None
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
        """Convert wind/current directions to u/v components."""
        logger.info("\n2. DIRECTION TRANSFORMS (Marine)")

        # Wind: meteorological "from" direction
        self.df_raw['wind_u_east_ms'], self.df_raw['wind_v_north_ms'] = speed_dir_to_uv(
            self.df_raw['wind_speed_ms'].values,
            self.df_raw['wind_direction_deg'].values,
            convention='from'
        )

        # Current: oceanographic "to" direction
        self.df_raw['current_u_east_ms'], self.df_raw['current_v_north_ms'] = speed_dir_to_uv(
            self.df_raw['current_speed_ms'].values,
            self.df_raw['current_direction_deg'].values,
            convention='to'
        )

        logger.info("✓ Wind and current converted to u/v components")

    def fit_tidal_baseline(self, train_mask):
        """Fit UTide on training data only."""
        logger.info("\n3. TIDAL BASELINE (UTide, Train-Only Fit)")

        self.utide_baseline = UTideBaseline(self.config)
        self.utide_baseline.fit(
            pd.DatetimeIndex(self.df_raw['timestamp']),
            self.df_raw['tidal_level_m'].values
        )

        # Compute residuals for all data
        tidal_baseline_full = self.utide_baseline.reconstruct(
            pd.DatetimeIndex(self.df_raw['timestamp'])
        )
        self.df_raw['tidal_residual_m'] = self.df_raw['tidal_level_m'] - tidal_baseline_full

        logger.info("✓ Tidal baseline computed and residuals extracted")

    def fit_clearsky_baseline(self, train_mask):
        """Fit pvlib clear-sky radiation."""
        logger.info("\n4. CLEAR-SKY RADIATION (pvlib)")

        self.clearsky_baseline = ClearSkyBaseline(self.config)
        self.clearsky_baseline.fit(pd.DatetimeIndex(self.df_raw['timestamp']))

        clearsky_wm2 = self.clearsky_baseline.get_clear_sky(
            pd.DatetimeIndex(self.df_raw['timestamp'])
        )

        # Global radiation residual = actual - clear_sky
        self.df_raw['global_radiation_residual_wm2'] = (
            self.df_raw['global_radiation_wm2'] - clearsky_wm2
        )

        logger.info("✓ Clear-sky radiation computed")

    def apply_transforms(self):
        """Apply log transforms."""
        logger.info("\n5. LOG TRANSFORMS (Marine)")

        # Log transforms (add small constant for stability)
        self.df_raw['log1p_global_radiation_wm2'] = np.log1p(
            np.maximum(self.df_raw['global_radiation_residual_wm2'], 0.0)
        )

        self.df_raw['log_significant_wave_height_m'] = apply_log_transform(
            self.df_raw['significant_wave_height_m'].values,
            eps=1e-4
        )

        self.df_raw['log_zero_crossing_period_s'] = apply_log_transform(
            self.df_raw['zero_crossing_period_s'].values,
            eps=1e-4
        )

        logger.info("✓ Log transforms applied")

    def create_cyclical_features(self):
        """Create hour and day-of-year cyclical features."""
        logger.info("\n6. CYCLICAL FEATURES")

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
        logger.info("\n7. CHRONOLOGICAL SPLIT")

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

    def fit_scalers(self, train_mask):
        """Fit scalers on training data only."""
        logger.info("\n8. SCALING (TRAIN-ONLY FIT)")

        # Fit target scaler on train data
        train_data = self.df_raw[train_mask][TARGET_MARINE]
        self.scaler_targets = StandardScaler()
        self.scaler_targets.fit(train_data)

        logger.info(f"✓ Target scaler fit on {train_mask.sum()} training samples")
        logger.info(f"  Targets: {TARGET_MARINE}")

        # Fit known features scaler on train data
        train_known = self.df_raw[train_mask][KNOWN_FEATURES_MARINE]
        self.scaler_known = StandardScaler()
        self.scaler_known.fit(train_known)

        logger.info(f"✓ Known scaler fit on {train_mask.sum()} training samples")
        logger.info(f"  Known features: {KNOWN_FEATURES_MARINE}")

        # Scale all data
        self.df_raw[TARGET_MARINE] = self.scaler_targets.transform(self.df_raw[TARGET_MARINE])
        self.df_raw[KNOWN_FEATURES_MARINE] = self.scaler_known.transform(
            self.df_raw[KNOWN_FEATURES_MARINE]
        )

        logger.info(f"✓ Scaled to {len(TARGET_MARINE) + len(KNOWN_FEATURES_MARINE)} features")

    def save_artifacts(self, split_labels):
        """Save preprocessed data and artifacts."""
        logger.info("\n9. SAVING ARTIFACTS")

        artifacts_dir = Path(self.config.paths.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        processed_dir = Path(self.config.paths.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Save processed data
        self.df_raw.to_parquet(processed_dir / "marine_preprocessed.parquet")
        np.save(processed_dir / "marine_split_labels.npy", split_labels)

        # Save scalers
        joblib.dump(self.scaler_targets, artifacts_dir / "marine_scaler_targets.joblib")
        joblib.dump(self.scaler_known, artifacts_dir / "marine_scaler_known.joblib")

        # Save baselines
        self.utide_baseline.save(artifacts_dir)
        self.clearsky_baseline.save(artifacts_dir)

        # Save target order (CRITICAL for inference)
        with open(artifacts_dir / "target_columns_marine.json", "w") as f:
            json.dump(TARGET_MARINE, f, indent=2)

        with open(artifacts_dir / "known_columns_marine.json", "w") as f:
            json.dump(KNOWN_FEATURES_MARINE, f, indent=2)

        logger.info(f"  Saved: {processed_dir}/marine_preprocessed.parquet")
        logger.info(f"  Saved: {processed_dir}/marine_split_labels.npy")
        logger.info(f"  Saved scalers and baselines to {artifacts_dir}")

    def preprocess(self):
        """Execute full pipeline."""
        logger.info("=" * 80)
        logger.info("MARINE PREPROCESSING PIPELINE START")
        logger.info("=" * 80)

        # Step 1: Validate
        self.validate()

        # Step 2: Transform directions
        self.transform_directions()

        # Step 3: Fit baselines on train data
        split_labels = self.chronological_split()
        train_mask = split_labels == 0

        self.fit_tidal_baseline(train_mask)
        self.fit_clearsky_baseline(train_mask)

        # Step 4: Apply transforms
        self.apply_transforms()

        # Step 5: Cyclical features
        self.create_cyclical_features()

        # Step 6: Scale
        self.fit_scalers(train_mask)

        # Step 7: Save
        self.save_artifacts(split_labels)

        logger.info("\n" + "=" * 80)
        logger.info("MARINE PREPROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  ✓ Processed: {len(self.df_raw)} rows × {len(TARGET_MARINE) + len(KNOWN_FEATURES_MARINE)} features")
        logger.info(f"  ✓ Targets: {len(TARGET_MARINE)}")
        logger.info(f"  ✓ Known features: {len(KNOWN_FEATURES_MARINE)}")

        return {
            'num_samples': len(self.df_raw),
            'num_features': len(TARGET_MARINE) + len(KNOWN_FEATURES_MARINE),
            'num_targets': len(TARGET_MARINE),
            'split_counts': {
                'train': train_mask.sum(),
                'validation': (split_labels == 1).sum(),
                'test': (split_labels == 2).sum(),
            }
        }
