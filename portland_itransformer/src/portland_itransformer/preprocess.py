"""Complete preprocessing pipeline: validate → transform → split → scale."""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json

from .validate import DataValidator
from .features import (
    speed_dir_to_uv,
    apply_log_transform,
    create_cyclical_features,
    normalize_degrees,
)
from .baselines import UTideBaseline, ClearSkyBaseline
from .scaling import ScalingPipeline
from .calibrators import DerivedCalibrators
from .constants import TARGET_FEATURES, KNOWN_FEATURES

logger = logging.getLogger(__name__)


class Preprocessor:
    """Complete preprocessing pipeline."""

    def __init__(self, config):
        """Initialize preprocessor."""
        self.config = config
        self.df_raw = None
        self.df_processed = None
        self.split_labels = None

    def preprocess(self) -> dict:
        """
        Run complete preprocessing pipeline.

        Steps:
        1. Validate raw CSV
        2. Direction → u/v transforms
        3. UTide fit and reconstruct
        4. pvlib clear-sky
        5. Log transforms
        6. Cyclical features
        7. Chronological split
        8. Scaling (training only)
        9. Fit calibrators (training only)
        10. Save artifacts

        Returns
        -------
        dict
            Result dictionary with paths to processed data and artifacts
        """
        logger.info("=" * 80)
        logger.info("PREPROCESSING PIPELINE START")
        logger.info("=" * 80)

        # 1. VALIDATE
        logger.info("\n1. VALIDATION")
        validator = DataValidator(self.config)
        report = validator.validate()

        raw_csv = self.config.paths.raw_csv
        self.df_raw = pd.read_csv(raw_csv)

        # Remove unnamed index if present
        if self.df_raw.columns[0].startswith("Unnamed"):
            self.df_raw = self.df_raw.iloc[:, 1:]

        self.df_raw['timestamp'] = pd.to_datetime(self.df_raw['timestamp'], utc=True)
        self.df_raw = self.df_raw.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"✓ Loaded {len(self.df_raw)} rows from {raw_csv}")

        # 2. DIRECTION → U/V
        logger.info("\n2. DIRECTION TRANSFORMS")

        wind_u, wind_v = speed_dir_to_uv(
            self.df_raw['wind_speed_ms'].values,
            self.df_raw['wind_direction_deg'].values,
            convention=self.config.site.wind_direction_convention,
        )
        self.df_raw['wind_u_east_ms'] = wind_u
        self.df_raw['wind_v_north_ms'] = wind_v

        current_u, current_v = speed_dir_to_uv(
            self.df_raw['current_speed_ms'].values,
            self.df_raw['current_direction_deg'].values,
            convention=self.config.site.current_direction_convention,
        )
        self.df_raw['current_u_east_ms'] = current_u
        self.df_raw['current_v_north_ms'] = current_v

        logger.info("✓ Wind and current converted to u/v components")

        # 3. UTIDE
        logger.info("\n3. UTIDE HARMONIC TIDE BASELINE")

        utide_baseline = UTideBaseline(self.config)
        utide_baseline.fit(self.df_raw['timestamp'].values, self.df_raw['tidal_level_m'].values)
        tide_baseline = utide_baseline.reconstruct(self.df_raw['timestamp'].values)
        self.df_raw['tide_baseline_m'] = tide_baseline
        self.df_raw['tidal_residual_m'] = self.df_raw['tidal_level_m'] - tide_baseline

        logger.info("✓ UTide baseline computed and residuals extracted")

        # 4. CLEAR-SKY RADIATION
        logger.info("\n4. PVLIB CLEAR-SKY RADIATION")

        clearsky_baseline = ClearSkyBaseline(self.config)
        clearsky_baseline.fit(self.df_raw['timestamp'])
        clear_sky = clearsky_baseline.get_clear_sky(self.df_raw['timestamp'])
        self.df_raw['clear_sky_radiation_wm2'] = clear_sky

        logger.info("✓ Clear-sky radiation computed")

        # 5. LOG TRANSFORMS
        logger.info("\n5. LOG TRANSFORMS")

        EPS = self.config.reconstruction.eps
        self.df_raw['log_significant_wave_height_m'] = apply_log_transform(
            self.df_raw['significant_wave_height_m'].values, eps=EPS
        )
        self.df_raw['log_zero_crossing_period_s'] = apply_log_transform(
            self.df_raw['zero_crossing_period_s'].values, eps=EPS
        )

        # Clearness index: log1p(radiation / max(clear_sky, 20))
        clearness = self.df_raw['global_radiation_wm2'].values / np.maximum(
            self.df_raw['clear_sky_radiation_wm2'].values, 20.0
        )
        clearness = np.clip(clearness, 0.0, self.config.reconstruction.clearness_index_max)
        self.df_raw['log_clearness_index'] = np.log1p(clearness)

        logger.info("✓ Log transforms applied to waves and clearness")

        # 6. CYCLICAL FEATURES
        logger.info("\n6. CYCLICAL FEATURES")

        cyclical = create_cyclical_features(self.df_raw['timestamp'])
        for name, values in cyclical.items():
            self.df_raw[name] = values

        logger.info("✓ Hour and day-of-year cyclical features created")

        # 7. CHRONOLOGICAL SPLIT
        logger.info("\n7. CHRONOLOGICAL SPLIT")

        self.split_labels = self._create_split_labels(self.df_raw['timestamp'])
        n_train = (self.split_labels == 0).sum()
        n_val = (self.split_labels == 1).sum()
        n_test = (self.split_labels == 2).sum()

        logger.info(f"  Train: {n_train} rows")
        logger.info(f"  Valid: {n_val} rows")
        logger.info(f"  Test:  {n_test} rows")

        # 8. SCALING
        logger.info("\n8. SCALING (TRAINING ONLY)")

        scaler = ScalingPipeline(self.config)
        scaler.fit(self.df_raw, self.split_labels, TARGET_FEATURES, KNOWN_FEATURES)

        # Transform all splits
        df_targets_scaled, df_known_scaled = scaler.transform(
            self.df_raw, TARGET_FEATURES, KNOWN_FEATURES
        )

        # Combine into processed dataframe
        self.df_processed = pd.concat([
            self.df_raw[['timestamp']].reset_index(drop=True),
            df_targets_scaled.reset_index(drop=True),
            df_known_scaled.reset_index(drop=True),
        ], axis=1)

        logger.info(f"✓ Scaled to {self.df_processed.shape[1]} features")

        # 9. FIT CALIBRATORS
        logger.info("\n9. DERIVED OUTPUT CALIBRATORS (TRAINING ONLY)")

        df_train = self.df_raw[self.split_labels == 0].reset_index(drop=True)
        calibrators = DerivedCalibrators()
        calibrators.fit(df_train)

        logger.info("✓ Calibrators fitted on training data")

        # 10. SAVE ARTIFACTS
        logger.info("\n10. SAVING ARTIFACTS")

        processed_dir = Path(self.config.paths.processed_dir)
        artifacts_dir = Path(self.config.paths.artifacts_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save processed data
        processed_path = processed_dir / 'portland_preprocessed.parquet'
        self.df_processed.to_parquet(processed_path, index=False)
        logger.info(f"  Saved: {processed_path}")

        # Save split labels
        split_path = processed_dir / 'split_labels.npy'
        np.save(split_path, self.split_labels)
        logger.info(f"  Saved: {split_path}")

        # Save scalers
        scaler.save(artifacts_dir)

        # Save calibrators
        calibrators.save(artifacts_dir)

        # Save UTide
        utide_baseline.save(artifacts_dir)

        # Create manifest
        manifest = self._create_manifest(
            scaler, calibrators, len(self.df_raw)
        )
        manifest_path = processed_dir / 'feature_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"  Saved: {manifest_path}")

        logger.info("\n" + "=" * 80)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 80)

        return {
            'processed_path': str(processed_path),
            'split_labels_path': str(split_path),
            'manifest_path': str(manifest_path),
            'num_samples': len(self.df_raw),
            'num_features': self.df_processed.shape[1],
            'split_counts': {
                'train': int(n_train),
                'validation': int(n_val),
                'test': int(n_test),
            },
        }

    def _create_split_labels(self, timestamps: pd.Series) -> np.ndarray:
        """
        Create chronological split labels.

        Labels:
        - 0: training (Mar 1 - Sep 2)
        - 1: validation (Sep 3 - Nov 1)
        - 2: test (Nov 2 - Dec 31)

        Parameters
        ----------
        timestamps : Series
            UTC timestamps

        Returns
        -------
        labels : array
            [n_samples] with values 0, 1, or 2
        """
        train_start = pd.Timestamp(self.config.data.train_target_start)
        train_end = pd.Timestamp(self.config.data.train_target_end)
        val_start = pd.Timestamp(self.config.data.val_target_start)
        val_end = pd.Timestamp(self.config.data.val_target_end)
        test_start = pd.Timestamp(self.config.data.test_target_start)
        test_end = pd.Timestamp(self.config.data.test_target_end)

        labels = np.full(len(timestamps), -1, dtype=int)

        train_mask = (timestamps >= train_start) & (timestamps < train_end)
        labels[train_mask] = 0

        val_mask = (timestamps >= val_start) & (timestamps < val_end)
        labels[val_mask] = 1

        test_mask = (timestamps >= test_start) & (timestamps < test_end)
        labels[test_mask] = 2

        # Check for unlabeled
        unlabeled = (labels == -1).sum()
        if unlabeled > 0:
            logger.warning(f"Warning: {unlabeled} samples not assigned to split")

        return labels

    def _create_manifest(self, scaler, calibrators, num_rows: int) -> dict:
        """Create feature manifest."""
        return {
            'source_file': self.config.paths.raw_csv,
            'dataset_label': 'SYNTHETIC - Portland Harbor 2025',
            'note': 'Synthetic data for pipeline testing. Not real NOAA/buoy data.',
            'num_rows': num_rows,
            'num_features': len(TARGET_FEATURES) + len(KNOWN_FEATURES),
            'target_features': TARGET_FEATURES,
            'known_features': KNOWN_FEATURES,
            'transforms': {
                'wind': 'speed + direction → u/v (meteorological convention)',
                'current': 'speed + direction → u/v (oceanographic convention)',
                'tide': 'residual = observed - UTide baseline',
                'waves': 'log(Hs + eps), log(Tz + eps)',
                'clearness': 'log1p(radiation / max(clear_sky, 20))',
                'cyclical': 'hour_sin/cos, dayofyear_sin/cos',
            },
            'scalers': scaler.get_feature_stats(),
            'calibrators': {
                'conductivity': 'f(salinity, water_temp)',
                'sig_wave_period': 'f(log_Tz, log_Hs)',
                'peak_wave_period': 'f(log_Tz, log_Hs)',
            },
            'split_ranges': {
                'train': {
                    'start': self.config.data.train_target_start,
                    'end': self.config.data.train_target_end,
                },
                'validation': {
                    'start': self.config.data.val_target_start,
                    'end': self.config.data.val_target_end,
                },
                'test': {
                    'start': self.config.data.test_target_start,
                    'end': self.config.data.test_target_end,
                },
            },
        }
