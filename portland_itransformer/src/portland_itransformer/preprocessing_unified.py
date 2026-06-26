"""Unified preprocessing for 22-parameter model with heterogeneous atmosphere residuals."""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
import joblib
from .baselines import UTideBaseline, ClearSkyBaseline, HarmonicAnomalyBaseline
from .features import speed_dir_to_uv, apply_log_transform

logger = logging.getLogger(__name__)

# 8 Marine targets (direct outputs)
TARGET_MARINE_UNIFIED = [
    'tidal_residual_m',
    'current_u_east_ms',
    'current_v_north_ms',
    'salinity_psu',
    'water_temp_c',
    'log1p_global_radiation_wm2',
    'log_significant_wave_height_m',
    'log_zero_crossing_period_s',
]

# 14 Atmosphere targets (anomaly-based + derived)
TARGET_ATMOSPHERE_UNIFIED = [
    'air_temp_anomaly_c',
    'air_temp_gradient_c',
    'log1p_dewpoint_depression_c',
    'air_pressure_anomaly_hpa',
    'air_pressure_gradient_hpa',
    'wind_u_anomaly_ms',
    'wind_v_anomaly_ms',
    'wind_shear_ms',
    'diurnal_cycle_temp_sin',
    'diurnal_cycle_temp_cos',
    'diurnal_cycle_pressure_sin',
    'diurnal_cycle_pressure_cos',
    'sea_surface_temp_feedback_c',
    'atmospheric_instability_index',
]

# All 22 targets
TARGET_UNIFIED = TARGET_MARINE_UNIFIED + TARGET_ATMOSPHERE_UNIFIED

# Input features (direct + residual + coupling)
KNOWN_FEATURES_UNIFIED = [
    'hour_sin',
    'hour_cos',
    'dayofyear_sin',
    'dayofyear_cos',
]


class UnifiedPreprocessor:
    """Unified 22-parameter preprocessor with heterogeneous atmosphere residuals."""

    def __init__(self, config):
        self.config = config
        self.scaler_targets = None
        self.scaler_known = None
        self.harmonic_temp = None
        self.harmonic_pressure = None
        self.wind_climatology = None
        self.df_raw = None

    def preprocess(self):
        """Run full preprocessing pipeline."""
        logger.info("=" * 80)
        logger.info("UNIFIED 22-PARAMETER PREPROCESSING PIPELINE START")
        logger.info("=" * 80)

        # 1. Validation
        logger.info("\n1. VALIDATION")
        from .validate import DataValidator
        validator = DataValidator(self.config)
        validator.validate()

        # Load raw data
        self.df_raw = pd.read_csv(self.config.paths.raw_csv)
        if self.df_raw.columns[0].startswith("Unnamed"):
            self.df_raw = self.df_raw.iloc[:, 1:]
        self.df_raw['timestamp'] = pd.to_datetime(self.df_raw['timestamp'], utc=True)
        self.df_raw = self.df_raw.sort_values('timestamp').reset_index(drop=True)
        logger.info(f"✓ Loaded {len(self.df_raw)} rows from {self.config.paths.raw_csv}")

        # 2. Marine transforms
        logger.info("\n2. DIRECTION TRANSFORMS (Marine)")
        self._transform_directions()

        # 7. Chronological split
        logger.info("\n7. CHRONOLOGICAL SPLIT")
        split_labels = self._chronological_split()

        # 3. Marine baselines
        logger.info("\n3. TIDAL & RADIATION BASELINES (Train-Only Fit)")
        self._fit_marine_baselines(split_labels)

        # 4. Atmosphere baselines
        logger.info("\n4. ATMOSPHERE BASELINES (Train-Only Fit)")
        self._fit_atmosphere_baselines(split_labels)

        # 5. Cyclical features
        logger.info("\n5. CYCLICAL FEATURES")
        self._create_cyclical_features()

        # 6. Derive heterogeneous atmosphere variables
        logger.info("\n6. HETEROGENEOUS ATMOSPHERE RESIDUALS")
        self._derive_heterogeneous_variables(split_labels)

        # 7. Log transforms
        logger.info("\n7. LOG TRANSFORMS (Marine)")
        self._apply_log_transforms()

        # 8. Scaling
        logger.info("\n8. SCALING (TRAIN-ONLY FIT)")
        self._fit_scalers(split_labels)

        # 9. Save artifacts
        logger.info("\n9. SAVING ARTIFACTS")
        self._save_artifacts(split_labels)

        logger.info("\n" + "=" * 80)
        logger.info("UNIFIED PREPROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  ✓ Processed: {len(self.df_raw)} rows × {len(self.df_raw.columns)} features")
        logger.info(f"  ✓ Targets: {len(TARGET_UNIFIED)} (8 marine + 14 atmosphere)")
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

    def _transform_directions(self):
        """Convert wind and current directions to u/v components."""
        logger.info("\n2. DIRECTION TRANSFORMS")

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

    def _chronological_split(self):
        """Create train/val/test split."""
        n = len(self.df_raw)
        train_end = int(n * 0.5)
        val_end = int(n * 0.75)

        split_labels = np.zeros(n, dtype=int)
        split_labels[train_end:val_end] = 1
        split_labels[val_end:] = 2

        logger.info(f"  Train: {train_end} rows")
        logger.info(f"  Valid: {val_end - train_end} rows")
        logger.info(f"  Test:  {n - val_end} rows")

        np.save(Path(self.config.paths.processed_dir) / "unified_split_labels.npy", split_labels)
        return split_labels

    def _fit_marine_baselines(self, split_labels):
        """Fit tidal and radiation baselines on training data."""
        logger.info("\n3. TIDAL & RADIATION BASELINES (Train-Only Fit)")

        train_mask = split_labels == 0

        # Tidal baseline
        utide_baseline = UTideBaseline(self.config)
        utide_baseline.fit(
            pd.DatetimeIndex(self.df_raw['timestamp']),
            self.df_raw['tidal_level_m'].values
        )

        # Compute residuals
        tidal_baseline_full = utide_baseline.reconstruct(
            pd.DatetimeIndex(self.df_raw['timestamp'])
        )
        self.df_raw['tidal_residual_m'] = self.df_raw['tidal_level_m'] - tidal_baseline_full

        # Clear-sky radiation
        clearsky_baseline = ClearSkyBaseline(self.config)
        clearsky_baseline.fit(pd.DatetimeIndex(self.df_raw['timestamp']))

        clearsky_wm2 = clearsky_baseline.get_clear_sky(
            pd.DatetimeIndex(self.df_raw['timestamp'])
        )

        # Global radiation residual
        self.df_raw['global_radiation_residual_wm2'] = (
            self.df_raw['global_radiation_wm2'] - clearsky_wm2
        )

        logger.info("✓ Marine baselines fitted (UTide, clear-sky)")

    def _fit_atmosphere_baselines(self, split_labels):
        """Fit harmonic baselines and wind climatology on training data."""
        logger.info("\n4. ATMOSPHERE BASELINES (Train-Only Fit)")

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

        # Wind climatology (hourly means from training)
        self.wind_climatology = {}
        for hour in range(24):
            hour_mask = (pd.DatetimeIndex(self.df_raw['timestamp']).hour == hour) & train_mask
            if hour_mask.sum() > 0:
                self.wind_climatology[hour] = {
                    'u': self.df_raw.loc[hour_mask, 'wind_u_east_ms'].mean(),
                    'v': self.df_raw.loc[hour_mask, 'wind_v_north_ms'].mean(),
                }

        logger.info("✓ Atmosphere baselines fitted (harmonic, wind climatology)")

    def _create_cyclical_features(self):
        """Create hourly and daily cyclical features."""
        logger.info("\n5. CYCLICAL FEATURES")

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

    def _derive_heterogeneous_variables(self, split_labels):
        """Derive heterogeneous atmosphere residual variables."""
        logger.info("\n6. HETEROGENEOUS ATMOSPHERE RESIDUALS")

        timestamps = pd.DatetimeIndex(self.df_raw['timestamp'])

        # Anomalies (residuals from baselines)
        self.df_raw['air_temp_anomaly_c'] = (
            self.df_raw['air_temp_c'] - self.harmonic_temp.predict(timestamps)
        )
        self.df_raw['air_pressure_anomaly_hpa'] = (
            self.df_raw['air_pressure_hpa'] - self.harmonic_pressure.predict(timestamps)
        )

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

        # Dew point depression
        self.df_raw['log1p_dewpoint_depression_c'] = np.log1p(
            np.maximum(self.df_raw['air_temp_c'] - self.df_raw['dew_point_c'], 0.01)
        )

        # Gradients (temporal changes)
        self.df_raw['air_temp_gradient_c'] = self.df_raw['air_temp_c'].diff().fillna(0)
        self.df_raw['air_pressure_gradient_hpa'] = self.df_raw['air_pressure_hpa'].diff().fillna(0)

        # Wind shear (temporal gradient)
        wind_speed = np.sqrt(
            self.df_raw['wind_u_east_ms']**2 + self.df_raw['wind_v_north_ms']**2
        )
        self.df_raw['wind_shear_ms'] = wind_speed.diff().fillna(0)

        # Diurnal cycle decomposition (phase in day)
        hour_angle = 2 * np.pi * timestamps.hour / 24.0
        self.df_raw['diurnal_cycle_temp_sin'] = np.sin(hour_angle)
        self.df_raw['diurnal_cycle_temp_cos'] = np.cos(hour_angle)
        self.df_raw['diurnal_cycle_pressure_sin'] = np.sin(hour_angle)
        self.df_raw['diurnal_cycle_pressure_cos'] = np.cos(hour_angle)

        # Sea surface temperature feedback (water temp → air temp influence)
        self.df_raw['sea_surface_temp_feedback_c'] = self.df_raw['water_temp_c'].rolling(
            window=24, min_periods=1
        ).mean() - self.df_raw['air_temp_c']

        # Atmospheric instability index (dew point depression increasing)
        dew_depression = self.df_raw['air_temp_c'] - self.df_raw['dew_point_c']
        self.df_raw['atmospheric_instability_index'] = dew_depression.diff().fillna(0)

        logger.info("✓ Heterogeneous atmosphere residuals derived")

    def _apply_log_transforms(self):
        """Apply log transforms to marine parameters."""
        logger.info("\n7. LOG TRANSFORMS (Marine)")

        # Global radiation residual
        self.df_raw['log1p_global_radiation_wm2'] = np.log1p(
            np.maximum(self.df_raw['global_radiation_residual_wm2'], 0.0)
        )

        # Wave parameters
        self.df_raw['log_significant_wave_height_m'] = apply_log_transform(
            self.df_raw['significant_wave_height_m'].values,
            eps=1e-4
        )

        self.df_raw['log_zero_crossing_period_s'] = apply_log_transform(
            self.df_raw['zero_crossing_period_s'].values,
            eps=1e-4
        )

        logger.info("✓ Log transforms applied")

    def _fit_scalers(self, split_labels):
        """Fit scalers on training data only."""
        logger.info("\n8. SCALING (TRAIN-ONLY FIT)")

        train_mask = split_labels == 0

        # Target scaler
        target_data = self.df_raw.loc[train_mask, TARGET_UNIFIED].values
        self.scaler_targets = StandardScaler()
        self.scaler_targets.fit(target_data)

        # Known features scaler
        known_data = self.df_raw.loc[train_mask, KNOWN_FEATURES_UNIFIED].values
        self.scaler_known = StandardScaler()
        self.scaler_known.fit(known_data)

        # Scale all data
        self.df_raw[TARGET_UNIFIED] = self.scaler_targets.transform(
            self.df_raw[TARGET_UNIFIED].values
        )
        self.df_raw[KNOWN_FEATURES_UNIFIED] = self.scaler_known.transform(
            self.df_raw[KNOWN_FEATURES_UNIFIED].values
        )

        logger.info(f"✓ Target scaler fit on {train_mask.sum()} training samples")
        logger.info(f"✓ Known scaler fit on {train_mask.sum()} training samples")

    def _save_artifacts(self, split_labels):
        """Save processed data and artifacts."""
        logger.info("\n9. SAVING ARTIFACTS")

        processed_dir = Path(self.config.paths.processed_dir)
        artifacts_dir = Path(self.config.paths.artifacts_dir)

        processed_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save processed data
        self.df_raw.to_parquet(processed_dir / "unified_preprocessed.parquet")
        np.save(processed_dir / "unified_split_labels.npy", split_labels)

        # Save scalers
        joblib.dump(self.scaler_targets, artifacts_dir / "unified_scaler_targets.joblib")
        joblib.dump(self.scaler_known, artifacts_dir / "unified_scaler_known.joblib")

        # Save baselines
        joblib.dump(self.harmonic_temp, artifacts_dir / "unified_harmonic_temp.joblib")
        joblib.dump(self.harmonic_pressure, artifacts_dir / "unified_harmonic_pressure.joblib")
        joblib.dump(self.wind_climatology, artifacts_dir / "unified_wind_climatology.joblib")

        logger.info(f"  Saved: {processed_dir / 'unified_preprocessed.parquet'}")
        logger.info(f"  Saved: {processed_dir / 'unified_split_labels.npy'}")
        logger.info(f"  Saved scalers and baselines to {artifacts_dir}")
