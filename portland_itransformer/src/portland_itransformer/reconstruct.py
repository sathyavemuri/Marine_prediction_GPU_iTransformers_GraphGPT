"""Reconstruct original 18-column output from 13 model targets."""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Tuple

from .features import (
    uv_to_speed_dir,
    relative_humidity_pct,
    inverse_log_transform,
    inverse_log1p_transform,
    clip_positive,
)

logger = logging.getLogger(__name__)


class OutputReconstructor:
    """Reconstruct original 18-column forecast from model predictions."""

    def __init__(
        self,
        config,
        scaler,
        calibrators,
        utide_baseline,
        clearsky_baseline,
    ):
        """
        Initialize reconstructor.

        Parameters
        ----------
        config : Config
            Configuration
        scaler : ScalingPipeline
            Fitted scaler for inverse transform
        calibrators : DerivedCalibrators
            Fitted calibrators for wave periods, conductivity
        utide_baseline : UTideBaseline
            Fitted tide model
        clearsky_baseline : ClearSkyBaseline
            Clear-sky radiation model
        """
        self.config = config
        self.scaler = scaler
        self.calibrators = calibrators
        self.utide_baseline = utide_baseline
        self.clearsky_baseline = clearsky_baseline

    def reconstruct(
        self,
        predictions_scaled: np.ndarray,
        known_features_scaled: np.ndarray,
        origin_timestamp: pd.Timestamp,
        target_names: list,
        known_names: list,
    ) -> pd.DataFrame:
        """
        Reconstruct full 18-column output.

        Parameters
        ----------
        predictions_scaled : array
            [horizon, 13] scaled target predictions
        known_features_scaled : array
            [horizon, 6] scaled known features
        origin_timestamp : Timestamp
            Forecast origin timestamp
        target_names : list
            13 target names
        known_names : list
            6 known feature names

        Returns
        -------
        DataFrame
            [horizon, 18] with original 18 columns
        """
        horizon = predictions_scaled.shape[0]

        # Step 1: Inverse transform targets and known
        predictions_physical = self.scaler.inverse_transform_targets(predictions_scaled)
        known_physical = self.scaler.inverse_transform_known(known_features_scaled)

        # Create result dictionary
        result = {}

        # Step 2: Timestamps (15-minute cadence)
        timestamps = []
        for i in range(horizon):
            ts = origin_timestamp + pd.Timedelta(minutes=15 * (i + 1))
            timestamps.append(ts)
        result['timestamp'] = timestamps

        # Step 3: Direct targets (just copy)
        direct_targets = [
            'air_temp_c', 'air_pressure_hpa', 'water_temp_c', 'dew_point_c',
            'salinity_psu'
        ]
        for target in direct_targets:
            if target in target_names:
                idx = target_names.index(target)
                result[target] = predictions_physical[:, idx]

        # Step 4: U/V components → speed and direction
        # Wind
        if 'wind_u_east_ms' in target_names and 'wind_v_north_ms' in target_names:
            wind_u_idx = target_names.index('wind_u_east_ms')
            wind_v_idx = target_names.index('wind_v_north_ms')

            wind_u = predictions_physical[:, wind_u_idx]
            wind_v = predictions_physical[:, wind_v_idx]

            wind_speed, wind_dir = uv_to_speed_dir(
                wind_u, wind_v,
                convention=self.config.site.wind_direction_convention
            )
            result['wind_speed_ms'] = clip_positive(wind_speed)
            result['wind_direction_deg'] = wind_dir

        # Current
        if 'current_u_east_ms' in target_names and 'current_v_north_ms' in target_names:
            current_u_idx = target_names.index('current_u_east_ms')
            current_v_idx = target_names.index('current_v_north_ms')

            current_u = predictions_physical[:, current_u_idx]
            current_v = predictions_physical[:, current_v_idx]

            current_speed, current_dir = uv_to_speed_dir(
                current_u, current_v,
                convention=self.config.site.current_direction_convention
            )
            result['current_speed_ms'] = clip_positive(current_speed)
            result['current_direction_deg'] = current_dir

        # Step 5: Tidal level = tide baseline + residual
        if 'tidal_residual_m' in target_names:
            tide_residual_idx = target_names.index('tidal_residual_m')
            tide_residual = predictions_physical[:, tide_residual_idx]

            # Get future tide baseline
            future_timestamps = pd.DatetimeIndex(timestamps)
            tide_baseline_future = self.utide_baseline.reconstruct(future_timestamps)

            result['tidal_level_m'] = tide_baseline_future + tide_residual

        # Step 6: Relative humidity from temp + dew point
        if 'air_temp_c' in result and 'dew_point_c' in result:
            rh = relative_humidity_pct(
                result['air_temp_c'],
                result['dew_point_c']
            )
            result['relative_humidity_pct'] = np.clip(rh, 0.0, 100.0)

        # Step 7: Wave heights and periods
        EPS = self.config.reconstruction.eps

        if 'log_significant_wave_height_m' in target_names:
            log_hs_idx = target_names.index('log_significant_wave_height_m')
            log_hs = predictions_physical[:, log_hs_idx]
            result['significant_wave_height_m'] = clip_positive(
                inverse_log_transform(log_hs, eps=EPS)
            )

        if 'log_zero_crossing_period_s' in target_names:
            log_tz_idx = target_names.index('log_zero_crossing_period_s')
            log_tz = predictions_physical[:, log_tz_idx]
            result['zero_crossing_period_s'] = clip_positive(
                inverse_log_transform(log_tz, eps=EPS)
            )

        # Derive wave periods using calibrators
        if 'log_zero_crossing_period_s' in target_names and \
           'log_significant_wave_height_m' in target_names:

            log_tz = predictions_physical[:, target_names.index('log_zero_crossing_period_s')]
            log_hs = predictions_physical[:, target_names.index('log_significant_wave_height_m')]

            sig_period = self.calibrators.predict_significant_wave_period(log_tz, log_hs)
            result['significant_wave_period_s'] = clip_positive(sig_period)

            peak_period = self.calibrators.predict_peak_wave_period(log_tz, log_hs)
            result['peak_wave_period_s'] = clip_positive(peak_period)

        # Step 8: Global radiation from clearness index
        if 'log_clearness_index' in target_names:
            log_clearness_idx = target_names.index('log_clearness_index')
            log_k = predictions_physical[:, log_clearness_idx]

            # Get future clear-sky radiation
            future_timestamps = pd.DatetimeIndex(timestamps)
            clear_sky_future = self.clearsky_baseline.get_clear_sky(future_timestamps)

            # Inverse: clearness = expm1(log_clearness)
            clearness = np.clip(
                inverse_log1p_transform(log_k),
                0.0,
                self.config.reconstruction.clearness_index_max
            )
            radiation = clear_sky_future * clearness

            # Zero out when clear_sky is low
            radiation[clear_sky_future < self.config.reconstruction.solar_daylight_threshold_wm2] = 0.0

            result['global_radiation_wm2'] = clip_positive(radiation)

        # Step 9: Conductivity from salinity + water temp using calibrator
        if 'salinity_psu' in result and 'water_temp_c' in result:
            conductivity = self.calibrators.predict_conductivity(
                result['salinity_psu'],
                result['water_temp_c']
            )
            result['conductivity_mscm'] = clip_positive(conductivity)

        # Step 10: Compass - set to NaN (not modeled)
        result['compass_deg'] = np.full(horizon, np.nan)

        # Create DataFrame in original column order
        column_order = [
            'timestamp', 'air_temp_c', 'air_pressure_hpa', 'wind_direction_deg',
            'water_temp_c', 'relative_humidity_pct', 'tidal_level_m',
            'current_direction_deg', 'dew_point_c', 'significant_wave_height_m',
            'global_radiation_wm2', 'current_speed_ms', 'wind_speed_ms',
            'salinity_psu', 'significant_wave_period_s', 'zero_crossing_period_s',
            'conductivity_mscm', 'compass_deg', 'peak_wave_period_s',
        ]

        df_result = pd.DataFrame(result)
        df_result = df_result[[col for col in column_order if col in df_result.columns]]

        logger.info(f"Reconstructed {horizon} timesteps × {len(df_result.columns)} columns")

        return df_result

    def validate_reconstruction(self, df_reconstructed: pd.DataFrame) -> bool:
        """
        Validate reconstructed output.

        Parameters
        ----------
        df_reconstructed : DataFrame
            Reconstructed 18-column output

        Returns
        -------
        bool
            True if valid, False otherwise
        """
        issues = []

        # Check columns
        required_cols = [
            'timestamp', 'air_temp_c', 'air_pressure_hpa', 'wind_direction_deg',
            'water_temp_c', 'relative_humidity_pct', 'tidal_level_m',
            'current_direction_deg', 'dew_point_c', 'significant_wave_height_m',
            'global_radiation_wm2', 'current_speed_ms', 'wind_speed_ms',
            'salinity_psu', 'significant_wave_period_s', 'zero_crossing_period_s',
            'conductivity_mscm', 'compass_deg', 'peak_wave_period_s',
        ]

        for col in required_cols:
            if col not in df_reconstructed.columns:
                issues.append(f"Missing column: {col}")

        # Check value ranges
        for col in ['wind_direction_deg', 'current_direction_deg']:
            if col in df_reconstructed.columns:
                mask = (df_reconstructed[col] < 0) | (df_reconstructed[col] >= 360)
                if mask.any():
                    issues.append(f"{col}: {mask.sum()} values outside [0, 360)")

        for col in ['relative_humidity_pct']:
            if col in df_reconstructed.columns:
                bad = df_reconstructed[col][(df_reconstructed[col] < 0) | (df_reconstructed[col] > 100)]
                if len(bad) > 0:
                    issues.append(f"{col}: {len(bad)} values outside [0, 100]")

        for col in ['significant_wave_height_m', 'current_speed_ms', 'wind_speed_ms',
                    'significant_wave_period_s', 'zero_crossing_period_s', 'peak_wave_period_s',
                    'global_radiation_wm2', 'conductivity_mscm']:
            if col in df_reconstructed.columns:
                bad = (df_reconstructed[col] < 0).sum()
                if bad > 0:
                    issues.append(f"{col}: {bad} negative values")

        if issues:
            logger.warning("Reconstruction validation issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            return False
        else:
            logger.info("✓ Reconstruction validation passed")
            return True
