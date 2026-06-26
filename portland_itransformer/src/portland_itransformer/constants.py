"""Constants and feature definitions."""

# Raw CSV columns (exact order from file)
RAW_COLUMNS = [
    "timestamp",
    "air_temp_c",
    "air_pressure_hpa",
    "wind_direction_deg",
    "water_temp_c",
    "relative_humidity_pct",
    "tidal_level_m",
    "current_direction_deg",
    "dew_point_c",
    "significant_wave_height_m",
    "global_radiation_wm2",
    "current_speed_ms",
    "wind_speed_ms",
    "salinity_psu",
    "significant_wave_period_s",
    "zero_crossing_period_s",
    "conductivity_mscm",
    "compass_deg",
    "peak_wave_period_s",
]

# Direct forecast targets (13 model outputs)
TARGET_FEATURES = [
    "air_temp_c",
    "air_pressure_hpa",
    "water_temp_c",
    "dew_point_c",
    "salinity_psu",
    "wind_u_east_ms",
    "wind_v_north_ms",
    "current_u_east_ms",
    "current_v_north_ms",
    "tidal_residual_m",
    "log_significant_wave_height_m",
    "log_zero_crossing_period_s",
    "log_clearness_index",
]

# Known deterministic covariates (6 inputs, not targets)
KNOWN_FEATURES = [
    "tide_baseline_m",
    "clear_sky_radiation_wm2",
    "hour_sin",
    "hour_cos",
    "dayofyear_sin",
    "dayofyear_cos",
]

# Full encoder input = targets + known
INPUT_FEATURES = TARGET_FEATURES + KNOWN_FEATURES

# Derived outputs (reconstructed from model predictions)
DERIVED_OUTPUTS = {
    "relative_humidity_pct": "Magnus(air_temp_c, dew_point_c)",
    "wind_speed_ms": "hypot(wind_u_east_ms, wind_v_north_ms)",
    "wind_direction_deg": "uv_to_dir(wind_u_east_ms, wind_v_north_ms, from)",
    "current_speed_ms": "hypot(current_u_east_ms, current_v_north_ms)",
    "current_direction_deg": "uv_to_dir(current_u_east_ms, current_v_north_ms, to)",
    "tidal_level_m": "tide_baseline_m + tidal_residual_m",
    "significant_wave_height_m": "exp(log_significant_wave_height_m) - eps",
    "zero_crossing_period_s": "exp(log_zero_crossing_period_s) - eps",
    "significant_wave_period_s": "RidgeCV(log_Tz, log_Hs)",
    "peak_wave_period_s": "RidgeCV(log_Tz, log_Hs)",
    "conductivity_mscm": "RidgeCV(salinity, water_temp)",
    "global_radiation_wm2": "clear_sky * clip(expm1(log_clearness_index), 0, 2)",
    "compass_deg": "NOT_MODELLED",
}

# Feature reporting order in final output
FINAL_REPORT_ORDER = [
    "timestamp",
    "air_temp_c",
    "air_pressure_hpa",
    "wind_direction_deg",
    "water_temp_c",
    "relative_humidity_pct",
    "tidal_level_m",
    "current_direction_deg",
    "dew_point_c",
    "significant_wave_height_m",
    "global_radiation_wm2",
    "current_speed_ms",
    "wind_speed_ms",
    "salinity_psu",
    "significant_wave_period_s",
    "zero_crossing_period_s",
    "conductivity_mscm",
    "compass_deg",
    "peak_wave_period_s",
]

# Scaling configuration
SCALERS = {
    "target_scaler": {
        "features": TARGET_FEATURES,
        "fit_on": "model_training_only",
    },
    "known_scaler": {
        "features": ["tide_baseline_m", "clear_sky_radiation_wm2"],
        "fit_on": "model_training_only",
    },
}

# Loss weights per target
TARGET_LOSS_WEIGHTS = {
    "air_temp_c": 1.0,
    "air_pressure_hpa": 1.0,
    "water_temp_c": 1.0,
    "dew_point_c": 1.0,
    "salinity_psu": 1.0,
    "wind_u_east_ms": 1.0,
    "wind_v_north_ms": 1.0,
    "current_u_east_ms": 1.0,
    "current_v_north_ms": 1.0,
    "tidal_residual_m": 0.8,
    "log_significant_wave_height_m": 1.2,
    "log_zero_crossing_period_s": 1.2,
    "log_clearness_index": 1.0,
}

# Evaluation horizon buckets (in hours)
HORIZON_BUCKETS = [
    (0, 6, "0-6h"),
    (6, 24, "6-24h"),
    (24, 72, "24-72h"),
    (72, 168, "72-168h"),
]

# Positive-only features (need log transform)
POSITIVE_ONLY = [
    "significant_wave_height_m",
    "zero_crossing_period_s",
    "current_speed_ms",
    "wind_speed_ms",
]

# Directional features (need u/v conversion)
DIRECTIONAL = {
    "wind_speed_ms": ("wind_direction_deg", "from"),
    "current_speed_ms": ("current_direction_deg", "to"),
}

# Feature index mapping
TARGET_INDEX = {name: i for i, name in enumerate(TARGET_FEATURES)}
KNOWN_INDEX = {name: i for i, name in enumerate(KNOWN_FEATURES)}
INPUT_INDEX = {name: i for i, name in enumerate(INPUT_FEATURES)}
