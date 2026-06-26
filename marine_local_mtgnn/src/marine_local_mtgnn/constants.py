"""Constants for marine local MTGNN."""

# ===== RAW DATA SCHEMA =====

RAW_CSV_COLUMNS = [
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

RAW_PARAMETERS = [col for col in RAW_CSV_COLUMNS if col != "timestamp"]

# ===== GRAPH INPUT NODE NAMES (19 nodes) =====

NODE_NAMES = [
    "air_temp_c",                         # scalar
    "air_pressure_hpa",                   # scalar
    "wind_u_east_ms",                     # vector component
    "wind_v_north_ms",                    # vector component
    "water_temp_c",                       # scalar
    "relative_humidity_pct",              # input-only; derived output
    "tidal_level_m",                      # scalar
    "current_u_east_ms",                  # vector component
    "current_v_north_ms",                 # vector component
    "dew_point_c",                        # scalar
    "log_significant_wave_height_m",      # log(x + eps)
    "log1p_global_radiation_wm2",         # log(1+x)
    "salinity_psu",                       # scalar
    "log_significant_wave_period_s",      # log(x + eps)
    "log_zero_crossing_period_s",         # log(x + eps)
    "conductivity_mscm",                  # input-only; derived if validated
    "compass_sin",                        # input-only
    "compass_cos",                        # input-only
    "log_peak_wave_period_s",             # log(x + eps)
]

NUM_INPUT_NODES = len(NODE_NAMES)

# ===== DIRECT RESIDUAL TARGET NODES (15 targets) =====

TARGET_NAMES = [
    "air_temp_c",
    "air_pressure_hpa",
    "wind_u_east_ms",
    "wind_v_north_ms",
    "water_temp_c",
    "tidal_level_m",
    "current_u_east_ms",
    "current_v_north_ms",
    "dew_point_c",
    "log1p_global_radiation_wm2",
    "salinity_psu",
    "log_significant_wave_height_m",
    "log_significant_wave_period_s",
    "log_zero_crossing_period_s",
    "log_peak_wave_period_s",
]

NUM_DIRECT_TARGETS = len(TARGET_NAMES)

# ===== DERIVED / INPUT-ONLY QUANTITIES =====

DERIVED_OUTPUTS = {
    "relative_humidity_pct": ["air_temp_c", "dew_point_c"],  # depends on these targets
    "conductivity_mscm": ["salinity_psu", "water_temp_c"],  # depends on these targets
    "wind_speed_ms": ["wind_u_east_ms", "wind_v_north_ms"],
    "wind_direction_deg": ["wind_u_east_ms", "wind_v_north_ms"],
    "current_speed_ms": ["current_u_east_ms", "current_v_north_ms"],
    "current_direction_deg": ["current_u_east_ms", "current_v_north_ms"],
}

INPUT_ONLY = ["relative_humidity_pct", "conductivity_mscm", "compass_sin", "compass_cos"]

# ===== CIRCULAR / DIRECTIONAL PARAMETERS =====

CIRCULAR_PARAMS = [
    "wind_direction_deg",
    "current_direction_deg",
    "compass_deg",
]

# ===== LOG-TRANSFORMED PARAMETERS =====
# Maps raw parameter names (as they appear in CSV) to their transformed names

LOG_PARAMS = [
    "log_significant_wave_height_m",
    "log_significant_wave_period_s",
    "log_zero_crossing_period_s",
    "log_peak_wave_period_s",
]

LOG1P_PARAMS = [
    "log1p_global_radiation_wm2",
]

# Raw → log-transformed mapping (for apply_log_transform to know which transform to use)
LOG_TRANSFORM_RAW = {
    "significant_wave_height_m": "log",
    "significant_wave_period_s": "log",
    "zero_crossing_period_s": "log",
    "peak_wave_period_s": "log",
    "global_radiation_wm2": "log1p",
}

# ===== POSITIVE-ONLY PARAMETERS (no negative physical values) =====

POSITIVE_ONLY = [
    "global_radiation_wm2",
    "wind_speed_ms",
    "current_speed_ms",
    "significant_wave_height_m",
    "significant_wave_period_s",
    "zero_crossing_period_s",
    "peak_wave_period_s",
]

# ===== TIME RESOLUTION (PRIMARY 15-MINUTE PROFILE) =====

PRIMARY_FREQUENCY = "15min"
PRIMARY_LOOKBACK_STEPS = 672       # 7 days at 15-minute cadence
PRIMARY_HORIZON_STEPS = 672        # 7-day forecast
PRIMARY_SAMPLE_STRIDE = 4          # one training origin per hour

# ===== DATA SPLIT TIMESTAMPS (ISO 8601 UTC) =====

SPLIT_TRAIN_START = "2026-02-23T00:00:00"
SPLIT_TRAIN_END_EXCLUSIVE = "2026-05-24T00:00:00"
SPLIT_VALIDATION_START = "2026-05-24T00:00:00"
SPLIT_VALIDATION_END_EXCLUSIVE = "2026-06-13T00:00:00"
SPLIT_TEST_START = "2026-06-13T00:00:00"
SPLIT_TEST_END_EXCLUSIVE = "2026-06-23T00:00:00"

# Train: 90 days
# Validation: 20 days (rolling daily origins)
# Test: 10 days (4 daily origins with complete 7-day targets)

# ===== NUMERICAL EPSILON VALUES =====

LOG_EPSILON = 1e-4  # for log(x + eps) transforms
ZERO_THRESHOLD = 1e-6

# ===== DIRECTION CONVENTIONS (MUST BE CONFIRMED) =====

# Default: "from" (meteorological) convention
# wind_direction_convention: "from" means direction the wind comes FROM, clockwise from north
# current_direction_convention: "to" means direction current flows TO, clockwise from north
# These MUST be explicitly set in config for each site.

DIRECTION_CONVENTION_FROM = "from"
DIRECTION_CONVENTION_TO = "to"
