"""Tests for data schema validation."""

import pytest
from marine_local_mtgnn.constants import (
    RAW_CSV_COLUMNS,
    NODE_NAMES,
    TARGET_NAMES,
    NUM_INPUT_NODES,
    NUM_DIRECT_TARGETS,
)


def test_raw_csv_columns_count():
    """Raw CSV must have exactly 19 columns (1 timestamp + 18 parameters)."""
    assert len(RAW_CSV_COLUMNS) == 19
    assert RAW_CSV_COLUMNS[0] == "timestamp"


def test_node_names_count():
    """Graph must have exactly 19 input nodes."""
    assert len(NODE_NAMES) == 19
    assert NUM_INPUT_NODES == 19


def test_target_names_count():
    """Model must have exactly 15 direct targets."""
    assert len(TARGET_NAMES) == 15
    assert NUM_DIRECT_TARGETS == 15


def test_direction_nodes_exist():
    """Direction components must be present in input nodes."""
    direction_nodes = ["wind_u_east_ms", "wind_v_north_ms", "current_u_east_ms", "current_v_north_ms"]
    for node in direction_nodes:
        assert node in NODE_NAMES, f"Missing direction node: {node}"


def test_circular_encoding_nodes():
    """Circular parameters must be encoded as sin/cos."""
    circular_nodes = ["compass_sin", "compass_cos"]
    for node in circular_nodes:
        assert node in NODE_NAMES, f"Missing circular encoding: {node}"


def test_log_transformed_in_targets():
    """Log-transformed parameters must be in targets."""
    log_targets = [
        "log_significant_wave_height_m",
        "log_significant_wave_period_s",
        "log_zero_crossing_period_s",
        "log_peak_wave_period_s",
    ]
    for target in log_targets:
        assert target in TARGET_NAMES, f"Missing log target: {target}"


def test_log1p_in_targets():
    """Log1p radiation must be in targets."""
    assert "log1p_global_radiation_wm2" in TARGET_NAMES


def test_input_only_nodes():
    """Input-only nodes must not be in direct targets."""
    input_only = ["relative_humidity_pct", "conductivity_mscm", "compass_sin", "compass_cos"]
    for node in input_only:
        assert node not in TARGET_NAMES, f"Input-only node in targets: {node}"


def test_scalar_targets():
    """Scalar targets must be in both nodes and targets."""
    scalar_targets = [
        "air_temp_c",
        "air_pressure_hpa",
        "water_temp_c",
        "tidal_level_m",
        "dew_point_c",
        "salinity_psu",
    ]
    for target in scalar_targets:
        assert target in NODE_NAMES, f"Missing scalar in nodes: {target}"
        assert target in TARGET_NAMES, f"Missing scalar in targets: {target}"


def test_no_duplicate_nodes():
    """Node names must be unique."""
    assert len(NODE_NAMES) == len(set(NODE_NAMES)), "Duplicate nodes detected"


def test_no_duplicate_targets():
    """Target names must be unique."""
    assert len(TARGET_NAMES) == len(set(TARGET_NAMES)), "Duplicate targets detected"
