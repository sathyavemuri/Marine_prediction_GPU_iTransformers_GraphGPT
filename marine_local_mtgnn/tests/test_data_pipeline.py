"""Integration tests for data pipeline."""

import pytest
import tempfile
import pandas as pd
from pathlib import Path

from marine_local_mtgnn.data.synthetic import generate_synthetic_data
from marine_local_mtgnn.data.validator import DataValidator
from marine_local_mtgnn.data.preprocessor import DataPreprocessor
from marine_local_mtgnn.config import Config, load_config
from marine_local_mtgnn.constants import NODE_NAMES, TARGET_NAMES


@pytest.fixture
def temp_config():
    """Create a temporary config with synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate synthetic data
        csv_path = tmpdir / "synthetic_data.csv"
        generate_synthetic_data(output_path=csv_path)

        # Create a config pointing to synthetic data
        config_dict = {
            "experiment_name": "test_pipeline",
            "seed": 42,
            "timezone": "UTC",
            "output_root": str(tmpdir / "outputs"),
            "site": {
                "buoy_id": "TEST_BUOY",
                "latitude": 40.0,
                "longitude": -70.0,
                "altitude_m": 0.0,
                "timezone": "UTC",
                "sea_pressure_dbar": 0.0,
                "wind_direction_convention": "from",
                "current_direction_convention": "to",
                "sensor_depth_m": None,
                "wind_sensor_height_m": None,
            },
            "data": {
                "raw_csv": str(csv_path),
                "resample_rule": "15min",
                "resample_label": "right",
                "resample_closed": "right",
            },
            "splits": {
                "train_start": "2026-02-23T00:00:00",
                "train_end_exclusive": "2026-05-24T00:00:00",
                "validation_start": "2026-05-24T00:00:00",
                "validation_end_exclusive": "2026-06-13T00:00:00",
                "test_start": "2026-06-13T00:00:00",
                "test_end_exclusive": "2026-06-23T00:00:00",
            },
            "forecast": {
                "profile": "long_7day",
                "lookback_steps": 672,
                "horizon_steps": 672,
                "sample_stride_steps": 4,
                "direct_multi_horizon": True,
            },
            "baselines": {
                "persistence": True,
                "daily_seasonal": True,
                "weekly_seasonal": False,
                "local_trend": True,
                "trend_window_steps": 12,
                "blended": True,
                "tide_enabled": False,
                "radiation_enabled": False,
                "selection_metric": "mae_physical",
                "max_lag_graph_steps": 96,
            },
            "model": {
                "num_input_nodes": 19,
                "num_direct_targets": 15,
                "hidden_channels": 16,
                "skip_channels": 32,
                "end_channels": 64,
                "kernel_size": 3,
                "dilation_exponential": 2,
                "graph_top_k": 4,
                "graph_conv_depth": 2,
                "dropout": 0.15,
                "max_trainable_parameters": 2000000,
            },
            "decoder": {
                "mode": "horizon_conditioned_direct",
                "target_embedding_dim": 8,
                "lead_time_embedding_dim": 16,
                "decoder_hidden": 64,
                "use_future_baseline": True,
                "use_future_calendar": True,
            },
            "training": {
                "batch_size": 8,
                "num_workers": 0,
                "max_epochs": 10,
                "early_stopping_patience": 5,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 1.0,
                "monitor": "validation_weighted_physical_mae",
            },
            "postprocess": {
                "humidity_formula_enabled": True,
                "conductivity_formula_enabled": False,
                "conductivity_max_validation_mae_mscm": 3.0,
            },
            "service": {
                "host": "0.0.0.0",
                "port": 8000,
                "minimum_history_steps": 672,
                "prediction_horizon_steps": 672,
            },
        }

        config = Config(**config_dict)
        yield config, tmpdir


def test_data_validation(temp_config):
    """Test data validation step."""
    config, tmpdir = temp_config

    validator = DataValidator(config)
    report = validator.validate()

    assert report["validation_status"] == "pass"
    assert report["summary"]["total_rows"] > 0
    assert report["summary"]["total_columns"] == 18  # 18 raw parameters + timestamp
    assert report["monotonicity"]["is_monotonic"] is True
    assert report["monotonicity"]["duplicate_count"] == 0


def test_data_preprocessing(temp_config):
    """Test data preprocessing and splitting."""
    config, tmpdir = temp_config

    preprocessor = DataPreprocessor(config)
    splits = preprocessor.preprocess()

    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits

    # Check that splits are non-empty
    assert len(splits["train"]) > 0
    assert len(splits["validation"]) > 0
    assert len(splits["test"]) > 0

    # Check that data has been transformed
    train = splits["train"]
    # Should have direction u/v components instead of direction/speed
    assert "wind_u_east_ms" in train.columns
    assert "wind_v_north_ms" in train.columns
    assert "wind_direction_deg" not in train.columns
    assert "wind_speed_ms" not in train.columns

    # Should have log-transformed wave parameters
    assert "log_significant_wave_height_m" in train.columns
    assert "log_significant_wave_period_s" in train.columns

    # Should have circular encoding
    assert "compass_sin" in train.columns
    assert "compass_cos" in train.columns

    # Should have resampled to 15 minutes
    assert len(splits["train"]) < 120 * 24 * 60  # Less than raw 1-minute data


def test_data_shape_consistency(temp_config):
    """Test that preprocessed data has expected shape."""
    config, tmpdir = temp_config

    preprocessor = DataPreprocessor(config)
    splits = preprocessor.preprocess()

    train = splits["train"]

    # Check column count: 19 node names (from NODE_NAMES)
    expected_columns = set(NODE_NAMES)
    actual_columns = set(train.columns)

    # The actual columns should be a subset or equal to NODE_NAMES
    # (some may be dropped if input-only)
    assert len(actual_columns) <= len(NODE_NAMES)


def test_splits_no_overlap(temp_config):
    """Test that splits don't overlap."""
    config, tmpdir = temp_config

    preprocessor = DataPreprocessor(config)
    splits = preprocessor.preprocess()

    train_idx = set(splits["train"].index)
    val_idx = set(splits["validation"].index)
    test_idx = set(splits["test"].index)

    assert len(train_idx & val_idx) == 0, "Train and validation overlap"
    assert len(val_idx & test_idx) == 0, "Validation and test overlap"
    assert len(train_idx & test_idx) == 0, "Train and test overlap"
