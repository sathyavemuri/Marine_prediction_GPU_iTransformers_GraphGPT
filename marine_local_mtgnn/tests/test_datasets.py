"""Tests for dataset creation components."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from marine_local_mtgnn.datasets.graph_prior import GraphPrior
from marine_local_mtgnn.datasets.scalers import ResidualScaler
from marine_local_mtgnn.baselines.selector import BaselineSelector
from marine_local_mtgnn.datasets.residuals import ResidualDataset
from marine_local_mtgnn.config import Config
from marine_local_mtgnn.constants import NODE_NAMES, TARGET_NAMES


@pytest.fixture
def synthetic_data():
    """Create synthetic data splits."""
    dates = pd.date_range("2026-02-01", periods=1500, freq="15min", tz="UTC")
    data = {
        name: np.sin(np.arange(1500) / 100) + np.random.normal(0, 0.1, 1500)
        for name in NODE_NAMES
    }
    df = pd.DataFrame(data, index=dates)

    # Split
    train = df.iloc[:1000]
    validation = df.iloc[1000:1300]
    test = df.iloc[1300:]

    return train, validation, test


@pytest.fixture
def test_config():
    """Create test configuration."""
    return Config(
        experiment_name="test",
        seed=42,
        timezone="UTC",
        output_root="outputs",
        site={
            "buoy_id": "TEST",
            "latitude": 40.0,
            "longitude": -70.0,
            "altitude_m": 0.0,
            "timezone": "UTC",
            "sea_pressure_dbar": 0.0,
            "wind_direction_convention": "from",
            "current_direction_convention": "to",
        },
        data={
            "raw_csv": "data.csv",
            "resample_rule": "15min",
            "resample_label": "right",
            "resample_closed": "right",
        },
        splits={
            "train_start": "2026-02-01T00:00:00",
            "train_end_exclusive": "2026-02-15T00:00:00",
            "validation_start": "2026-02-15T00:00:00",
            "validation_end_exclusive": "2026-02-21T00:00:00",
            "test_start": "2026-02-21T00:00:00",
            "test_end_exclusive": "2026-02-27T00:00:00",
        },
        forecast={
            "profile": "long_7day",
            "lookback_steps": 96,
            "horizon_steps": 96,
            "sample_stride_steps": 12,
            "direct_multi_horizon": True,
        },
        baselines={
            "persistence": True,
            "daily_seasonal": False,
            "weekly_seasonal": False,
            "local_trend": False,
            "trend_window_steps": 12,
            "blended": False,
            "tide_enabled": False,
            "radiation_enabled": False,
            "selection_metric": "mae_physical",
            "max_lag_graph_steps": 24,
        },
        model={
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
        decoder={
            "mode": "horizon_conditioned_direct",
            "target_embedding_dim": 8,
            "lead_time_embedding_dim": 16,
            "decoder_hidden": 64,
            "use_future_baseline": True,
            "use_future_calendar": True,
        },
        training={
            "batch_size": 8,
            "num_workers": 0,
            "max_epochs": 10,
            "early_stopping_patience": 5,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "monitor": "validation_weighted_physical_mae",
        },
        postprocess={
            "humidity_formula_enabled": True,
            "conductivity_formula_enabled": False,
            "conductivity_max_validation_mae_mscm": 3.0,
        },
        service={
            "host": "0.0.0.0",
            "port": 8000,
            "minimum_history_steps": 672,
            "prediction_horizon_steps": 672,
        },
    )


class TestGraphPrior:
    """Test graph prior construction."""

    def test_graph_prior_fit(self, synthetic_data):
        """Test graph prior fitting."""
        train, _, _ = synthetic_data
        prior = GraphPrior(max_lag_steps=12)
        adj = prior.fit(train)

        assert adj.shape == (len(NODE_NAMES), len(NODE_NAMES))
        assert np.all(np.isfinite(adj))
        assert np.all(adj >= 0)
        assert np.all(adj <= 1)  # Correlations in [0, 1]

    def test_sparse_adjacency(self, synthetic_data):
        """Test sparse adjacency extraction."""
        train, _, _ = synthetic_data
        prior = GraphPrior()
        prior.fit(train)

        sparse = prior.get_sparse_adjacency(top_k=4)
        assert sparse.shape == (len(NODE_NAMES), len(NODE_NAMES))

        # Check sparsity: each row should have at most 4 non-zero entries
        for i in range(sparse.shape[0]):
            non_zero = np.sum(sparse[i, :] > 0)
            assert non_zero <= 4


class TestResidualScaler:
    """Test residual scaler."""

    def test_scaler_fit_and_transform(self):
        """Test scaler fitting and transformation."""
        # Create synthetic residuals
        residuals_list = [
            np.random.normal(0, 1, (100, 15)) for _ in range(5)
        ]

        scaler = ResidualScaler()
        scaler.fit(residuals_list)

        assert scaler.is_fitted
        assert scaler.mean.shape == (15,)
        assert scaler.std.shape == (15,)

    def test_scaler_transform_inverse(self):
        """Test transform and inverse_transform are inverses."""
        residuals = np.random.normal(0, 1, (100, 15))
        scaler = ResidualScaler()
        scaler.fit([residuals])

        transformed = scaler.transform(residuals)
        recovered = scaler.inverse_transform(transformed)

        np.testing.assert_allclose(residuals, recovered, rtol=1e-5)

    def test_scaler_standardization(self):
        """Test that transformed residuals are standardized."""
        residuals = np.random.normal(5, 2, (1000, 15))
        scaler = ResidualScaler()
        scaler.fit([residuals])

        transformed = scaler.transform(residuals)

        # Mean should be close to 0
        assert np.abs(np.mean(transformed)) < 0.1
        # Std should be close to 1
        assert np.abs(np.std(transformed) - 1) < 0.1


class TestResidualDataset:
    """Test residual dataset creation."""

    def test_residual_dataset_creation(self, synthetic_data, test_config):
        """Test residual dataset creation."""
        train, validation, test = synthetic_data

        # Create baseline selector
        selector = BaselineSelector(test_config)
        selector.fit_all(train, validation)

        # Create residual dataset
        creator = ResidualDataset(test_config, selector)
        residuals = creator.create(train, validation, test)

        assert "train" in residuals
        assert "validation" in residuals
        assert "test" in residuals

        # Check structure
        for split_data in residuals.values():
            assert "windows" in split_data
            assert "baselines" in split_data
            assert "targets" in split_data
            assert "num_samples" in split_data

    def test_residuals_have_correct_shape(self, synthetic_data, test_config):
        """Test residual shapes."""
        train, validation, test = synthetic_data

        selector = BaselineSelector(test_config)
        selector.fit_all(train, validation)

        creator = ResidualDataset(test_config, selector)
        residuals = creator.create(train, validation, test)

        # Check a training window
        if residuals["train"]["num_samples"] > 0:
            window = residuals["train"]["windows"][0]
            assert window["history"].shape == (96, len(NODE_NAMES))
            assert window["residuals"].shape == (96, len(TARGET_NAMES))
            assert window["actual_targets"].shape == (96, len(TARGET_NAMES))
            assert window["baseline_forecast"].shape[0] == 96
