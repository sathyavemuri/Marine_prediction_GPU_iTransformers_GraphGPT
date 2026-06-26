"""Tests for training pipeline."""

import pytest
import torch
from torch.utils.data import DataLoader
import tempfile
from pathlib import Path

from marine_local_mtgnn.training.dataset import ResidualWindowDataset
from marine_local_mtgnn.training.trainer import Trainer
from marine_local_mtgnn.models import MTGNN
from marine_local_mtgnn.config import Config


@pytest.fixture
def synthetic_windows():
    """Create synthetic residual windows."""
    windows = []
    for i in range(20):
        window = {
            "history": torch.randn(96, 19).numpy(),  # (lookback, nodes)
            "baseline_forecast": torch.randn(96, 19).numpy(),  # (horizon, nodes)
            "residuals": torch.randn(96, 15).numpy(),  # (horizon, targets)
        }
        windows.append(window)
    return windows


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
            "batch_size": 4,
            "num_workers": 0,
            "max_epochs": 5,
            "early_stopping_patience": 3,
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


class TestResidualWindowDataset:
    """Test PyTorch dataset."""

    def test_dataset_length(self, synthetic_windows):
        """Test dataset length."""
        dataset = ResidualWindowDataset(synthetic_windows)
        assert len(dataset) == 20

    def test_dataset_getitem(self, synthetic_windows):
        """Test dataset sample retrieval."""
        dataset = ResidualWindowDataset(synthetic_windows)
        sample = dataset[0]

        assert "history" in sample
        assert "baseline" in sample
        assert "targets" in sample
        assert sample["history"].shape == (96, 19)
        assert sample["targets"].shape == (96, 15)

    def test_dataloader(self, synthetic_windows):
        """Test DataLoader with dataset."""
        dataset = ResidualWindowDataset(synthetic_windows)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        batch = next(iter(loader))
        assert batch["history"].shape[0] <= 4
        assert batch["history"].shape[1:] == (96, 19)
        assert batch["targets"].shape[1:] == (96, 15)


class TestTrainer:
    """Test trainer."""

    def test_trainer_initialization(self, test_config):
        """Test trainer initialization."""
        model = MTGNN(
            num_nodes=19,
            num_targets=15,
            lookback_steps=96,
            horizon_steps=96,
            hidden_channels=test_config.model.hidden_channels,
        )
        trainer = Trainer(model, test_config, device="cpu")

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.best_val_loss == float("inf")
        assert len(trainer.train_losses) == 0

    def test_trainer_train_epoch(self, test_config, synthetic_windows):
        """Test training one epoch."""
        model = MTGNN(
            num_nodes=19,
            num_targets=15,
            lookback_steps=96,
            horizon_steps=96,
            hidden_channels=test_config.model.hidden_channels,
        )
        trainer = Trainer(model, test_config, device="cpu")

        dataset = ResidualWindowDataset(synthetic_windows)
        loader = DataLoader(dataset, batch_size=4)

        loss = trainer.train_epoch(loader)

        assert isinstance(loss, float)
        assert loss > 0
        assert len(trainer.train_losses) == 0  # Not updated in train_epoch

    def test_trainer_validate(self, test_config, synthetic_windows):
        """Test validation."""
        model = MTGNN(
            num_nodes=19,
            num_targets=15,
            lookback_steps=96,
            horizon_steps=96,
            hidden_channels=test_config.model.hidden_channels,
        )
        trainer = Trainer(model, test_config, device="cpu")

        dataset = ResidualWindowDataset(synthetic_windows)
        loader = DataLoader(dataset, batch_size=4)

        loss = trainer.validate(loader)

        assert isinstance(loss, float)
        assert loss > 0

    def test_trainer_fit(self, test_config, synthetic_windows):
        """Test full training loop."""
        model = MTGNN(
            num_nodes=19,
            num_targets=15,
            lookback_steps=96,
            horizon_steps=96,
            hidden_channels=test_config.model.hidden_channels,
        )
        trainer = Trainer(model, test_config, device="cpu")

        dataset = ResidualWindowDataset(synthetic_windows)
        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            history = trainer.fit(train_loader, val_loader, output_dir=tmpdir)

        assert "train_losses" in history
        assert "val_losses" in history
        assert len(history["train_losses"]) > 0
        assert len(history["val_losses"]) > 0

    def test_trainer_checkpoint(self, test_config, synthetic_windows):
        """Test checkpoint saving and loading."""
        model = MTGNN(
            num_nodes=19,
            num_targets=15,
            lookback_steps=96,
            horizon_steps=96,
            hidden_channels=test_config.model.hidden_channels,
        )
        trainer = Trainer(model, test_config, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            trainer._save_checkpoint(checkpoint_path, epoch=0)

            assert checkpoint_path.exists()

            # Load checkpoint
            trainer.load_checkpoint(checkpoint_path)

            # Verify state is loaded
            assert trainer.model is not None

    def test_trainer_evaluate(self, test_config, synthetic_windows):
        """Test evaluation on test set."""
        model = MTGNN(
            num_nodes=19,
            num_targets=15,
            lookback_steps=96,
            horizon_steps=96,
            hidden_channels=test_config.model.hidden_channels,
        )
        trainer = Trainer(model, test_config, device="cpu")

        dataset = ResidualWindowDataset(synthetic_windows)
        loader = DataLoader(dataset, batch_size=4)

        metrics = trainer.evaluate(loader)

        assert "test_loss" in metrics
        assert "mse" in metrics
        assert "mae" in metrics
        assert "per_target_mae" in metrics
        assert len(metrics["per_target_mae"]) == 15
