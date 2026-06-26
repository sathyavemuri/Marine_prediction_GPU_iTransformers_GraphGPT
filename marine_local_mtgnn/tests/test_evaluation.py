"""Tests for evaluation metrics."""

import pytest
import numpy as np

from marine_local_mtgnn.evaluation.metrics import compute_metrics, skill_score


@pytest.fixture
def synthetic_predictions():
    """Create synthetic predictions and actuals."""
    np.random.seed(42)
    num_samples = 10
    horizon = 96
    num_targets = 15

    # Perfect forecast plus small noise
    actuals = np.random.randn(num_samples, horizon, num_targets)
    predictions = actuals + np.random.randn(num_samples, horizon, num_targets) * 0.1

    target_names = [f"target_{i}" for i in range(num_targets)]

    return predictions, actuals, target_names


class TestMetrics:
    """Test metrics computation."""

    def test_compute_metrics_shape(self, synthetic_predictions):
        """Test metrics output structure."""
        predictions, actuals, target_names = synthetic_predictions

        metrics = compute_metrics(predictions, actuals, target_names)

        assert "overall" in metrics
        assert "by_target" in metrics
        assert "by_horizon" in metrics

    def test_compute_metrics_overall(self, synthetic_predictions):
        """Test overall metrics computation."""
        predictions, actuals, target_names = synthetic_predictions

        metrics = compute_metrics(predictions, actuals, target_names)
        overall = metrics["overall"]

        assert "mse" in overall
        assert "rmse" in overall
        assert "mae" in overall
        assert "skill_vs_persistence" in overall

        # All metrics should be positive
        assert overall["mse"] > 0
        assert overall["rmse"] > 0
        assert overall["mae"] > 0

    def test_compute_metrics_per_target(self, synthetic_predictions):
        """Test per-target metrics."""
        predictions, actuals, target_names = synthetic_predictions

        metrics = compute_metrics(predictions, actuals, target_names)
        per_target = metrics["by_target"]

        assert len(per_target) == 15
        for target_metric in per_target:
            assert "target" in target_metric
            assert "mae" in target_metric
            assert "rmse" in target_metric

    def test_compute_metrics_per_horizon(self, synthetic_predictions):
        """Test per-horizon metrics."""
        predictions, actuals, target_names = synthetic_predictions

        metrics = compute_metrics(predictions, actuals, target_names)
        per_horizon = metrics["by_horizon"]

        assert len(per_horizon) == 96
        for horizon_metric in per_horizon:
            assert "lead_time_steps" in horizon_metric
            assert "mae" in horizon_metric
            assert "rmse" in horizon_metric

    def test_skill_score_perfect(self):
        """Test skill score with perfect forecast."""
        # Create actuals with some trend (not constant)
        actuals = np.random.randn(10, 50, 5) + np.linspace(0, 1, 50).reshape(1, 50, 1)
        predictions = actuals.copy()

        skill = skill_score(predictions, actuals)

        assert np.isclose(skill, 1.0, atol=1e-6)  # Perfect forecast

    def test_skill_score_persistence(self):
        """Test skill score with persistence baseline."""
        actuals = np.random.randn(10, 50, 5)
        horizon = actuals.shape[1]

        # Use persistence as prediction
        predictions = np.repeat(actuals[:, :1, :], horizon, axis=1)

        skill = skill_score(predictions, actuals)

        assert skill == 0.0  # Same as persistence baseline

    def test_skill_score_worse_than_persistence(self):
        """Test skill score when worse than persistence."""
        actuals = np.random.randn(10, 50, 5)
        horizon = actuals.shape[1]

        # Persistence baseline
        persistence = np.repeat(actuals[:, :1, :], horizon, axis=1)

        # Random predictions (much worse than persistence)
        predictions = np.random.randn(10, horizon, 5) * 10

        skill = skill_score(predictions, actuals)

        # Should be negative (worse than persistence)
        assert skill < 0
