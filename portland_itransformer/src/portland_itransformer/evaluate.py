"""Test evaluation and reporting."""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import logging
from typing import Dict
import matplotlib.pyplot as plt

from .metrics import (
    compute_metrics,
    compute_baseline_metrics,
    create_metric_dataframe,
    create_horizon_dataframe,
    print_metrics_summary,
)
from .baselines import get_baseline_forecasts

logger = logging.getLogger(__name__)


class Evaluator:
    """Test evaluation orchestrator."""

    def __init__(self, config, scaling_pipeline, output_dir: Path):
        """
        Initialize evaluator.

        Parameters
        ----------
        config : Config
            Configuration object
        scaling_pipeline : ScalingPipeline
            Fitted scaler for inverse transform
        output_dir : Path
            Directory for saving results
        """
        self.config = config
        self.scaler = scaling_pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        predictions_scaled: np.ndarray,
        actuals_scaled: np.ndarray,
        target_names: list,
        split_name: str = 'test',
    ) -> Dict:
        """
        Evaluate test predictions.

        Parameters
        ----------
        predictions_scaled : array
            [num_samples, horizon, num_targets] scaled predictions
        actuals_scaled : array
            [num_samples, horizon, num_targets] scaled actuals
        target_names : list
            13 target variable names
        split_name : str
            'test' or 'validation'

        Returns
        -------
        dict
            Results dictionary with metrics
        """
        logger.info("\n" + "=" * 100)
        logger.info(f"EVALUATION: {split_name.upper()}")
        logger.info("=" * 100)

        # Inverse transform to physical scale
        num_samples, horizon, num_targets = predictions_scaled.shape

        # Reshape for inverse transform
        pred_flat = predictions_scaled.reshape(-1, num_targets)
        actual_flat = actuals_scaled.reshape(-1, num_targets)

        predictions_physical = self.scaler.inverse_transform(pred_flat)
        actuals_physical = self.scaler.inverse_transform(actual_flat)

        predictions_physical = predictions_physical.reshape(num_samples, horizon, num_targets)
        actuals_physical = actuals_physical.reshape(num_samples, horizon, num_targets)

        logger.info(f"Predictions (physical): shape {predictions_physical.shape}")
        logger.info(f"Actuals (physical): shape {actuals_physical.shape}")

        # Compute metrics on physical scale
        metrics = compute_metrics(predictions_physical, actuals_physical, target_names)
        print_metrics_summary(metrics, split_name)

        # Save metrics
        self._save_metrics(metrics, split_name)

        # Create comparison visualizations
        self._create_plots(
            predictions_physical,
            actuals_physical,
            target_names,
            split_name,
        )

        return {
            'metrics': metrics,
            'predictions_physical': predictions_physical,
            'actuals_physical': actuals_physical,
        }

    def _save_metrics(self, metrics: Dict, split_name: str):
        """Save metrics to CSV and JSON."""
        # By-target metrics
        df_target = create_metric_dataframe(metrics, split_name)
        target_path = self.output_dir / f'{split_name}_metrics_by_target.csv'
        df_target.to_csv(target_path, index=False)
        logger.info(f"Saved: {target_path}")

        # By-horizon metrics
        df_horizon = create_horizon_dataframe(metrics, split_name)
        horizon_path = self.output_dir / f'{split_name}_metrics_by_horizon.csv'
        df_horizon.to_csv(horizon_path, index=False)
        logger.info(f"Saved: {horizon_path}")

        # Overall metrics
        overall_path = self.output_dir / f'{split_name}_metrics_overall.txt'
        with open(overall_path, 'w') as f:
            f.write(f"Overall Metrics - {split_name.upper()}\n")
            f.write("=" * 50 + "\n\n")
            for key, value in metrics['overall'].items():
                f.write(f"{key:30s}: {value:.6f}\n")
        logger.info(f"Saved: {overall_path}")

    def _create_plots(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        target_names: list,
        split_name: str,
    ):
        """Create diagnostic plots."""
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        # Plot 1: Error by lead time
        self._plot_error_by_horizon(predictions, actuals, split_name, plots_dir)

        # Plot 2: Error by target
        self._plot_error_by_target(predictions, actuals, target_names, split_name, plots_dir)

        # Plot 3: Sample forecasts (3 random samples)
        self._plot_sample_forecasts(
            predictions, actuals, target_names, split_name, plots_dir
        )

    def _plot_error_by_horizon(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        split_name: str,
        plots_dir: Path,
    ):
        """Plot MAE degradation over forecast horizon."""
        horizon = predictions.shape[1]
        mae_by_step = np.nanmean(np.abs(predictions - actuals), axis=(0, 2))

        fig, ax = plt.subplots(figsize=(12, 6))
        hours = np.arange(horizon) * 0.25  # 15-minute cadence
        ax.plot(hours, mae_by_step, 'b-o', linewidth=2, markersize=4)
        ax.fill_between(hours, mae_by_step, alpha=0.3)

        ax.set_xlabel('Lead Time (hours)', fontsize=11)
        ax.set_ylabel('MAE', fontsize=11)
        ax.set_title(f'Forecast Error Degradation - {split_name.upper()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = plots_dir / f'{split_name}_error_by_horizon.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {path}")

    def _plot_error_by_target(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        target_names: list,
        split_name: str,
        plots_dir: Path,
    ):
        """Plot MAE by target variable."""
        mae_by_target = np.nanmean(np.abs(predictions - actuals), axis=(0, 1))

        fig, ax = plt.subplots(figsize=(14, 6))
        x_pos = np.arange(len(target_names))
        ax.bar(x_pos, mae_by_target, alpha=0.7, color='steelblue')

        ax.set_xlabel('Target Variable', fontsize=11)
        ax.set_ylabel('MAE', fontsize=11)
        ax.set_title(f'Error by Target - {split_name.upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(target_names, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        path = plots_dir / f'{split_name}_error_by_target.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {path}")

    def _plot_sample_forecasts(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        target_names: list,
        split_name: str,
        plots_dir: Path,
    ):
        """Plot 3 sample forecast time series."""
        # Select 3 random samples
        num_samples = predictions.shape[0]
        sample_indices = np.random.choice(num_samples, 3, replace=False)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        for idx, sample_idx in enumerate(sample_indices):
            # Select first target (air_temp)
            target_idx = 0
            pred_sample = predictions[sample_idx, :, target_idx]
            actual_sample = actuals[sample_idx, :, target_idx]

            hours = np.arange(len(pred_sample)) * 0.25

            ax = axes[idx]
            ax.plot(hours, actual_sample, 'b-o', label='Actual', linewidth=2, markersize=4)
            ax.plot(hours, pred_sample, 'r--s', label='Forecast', linewidth=2, markersize=4)
            ax.fill_between(
                hours,
                actual_sample,
                pred_sample,
                alpha=0.2,
                color='gray',
                label='Error'
            )

            ax.set_xlabel('Lead Time (hours)', fontsize=10)
            ax.set_ylabel(target_names[target_idx], fontsize=10)
            ax.set_title(f'Sample {idx+1}: {target_names[target_idx]}', fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Sample Forecasts - {split_name.upper()}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        path = plots_dir / f'{split_name}_sample_forecasts.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {path}")


def run_full_evaluation(
    trainer,
    test_loader,
    config,
    scaler,
    target_names: list,
    output_dir: Path,
):
    """
    Run full test evaluation.

    Parameters
    ----------
    trainer : Trainer
        Trained model in trainer
    test_loader : DataLoader
        Test data loader
    config : Config
        Configuration
    scaler : ScalingPipeline
        Fitted scaler
    target_names : list
        13 target names
    output_dir : Path
        Output directory
    """
    logger.info("\n" + "=" * 100)
    logger.info("FULL TEST EVALUATION")
    logger.info("=" * 100)

    # Run inference on test set
    test_result = trainer.evaluate(test_loader)

    predictions = test_result['predictions']
    targets = test_result['targets']

    logger.info(f"Test predictions shape: {predictions.shape}")
    logger.info(f"Test targets shape: {targets.shape}")

    # Create evaluator
    evaluator = Evaluator(config, scaler, output_dir)

    # Evaluate (inverse transform happens inside)
    eval_result = evaluator.evaluate(
        predictions,
        targets,
        target_names,
        split_name='test'
    )

    logger.info("\n✓ Full evaluation complete")
    logger.info(f"Results saved to: {output_dir}")

    return eval_result
