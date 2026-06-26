"""Residual dataset creation: baseline subtraction and windowing."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import logging
import json

from ..baselines.selector import BaselineSelector
from ..constants import NODE_NAMES, TARGET_NAMES
from ..config import Config

logger = logging.getLogger(__name__)


class ResidualDataset:
    """Create residual dataset for MTGNN training."""

    def __init__(self, config: Config, baseline_selector: BaselineSelector):
        """
        Initialize residual dataset creator.

        Parameters
        ----------
        config : Config
            Configuration object
        baseline_selector : BaselineSelector
            Fitted baseline selector with selected baseline
        """
        self.config = config
        self.baseline_selector = baseline_selector
        self.lookback_steps = config.forecast.lookback_steps
        self.horizon_steps = config.forecast.horizon_steps
        self.stride = config.forecast.sample_stride_steps

    def create(
        self,
        df_train: pd.DataFrame,
        df_validation: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> dict:
        """
        Create residual dataset for all splits.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data
        df_validation : pd.DataFrame
            Validation data
        df_test : pd.DataFrame
            Test data

        Returns
        -------
        dict
            Dictionary with 'train', 'validation', 'test' datasets
        """
        logger.info("Creating residual datasets...")

        # Create training residuals
        logger.info("Processing training split...")
        train_data = self._create_split_residuals(
            df_train,
            split_name="train",
            history_for_baseline=df_train,
        )

        # Create validation residuals (using train as history for baseline)
        logger.info("Processing validation split...")
        combined_train_val = pd.concat([df_train, df_validation], ignore_index=False)
        val_data = self._create_split_residuals(
            df_validation,
            split_name="validation",
            history_for_baseline=combined_train_val,
        )

        # Create test residuals (using train+val as history for baseline)
        logger.info("Processing test split...")
        combined_all = pd.concat([df_train, df_validation, df_test], ignore_index=False)
        test_data = self._create_split_residuals(
            df_test,
            split_name="test",
            history_for_baseline=combined_all,
        )

        return {
            "train": train_data,
            "validation": val_data,
            "test": test_data,
        }

    def _create_split_residuals(
        self,
        df_split: pd.DataFrame,
        split_name: str,
        history_for_baseline: pd.DataFrame,
    ) -> dict:
        """
        Create residuals for a single split.

        Parameters
        ----------
        df_split : pd.DataFrame
            Data for this split
        split_name : str
            Name of split (for logging)
        history_for_baseline : pd.DataFrame
            Full history for baseline (includes prior splits)

        Returns
        -------
        dict
            Dictionary with 'windows', 'baselines', 'targets'
        """
        windows = []
        baselines_list = []
        targets_list = []

        # Create windowed samples
        for origin in range(0, len(df_split) - self.horizon_steps, self.stride):
            # Get history up to this origin
            lookback_start = max(0, origin + len(history_for_baseline) - len(df_split) - self.lookback_steps)
            history_start_idx = max(0, len(history_for_baseline) - len(df_split) + origin - self.lookback_steps)
            history = history_for_baseline.iloc[history_start_idx : history_start_idx + self.lookback_steps]

            if len(history) < self.lookback_steps:
                # Skip if not enough history
                continue

            # Get actual future
            future_start = len(history_for_baseline) - len(df_split) + origin
            future_end = future_start + self.horizon_steps
            actual_future = history_for_baseline.iloc[future_start:future_end].values

            if len(actual_future) < self.horizon_steps:
                # Skip if not enough future
                continue

            # Generate baseline forecast
            try:
                baseline_forecast = self.baseline_selector.forecast(history, self.horizon_steps)
            except Exception as e:
                logger.warning(f"Failed to generate baseline forecast at origin {origin}: {e}")
                continue

            # Compute residuals (only for targets)
            # Get target indices
            target_indices = [NODE_NAMES.index(target) for target in TARGET_NAMES]
            actual_targets = actual_future[:, target_indices]
            baseline_targets = baseline_forecast[:, target_indices]
            residuals = actual_targets - baseline_targets

            # Store window
            window_data = {
                "history": history.values,
                "baseline_forecast": baseline_forecast,
                "actual_targets": actual_targets,
                "residuals": residuals,
            }

            windows.append(window_data)
            baselines_list.append(baseline_forecast)
            targets_list.append(actual_targets)

        logger.info(f"{split_name}: Created {len(windows)} windows")

        return {
            "windows": windows,
            "baselines": baselines_list,
            "targets": targets_list,
            "num_samples": len(windows),
        }

    def save(self, output_dir: str | Path = "outputs") -> None:
        """
        Save residual dataset metadata.

        Parameters
        ----------
        output_dir : str | Path
            Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save dataset metadata
        metadata = {
            "lookback_steps": self.lookback_steps,
            "horizon_steps": self.horizon_steps,
            "stride": self.stride,
            "input_nodes": NODE_NAMES,
            "target_nodes": TARGET_NAMES,
            "num_input_nodes": len(NODE_NAMES),
            "num_target_nodes": len(TARGET_NAMES),
        }

        json_path = output_dir / "residual_dataset_metadata.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved dataset metadata: {json_path}")
