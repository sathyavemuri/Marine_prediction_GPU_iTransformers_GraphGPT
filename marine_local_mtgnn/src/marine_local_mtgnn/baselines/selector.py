"""Baseline selector: fit candidates and select based on validation performance."""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional
import json

from .persistence import PersistenceBaseline, SeasonalBaseline
from .seasonal import DailySeasonalBaseline, WeeklySeasonalBaseline
from .trend import TrendBaseline
from ..config import Config

logger = logging.getLogger(__name__)


class BaselineSelector:
    """Fit baseline candidates and select best based on validation MAE."""

    def __init__(self, config: Config):
        """
        Initialize baseline selector.

        Parameters
        ----------
        config : Config
            Configuration object
        """
        self.config = config
        self.baselines = {}
        self.selected_baseline = None
        self.validation_mae = {}

    def fit_all(self, df_train: pd.DataFrame, df_validation: pd.DataFrame) -> dict[str, float]:
        """
        Fit all baseline candidates and select best on validation.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data
        df_validation : pd.DataFrame
            Validation data

        Returns
        -------
        dict[str, float]
            Validation MAE for each baseline
        """
        logger.info("Fitting baseline candidates...")

        # Initialize baselines
        candidates = {}

        if self.config.baselines.persistence:
            candidates["persistence"] = PersistenceBaseline()

        if self.config.baselines.daily_seasonal:
            candidates["daily_seasonal"] = DailySeasonalBaseline(cadence_minutes=15)

        if self.config.baselines.weekly_seasonal:
            candidates["weekly_seasonal"] = WeeklySeasonalBaseline(cadence_minutes=15)

        if self.config.baselines.local_trend:
            candidates["trend"] = TrendBaseline(window_steps=self.config.baselines.trend_window_steps)

        # Fit all candidates
        for name, baseline in candidates.items():
            try:
                baseline.fit(df_train)
                logger.debug(f"Fitted baseline: {name}")
            except Exception as e:
                logger.warning(f"Failed to fit baseline {name}: {e}")
                continue

            self.baselines[name] = baseline

        # Evaluate on validation
        logger.info(f"Evaluating {len(self.baselines)} baselines on validation data...")
        self.validation_mae = self._evaluate_baselines(df_validation, df_train)

        # Select best
        if self.validation_mae:
            best_name = min(self.validation_mae, key=self.validation_mae.get)
            self.selected_baseline = self.baselines[best_name]
            logger.info(f"Selected baseline: {best_name} (validation MAE: {self.validation_mae[best_name]:.4f})")
        else:
            raise ValueError("No baselines successfully fitted")

        return self.validation_mae

    def _evaluate_baselines(
        self,
        df_validation: pd.DataFrame,
        df_train: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Evaluate each baseline on validation data.

        Parameters
        ----------
        df_validation : pd.DataFrame
            Validation data
        df_train : pd.DataFrame
            Training data (for creating history)

        Returns
        -------
        dict[str, float]
            Validation MAE for each baseline
        """
        mae_dict = {}
        horizon_steps = self.config.forecast.horizon_steps

        # Use training data as initial history
        lookback_steps = self.config.forecast.lookback_steps
        history_start = max(0, len(df_train) - lookback_steps)
        history = df_train.iloc[history_start:].copy()

        # Validation origins (rolling window)
        stride = self.config.forecast.sample_stride_steps
        origins = list(range(0, len(df_validation) - horizon_steps, stride))

        if not origins:
            origins = [0]

        for baseline_name, baseline in self.baselines.items():
            mae_list = []

            for origin in origins:
                # Get history up to this origin
                current_history = pd.concat([history, df_validation.iloc[:origin]], ignore_index=False)
                current_history = current_history.iloc[max(0, len(current_history) - lookback_steps):]

                # Get actual future
                future_start = origin
                future_end = min(origin + horizon_steps, len(df_validation))
                actual_future = df_validation.iloc[future_start:future_end].values

                if future_end - future_start < horizon_steps:
                    # Pad with NaN if not enough future data
                    padded = np.full((horizon_steps, actual_future.shape[1]), np.nan)
                    padded[: future_end - future_start] = actual_future
                    actual_future = padded

                try:
                    # Forecast
                    forecast = baseline.forecast(current_history, horizon_steps)

                    # Compute MAE
                    mae = np.nanmean(np.abs(forecast - actual_future))
                    mae_list.append(mae)
                except Exception as e:
                    logger.debug(f"Error forecasting with {baseline_name} at origin {origin}: {e}")
                    mae_list.append(np.inf)

            if mae_list:
                avg_mae = np.nanmean([m for m in mae_list if np.isfinite(m)])
                mae_dict[baseline_name] = float(avg_mae)
            else:
                mae_dict[baseline_name] = np.inf

        return mae_dict

    def forecast(self, df_history: pd.DataFrame, horizon_steps: int) -> np.ndarray:
        """
        Generate forecast using selected baseline.

        Parameters
        ----------
        df_history : pd.DataFrame
            History data
        horizon_steps : int
            Number of future steps

        Returns
        -------
        np.ndarray
            Forecast of shape (horizon_steps, n_targets)
        """
        if self.selected_baseline is None:
            raise ValueError("No baseline selected; must call fit_all first")

        return self.selected_baseline.forecast(df_history, horizon_steps)

    def save(self, output_dir: str | Path = "outputs") -> None:
        """
        Save baseline metadata and selection results.

        Parameters
        ----------
        output_dir : str | Path
            Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save validation results
        results = {
            "selected_baseline": type(self.selected_baseline).__name__ if self.selected_baseline else None,
            "validation_mae": self.validation_mae,
        }

        json_path = output_dir / "baseline_selection.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved baseline selection results: {json_path}")
