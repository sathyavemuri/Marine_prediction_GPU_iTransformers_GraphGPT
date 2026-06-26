"""Data preprocessing: resampling and splitting."""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Literal

from .loader import load_raw_csv
from .transforms import TransformPipeline
from ..config import Config

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess raw data: transform, resample, split."""

    def __init__(self, config: Config):
        """
        Initialize preprocessor.

        Parameters
        ----------
        config : Config
            Configuration object
        """
        self.config = config
        self.transform_pipeline = TransformPipeline(
            wind_direction_convention=config.site.wind_direction_convention,
            current_direction_convention=config.site.current_direction_convention,
        )

    def preprocess(self) -> dict[Literal["train", "validation", "test"], pd.DataFrame]:
        """
        Preprocess raw data: load, transform, resample, split.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary with "train", "validation", "test" DataFrames
        """
        # Load raw CSV
        logger.info(f"Loading raw CSV: {self.config.data.raw_csv}")
        df = load_raw_csv(
            self.config.data.raw_csv,
            timezone=self.config.site.timezone,
        )
        logger.info(f"Loaded {len(df)} rows with cadence ~1 minute")

        # Transform
        logger.info("Applying transforms: direction → u/v, log transforms, circular encoding")
        df = self.transform_pipeline.transform(df)
        logger.info(f"Transformed to {len(df.columns)} columns")

        # Resample
        logger.info(f"Resampling to {self.config.data.resample_rule}")
        df = self._resample(df)
        logger.info(f"Resampled to {len(df)} rows")

        # Split
        logger.info("Splitting into train/validation/test")
        splits = self._split(df)
        for split_name, split_df in splits.items():
            logger.info(f"  {split_name}: {len(split_df)} rows ({len(split_df) * 15 / 1440:.1f} days)")

        return splits

    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample to target cadence.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with datetime index

        Returns
        -------
        pd.DataFrame
            Resampled data
        """
        # Resample with aggregation strategy (mean for most, forward-fill for categorical)
        df_resampled = df.resample(
            self.config.data.resample_rule,
            label=self.config.data.resample_label,
            closed=self.config.data.resample_closed,
        ).mean()

        # Forward fill NaNs introduced by resampling
        df_resampled = df_resampled.ffill()

        # Drop any remaining NaNs at the start
        df_resampled = df_resampled.dropna()

        return df_resampled

    def _split(self, df: pd.DataFrame) -> dict[Literal["train", "validation", "test"], pd.DataFrame]:
        """
        Split into chronological train/validation/test.

        Parameters
        ----------
        df : pd.DataFrame
            Full data

        Returns
        -------
        dict[str, pd.DataFrame]
            Splits
        """
        splits_config = self.config.splits

        # Parse split boundaries (ensure UTC)
        train_start = pd.Timestamp(splits_config.train_start, tz="UTC")
        train_end = pd.Timestamp(splits_config.train_end_exclusive, tz="UTC")
        val_start = pd.Timestamp(splits_config.validation_start, tz="UTC")
        val_end = pd.Timestamp(splits_config.validation_end_exclusive, tz="UTC")
        test_start = pd.Timestamp(splits_config.test_start, tz="UTC")
        test_end = pd.Timestamp(splits_config.test_end_exclusive, tz="UTC")

        # Ensure index is UTC for comparison
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")

        # Extract splits
        train = df[(idx >= train_start) & (idx < train_end)]
        validation = df[(idx >= val_start) & (idx < val_end)]
        test = df[(idx >= test_start) & (idx < test_end)]

        # Validate
        if len(train) == 0 or len(validation) == 0 or len(test) == 0:
            raise ValueError(
                f"Empty split: train={len(train)}, validation={len(validation)}, test={len(test)}"
            )

        # Check for overlap
        train_idx = train.index.tz_convert("UTC") if train.index.tz else train.index.tz_localize("UTC")
        val_idx = validation.index.tz_convert("UTC") if validation.index.tz else validation.index.tz_localize("UTC")
        test_idx = test.index.tz_convert("UTC") if test.index.tz else test.index.tz_localize("UTC")

        if len(set(train_idx) & set(val_idx)) > 0:
            raise ValueError("Train and validation splits overlap")
        if len(set(val_idx) & set(test_idx)) > 0:
            raise ValueError("Validation and test splits overlap")

        return {"train": train, "validation": validation, "test": test}

    def save_splits(self, splits: dict[str, pd.DataFrame], output_dir: str | Path = "outputs") -> None:
        """
        Save split data to parquet files.

        Parameters
        ----------
        splits : dict[str, pd.DataFrame]
            Splits dictionary
        output_dir : str | Path
            Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_df in splits.items():
            path = output_dir / f"{split_name}_split.parquet"
            split_df.to_parquet(path)
            logger.info(f"Saved {split_name} split: {path}")
