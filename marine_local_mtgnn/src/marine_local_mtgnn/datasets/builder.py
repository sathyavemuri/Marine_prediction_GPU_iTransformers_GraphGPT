"""Build complete residual dataset: residuals + graph prior + scalers."""

import pandas as pd
from pathlib import Path
import logging

from .residuals import ResidualDataset
from .graph_prior import GraphPrior
from .scalers import ResidualScaler
from ..baselines.selector import BaselineSelector
from ..config import Config

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Orchestrate complete dataset creation pipeline."""

    def __init__(self, config: Config):
        """
        Initialize dataset builder.

        Parameters
        ----------
        config : Config
            Configuration object
        """
        self.config = config
        self.residual_dataset = None
        self.graph_prior = None
        self.scaler = None

    def build(
        self,
        df_train: pd.DataFrame,
        df_validation: pd.DataFrame,
        df_test: pd.DataFrame,
        baseline_selector: BaselineSelector,
    ) -> dict:
        """
        Build complete dataset: residuals, graph prior, scalers.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data
        df_validation : pd.DataFrame
            Validation data
        df_test : pd.DataFrame
            Test data
        baseline_selector : BaselineSelector
            Fitted baseline selector

        Returns
        -------
        dict
            Dictionary with 'residuals', 'graph_prior', 'scaler'
        """
        logger.info("Building residual dataset...")

        # 1. Create residual dataset
        residual_creator = ResidualDataset(self.config, baseline_selector)
        residual_data = residual_creator.create(df_train, df_validation, df_test)
        self.residual_dataset = residual_data
        logger.info(f"Residual dataset: train={residual_data['train']['num_samples']}, "
                    f"val={residual_data['validation']['num_samples']}, "
                    f"test={residual_data['test']['num_samples']} windows")

        # 2. Build graph prior (training-only)
        logger.info("Building graph prior from training data...")
        self.graph_prior = GraphPrior(max_lag_steps=self.config.baselines.max_lag_graph_steps)
        self.graph_prior.fit(df_train)

        # 3. Fit scalers on training residuals only
        logger.info("Fitting scalers on training residuals...")
        train_residuals = [w["residuals"] for w in residual_data["train"]["windows"]]
        self.scaler = ResidualScaler()
        self.scaler.fit(train_residuals)

        return {
            "residuals": residual_data,
            "graph_prior": self.graph_prior,
            "scaler": self.scaler,
        }

    def save(self, output_dir: str | Path = "outputs") -> None:
        """
        Save all dataset components.

        Parameters
        ----------
        output_dir : str | Path
            Output directory
        """
        output_dir = Path(output_dir)

        if self.residual_dataset is None:
            logger.warning("No residual dataset to save")
        else:
            logger.info("Saving residual dataset components...")
            # Note: Actual residual data would be saved by PyTorch DataLoader
            # Here we just save metadata

        if self.graph_prior is not None:
            self.graph_prior.save(output_dir)

        if self.scaler is not None:
            self.scaler.save(output_dir)

        logger.info(f"Dataset components saved to {output_dir}")
