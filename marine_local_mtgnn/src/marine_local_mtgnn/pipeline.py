"""End-to-end pipeline for training and evaluation."""

import pandas as pd
from pathlib import Path
import json
import logging

from .data.validator import DataValidator
from .data.preprocessor import DataPreprocessor
from .baselines.selector import BaselineSelector
from .datasets.builder import DatasetBuilder
from .datasets.scalers import load_scaler
from .models import MTGNN
from .training.dataset import ResidualWindowDataset
from .training.trainer import Trainer
from .evaluation.metrics import compute_metrics, skill_score
from .evaluation.plotter import ResultsPlotter
from .config import Config

logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end MTGNN training and evaluation pipeline."""

    def __init__(self, config: Config):
        """
        Initialize pipeline.

        Parameters
        ----------
        config : Config
            Configuration object
        """
        self.config = config
        self.output_dir = Path(config.output_root)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        """
        Run complete pipeline: validate → preprocess → select baseline → create dataset → train → evaluate.

        Returns
        -------
        dict
            Results dictionary with all metrics and artifacts
        """
        logger.info("=" * 80)
        logger.info("MTGNN PIPELINE START")
        logger.info("=" * 80)

        # 1. Validate raw data
        logger.info("\n1. VALIDATION")
        validator = DataValidator(self.config)
        try:
            report = validator.validate()
            validator.save_reports(self.output_dir)
            logger.info(f"✓ Data validation passed")
        except Exception as e:
            logger.error(f"✗ Data validation failed: {e}")
            raise

        # 2. Preprocess data
        logger.info("\n2. PREPROCESSING")
        preprocessor = DataPreprocessor(self.config)
        try:
            splits = preprocessor.preprocess()
            preprocessor.save_splits(splits, self.output_dir)
            df_train = splits["train"]
            df_validation = splits["validation"]
            df_test = splits["test"]
            logger.info(f"✓ Preprocessing complete: train={len(df_train)}, val={len(df_validation)}, test={len(df_test)}")
        except Exception as e:
            logger.error(f"✗ Preprocessing failed: {e}")
            raise

        # 3. Select baseline
        logger.info("\n3. BASELINE SELECTION")
        selector = BaselineSelector(self.config)
        try:
            mae_results = selector.fit_all(df_train, df_validation)
            selector.save(self.output_dir)
            best_baseline = min(mae_results, key=mae_results.get)
            logger.info(f"✓ Selected baseline: {best_baseline} (val MAE: {mae_results[best_baseline]:.4f})")
        except Exception as e:
            logger.error(f"✗ Baseline selection failed: {e}")
            raise

        # 4. Create residual dataset
        logger.info("\n4. DATASET CREATION")
        try:
            builder = DatasetBuilder(self.config)
            dataset_result = builder.build(df_train, df_validation, df_test, selector)
            builder.save(self.output_dir)

            residual_data = dataset_result["residuals"]
            scaler = dataset_result["scaler"]

            train_dataset = ResidualWindowDataset(
                residual_data["train"]["windows"],
                scaler=scaler,
            )
            val_dataset = ResidualWindowDataset(
                residual_data["validation"]["windows"],
                scaler=scaler,
            )
            test_dataset = ResidualWindowDataset(
                residual_data["test"]["windows"],
                scaler=scaler,
            )

            logger.info(
                f"✓ Datasets created: "
                f"train={len(train_dataset)}, "
                f"val={len(val_dataset)}, "
                f"test={len(test_dataset)}"
            )
        except Exception as e:
            logger.error(f"✗ Dataset creation failed: {e}")
            raise

        # 5. Create model
        logger.info("\n5. MODEL CREATION")
        try:
            model = MTGNN(
                num_nodes=self.config.model.num_input_nodes,
                num_targets=self.config.model.num_direct_targets,
                lookback_steps=self.config.forecast.lookback_steps,
                horizon_steps=self.config.forecast.horizon_steps,
                hidden_channels=self.config.model.hidden_channels,
                skip_channels=self.config.model.skip_channels,
                end_channels=self.config.model.end_channels,
                kernel_size=self.config.model.kernel_size,
                dilation_exponential=self.config.model.dilation_exponential,
                dropout=self.config.model.dropout,
            )

            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"✓ Model created with {num_params:,} parameters")
        except Exception as e:
            logger.error(f"✗ Model creation failed: {e}")
            raise

        # 6. Train model
        logger.info("\n6. TRAINING")
        try:
            from torch.utils.data import DataLoader

            trainer = Trainer(model, self.config, device="cpu")

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                num_workers=self.config.training.num_workers,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                num_workers=self.config.training.num_workers,
            )

            training_result = trainer.fit(train_loader, val_loader, self.output_dir)

            logger.info(
                f"✓ Training complete: "
                f"epochs={training_result['epochs_trained']}, "
                f"best_val_loss={training_result['best_val_loss']:.6f}"
            )
        except Exception as e:
            logger.error(f"✗ Training failed: {e}")
            raise

        # 7. Evaluate
        logger.info("\n7. EVALUATION")
        try:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.training.batch_size,
                num_workers=self.config.training.num_workers,
            )

            test_metrics = trainer.evaluate(test_loader)

            logger.info(f"✓ Test metrics computed")
            logger.info(f"  - Test Loss: {test_metrics['test_loss']:.6f}")
            logger.info(f"  - MAE: {test_metrics['mae']:.6f}")
            logger.info(f"  - RMSE: {test_metrics['mse']**0.5:.6f}")
        except Exception as e:
            logger.error(f"✗ Evaluation failed: {e}")
            raise

        # 8. Generate plots
        logger.info("\n8. VISUALIZATION")
        try:
            plotter = ResultsPlotter(self.output_dir)

            plotter.plot_training_history(
                training_result["train_losses"],
                training_result["val_losses"],
            )

            logger.info(f"✓ Plots generated")
        except Exception as e:
            logger.warning(f"⚠ Plotting failed: {e}")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)

        results = {
            "data_validation": report,
            "baseline_selection": mae_results,
            "training": training_result,
            "test_metrics": test_metrics,
            "output_dir": str(self.output_dir),
        }

        # Save summary
        summary_path = self.output_dir / "pipeline_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Summary saved to {summary_path}")

        return results
