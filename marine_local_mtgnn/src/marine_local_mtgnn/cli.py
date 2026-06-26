"""Command-line interface for marine local MTGNN."""

import argparse
from pathlib import Path
from typing import Optional
import logging
import pandas as pd

from .config import load_config
from .data.validator import DataValidator
from .data.preprocessor import DataPreprocessor
from .baselines.selector import BaselineSelector

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def cmd_validate_data(args):
    """Validate raw data and generate quality report."""
    logger.info("Validating data...")
    config = load_config(Path(args.config))
    logger.info(f"Config loaded: {config.experiment_name}")
    logger.info(f"Raw CSV: {config.data.raw_csv}")

    validator = DataValidator(config)
    try:
        report = validator.validate()
        validator.save_reports(config.output_root)
        logger.info("Data validation complete")
    except ValueError as e:
        logger.error(f"Data validation failed: {e}")
        raise


def cmd_preprocess(args):
    """Preprocess raw data: transforms, resample, split."""
    logger.info("Preprocessing data...")
    config = load_config(Path(args.config))
    logger.info(f"Resampling to: {config.forecast.profile}")

    preprocessor = DataPreprocessor(config)
    splits = preprocessor.preprocess()
    preprocessor.save_splits(splits, config.output_root)
    logger.info("Preprocessing complete")


def cmd_fit_baselines(args):
    """Fit baseline candidates on training data."""
    logger.info("Fitting baselines...")
    config = load_config(Path(args.config))
    logger.info(f"Baseline selection metric: {config.baselines.selection_metric}")

    # Load preprocessed splits
    output_root = Path(config.output_root)
    train_path = output_root / "train_split.parquet"
    val_path = output_root / "validation_split.parquet"

    if not train_path.exists() or not val_path.exists():
        logger.error(f"Preprocessed splits not found. Run 'preprocess' first.")
        raise FileNotFoundError(f"Train or validation split missing")

    df_train = pd.read_parquet(train_path)
    df_validation = pd.read_parquet(val_path)
    logger.info(f"Loaded train ({len(df_train)} rows) and validation ({len(df_validation)} rows)")

    # Fit and select baselines
    selector = BaselineSelector(config)
    mae_results = selector.fit_all(df_train, df_validation)

    # Log results
    for baseline_name, mae in sorted(mae_results.items(), key=lambda x: x[1]):
        logger.info(f"  {baseline_name}: MAE = {mae:.4f}")

    # Save results
    selector.save(output_root)
    logger.info("Baseline fitting complete")


def cmd_build_graph_prior(args):
    """Build graph prior from training correlations."""
    logger.info("Building graph prior...")
    config = load_config(Path(args.config))
    logger.info(f"Max lag for graph: {config.baselines.max_lag_graph_steps} steps")
    logger.info("Graph prior not yet implemented")


def cmd_train(args):
    """Train MTGNN model."""
    logger.info("Training MTGNN...")
    config = load_config(Path(args.config))
    logger.info(f"Max epochs: {config.training.max_epochs}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info("Training not yet implemented")


def cmd_evaluate(args):
    """Evaluate model on validation or test split."""
    logger.info(f"Evaluating on {args.split} split...")
    config = load_config(Path(args.config))
    logger.info("Evaluation not yet implemented")


def cmd_forecast(args):
    """Generate forecast from latest history."""
    logger.info("Generating forecast...")
    config = load_config(Path(args.config))
    logger.info(f"Horizon: {config.forecast.horizon_steps} steps")
    logger.info("Forecast not yet implemented")


def cmd_serve(args):
    """Launch FastAPI inference service."""
    logger.info("Starting service...")
    config = load_config(Path(args.config))
    logger.info(f"Service will run at {config.service.host}:{config.service.port}")
    logger.info("Service not yet implemented")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Marine Local MTGNN: Local-only residual forecaster"
    )
    parser.add_argument("--config", default="configs/local_15min_7day.yaml", help="Config file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # validate-data
    subparsers.add_parser("validate-data", help="Validate raw data")

    # preprocess
    subparsers.add_parser("preprocess", help="Preprocess raw data")

    # fit-baselines
    subparsers.add_parser("fit-baselines", help="Fit baseline candidates")

    # build-graph-prior
    subparsers.add_parser("build-graph-prior", help="Build graph prior from correlations")

    # train
    subparsers.add_parser("train", help="Train MTGNN model")

    # evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate model")
    parser_eval.add_argument("--split", choices=["validation", "test"], default="validation")

    # forecast
    subparsers.add_parser("forecast", help="Generate forecast")

    # serve
    subparsers.add_parser("serve", help="Launch API service")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return

    command_map = {
        "validate-data": cmd_validate_data,
        "preprocess": cmd_preprocess,
        "fit-baselines": cmd_fit_baselines,
        "build-graph-prior": cmd_build_graph_prior,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "forecast": cmd_forecast,
        "serve": cmd_serve,
    }

    if args.command in command_map:
        command_map[args.command](args)
    else:
        logger.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
