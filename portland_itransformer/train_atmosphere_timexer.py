#!/usr/bin/env python3
"""Atmospheric TimeXer (anomaly-based) training pipeline: 5 targets, 0-24h focus."""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from portland_itransformer.config import load_config
from portland_itransformer.preprocessing_atmosphere_timexer import (
    AtmosphereTimeXerPreprocessor,
    TARGET_ATMOSPHERE_TIMEXER,
    KNOWN_FEATURES_ATMOSPHERE,
)
from portland_itransformer.dataset import create_data_loaders
from portland_itransformer.models import MarineITransformer
from portland_itransformer.train import Trainer
from portland_itransformer.evaluate import Evaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)8s | %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run Atmospheric TimeXer training."""

    logger.info("=" * 100)
    logger.info("ATMOSPHERIC TIMEXER (ANOMALY-BASED) TRAINING PIPELINE")
    logger.info("=" * 100)

    # Load config
    logger.info("\n[1/6] Loading configuration...")
    config_path = project_root / "configs" / "atmosphere_timexer.yaml"
    config = load_config(str(config_path))

    logger.info(f"  Config: {config.project_name}")
    logger.info(f"  Targets: {len(TARGET_ATMOSPHERE_TIMEXER)} (anomaly-based)")
    logger.info(f"  Sequence: {config.data.seq_len} → {config.data.pred_len} steps (24h horizon)")

    # Preprocess
    logger.info("\n[2/6] Preprocessing atmospheric data...")
    preprocessor = AtmosphereTimeXerPreprocessor(config)
    preprocess_result = preprocessor.preprocess()

    logger.info(f"  ✓ Processed: {preprocess_result['num_samples']} rows")
    logger.info(f"  ✓ Split: {preprocess_result['split_counts']['train']} train, "
                f"{preprocess_result['split_counts']['validation']} val, "
                f"{preprocess_result['split_counts']['test']} test")

    # Create dataloaders
    logger.info("\n[3/6] Creating PyTorch DataLoaders...")
    df = pd.read_parquet(Path(config.paths.processed_dir) / "atmosphere_timexer_preprocessed.parquet")
    split_labels = np.load(Path(config.paths.processed_dir) / "atmosphere_timexer_split_labels.npy")

    loaders = create_data_loaders(
        df,
        split_labels,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        train_stride=config.data.train_stride_steps,
        eval_stride=config.data.eval_stride_steps,
        target_cols=TARGET_ATMOSPHERE_TIMEXER,
        known_cols=KNOWN_FEATURES_ATMOSPHERE,
    )

    logger.info(f"  Train: {len(loaders['train'].dataset)} windows")
    logger.info(f"  Validation: {len(loaders['validation'].dataset)} windows")
    logger.info(f"  Test: {len(loaders['test'].dataset)} windows")

    # Create & train model
    logger.info("\n[4/6] Creating and training model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"  Using device: {device}")

    # Atmospheric target loss weights (anomaly-based)
    atmosphere_weights = {
        'air_temp_anomaly_c': 1.0,
        'log1p_dewpoint_depression_c': 0.8,
        'air_pressure_anomaly_hpa': 0.8,
        'wind_u_anomaly_ms': 0.6,
        'wind_v_anomaly_ms': 0.6,
    }
    target_loss_weights = {
        f'target_{i}': atmosphere_weights.get(TARGET_ATMOSPHERE_TIMEXER[i], 1.0)
        for i in range(len(TARGET_ATMOSPHERE_TIMEXER))
    }

    model = MarineITransformer(
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        n_input_features=len(TARGET_ATMOSPHERE_TIMEXER) + len(KNOWN_FEATURES_ATMOSPHERE),
        n_target_features=len(TARGET_ATMOSPHERE_TIMEXER),
        n_future_known=len(KNOWN_FEATURES_ATMOSPHERE),
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        e_layers=config.model.e_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        use_instance_norm=config.model.use_instance_norm,
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model created: {num_params:,} parameters")

    output_dir = Path(config.paths.outputs_dir) / "atmosphere_timexer"
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model, config, target_loss_weights, device=device)

    logger.info(f"  Training for max {config.training.epochs} epochs...")
    train_result = trainer.fit(
        loaders['train'],
        loaders['validation'],
        output_dir
    )

    logger.info(f"  ✓ Training complete: {train_result['epochs_trained']} epochs")
    logger.info(f"  ✓ Best epoch: {train_result['best_epoch']} (val_loss: {train_result['best_val_loss']:.6f})")

    # Evaluate
    logger.info("\n[5/6] Evaluating on test set...")
    test_result = trainer.evaluate(loaders['test'])
    predictions_scaled = test_result['predictions']
    targets_scaled = test_result['targets']

    logger.info(f"  Test loss: {test_result['test_loss']:.6f}")
    logger.info(f"  Predictions shape: {predictions_scaled.shape}")

    # Metrics
    logger.info("\n[6/6] Computing metrics...")
    import joblib
    scaler_targets = joblib.load(
        Path(config.paths.artifacts_dir) / "atmosphere_timexer_scaler_targets.joblib"
    )
    evaluator = Evaluator(config, scaler_targets, output_dir)
    eval_result = evaluator.evaluate(
        predictions_scaled,
        targets_scaled,
        TARGET_ATMOSPHERE_TIMEXER,
        split_name='test'
    )

    metrics = eval_result['metrics']

    logger.info("\n" + "=" * 100)
    logger.info("ATMOSPHERIC TIMEXER FINAL RESULTS")
    logger.info("=" * 100)

    overall = metrics['overall']
    logger.info(f"\nOVERALL METRICS (Anomaly-Based, 0-24h horizon):")
    logger.info(f"  MAE (anomaly):   {overall['mae']:.6f}")
    logger.info(f"  RMSE (anomaly):  {overall['rmse']:.6f}")
    logger.info(f"  Skill vs Persistence: {overall['skill_vs_persistence']:.4f}")

    by_target = sorted(metrics['by_target'], key=lambda x: x['mae'])
    logger.info(f"\nBEST 3 TARGETS (lowest MAE):")
    for target_metric in by_target[:3]:
        logger.info(
            f"  {target_metric['target']:35s} MAE={target_metric['mae']:.6f}"
        )

    logger.info(f"\nWORST 3 TARGETS (highest MAE):")
    for target_metric in by_target[-3:]:
        logger.info(
            f"  {target_metric['target']:35s} MAE={target_metric['mae']:.6f}"
        )

    logger.info(f"\nHORIZON DEGRADATION (0-24h only):")
    for horizon_metric in metrics['by_horizon']:
        logger.info(
            f"  {horizon_metric['horizon_label']:10s} MAE={horizon_metric['mae']:.6f} "
            f"Skill={horizon_metric['skill']:.4f}"
        )

    logger.info("\n" + "=" * 100)
    logger.info("FILES SAVED")
    logger.info("=" * 100)
    logger.info(f"  Model checkpoint:     {output_dir / 'best_model.pt'}")
    logger.info(f"  Test metrics:         {output_dir / 'test_metrics_by_target.csv'}")
    logger.info(f"  Horizon metrics:      {output_dir / 'test_metrics_by_horizon.csv'}")

    logger.info("\n✅ Atmospheric TimeXer training complete!")
    return train_result, metrics


if __name__ == "__main__":
    try:
        train_result, metrics = main()
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        sys.exit(1)
