#!/usr/bin/env python3
"""Complete end-to-end training pipeline: preprocess → train → evaluate."""

import sys
import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from portland_itransformer.config import load_config
from portland_itransformer.preprocess import Preprocessor
from portland_itransformer.dataset import create_data_loaders
from portland_itransformer.models import MarineITransformer
from portland_itransformer.train import Trainer
from portland_itransformer.evaluate import Evaluator
from portland_itransformer.reconstruct import OutputReconstructor
from portland_itransformer.constants import TARGET_FEATURES, KNOWN_FEATURES
from portland_itransformer.metrics import print_metrics_summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete training pipeline."""

    logger.info("=" * 100)
    logger.info("PORTLAND ITRANSFORMER - COMPLETE TRAINING PIPELINE")
    logger.info("=" * 100)

    # ========== PHASE 1: LOAD CONFIG ==========
    logger.info("\n[1/6] Loading configuration...")
    config_path = project_root / "configs" / "portland_7day.yaml"
    config = load_config(str(config_path))

    logger.info(f"  Config: {config.project_name}")
    logger.info(f"  Site: {config.site.latitude}°N, {abs(config.site.longitude)}°W")
    logger.info(f"  Sequence: {config.data.seq_len} → {config.data.pred_len} steps")

    # ========== PHASE 2: PREPROCESS ==========
    logger.info("\n[2/6] Preprocessing raw data...")
    preprocessor = Preprocessor(config)
    preprocess_result = preprocessor.preprocess()

    logger.info(f"  ✓ Processed: {preprocess_result['num_samples']} rows × {preprocess_result['num_features']} features")
    logger.info(f"  ✓ Split: {preprocess_result['split_counts']['train']} train, "
                f"{preprocess_result['split_counts']['validation']} val, "
                f"{preprocess_result['split_counts']['test']} test")

    # ========== PHASE 3: LOAD DATA & CREATE DATALOADERS ==========
    logger.info("\n[3/6] Creating PyTorch DataLoaders...")
    df = pd.read_parquet(Path(config.paths.processed_dir) / "portland_preprocessed.parquet")
    split_labels = np.load(Path(config.paths.processed_dir) / "split_labels.npy")

    loaders = create_data_loaders(
        df,
        split_labels,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        train_stride=config.data.train_stride_steps,
        eval_stride=config.data.eval_stride_steps,
        target_cols=TARGET_FEATURES,
        known_cols=KNOWN_FEATURES,
    )

    # ========== PHASE 4: CREATE & TRAIN MODEL ==========
    logger.info("\n[4/6] Creating and training model...")

    # Target loss weights
    target_loss_weights = {
        'target_0': 1.0,   # air_temp_c
        'target_1': 1.0,   # air_pressure_hpa
        'target_2': 1.0,   # water_temp_c
        'target_3': 1.0,   # dew_point_c
        'target_4': 1.0,   # salinity_psu
        'target_5': 1.0,   # wind_u_east_ms
        'target_6': 1.0,   # wind_v_north_ms
        'target_7': 1.0,   # current_u_east_ms
        'target_8': 1.0,   # current_v_north_ms
        'target_9': 0.8,   # tidal_residual_m
        'target_10': 1.2,  # log_significant_wave_height_m
        'target_11': 1.2,  # log_zero_crossing_period_s
        'target_12': 1.0,  # log_clearness_index
    }

    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"  Using device: {device}")

    model = MarineITransformer(
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        n_input_features=len(TARGET_FEATURES) + len(KNOWN_FEATURES),
        n_target_features=len(TARGET_FEATURES),
        n_future_known=len(KNOWN_FEATURES),
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        e_layers=config.model.e_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        use_instance_norm=config.model.use_instance_norm,
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model created: {num_params:,} parameters")

    # Train
    output_dir = Path(config.paths.outputs_dir)
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

    # ========== PHASE 5: EVALUATE ==========
    logger.info("\n[5/6] Evaluating on test set...")

    # Load scalers and calibrators
    from portland_itransformer.scaling import ScalingPipeline
    from portland_itransformer.calibrators import DerivedCalibrators
    from portland_itransformer.baselines import UTideBaseline, ClearSkyBaseline

    scaler = ScalingPipeline(config)
    scaler.load(Path(config.paths.artifacts_dir))

    calibrators = DerivedCalibrators()
    calibrators.load(Path(config.paths.artifacts_dir))

    utide_baseline = UTideBaseline(config)
    utide_baseline.load(Path(config.paths.artifacts_dir))

    clearsky_baseline = ClearSkyBaseline(config)
    clearsky_baseline.fit(pd.DatetimeIndex(df['timestamp']))

    # Evaluate
    evaluator = Evaluator(config, scaler, output_dir)
    test_result = trainer.evaluate(loaders['test'])

    predictions_scaled = test_result['predictions']
    targets_scaled = test_result['targets']

    logger.info(f"  Predictions shape: {predictions_scaled.shape}")
    logger.info(f"  Test loss: {test_result['test_loss']:.6f}")

    # ========== PHASE 6: REPORT RESULTS ==========
    logger.info("\n[6/6] Generating results and metrics...")

    # Evaluate (inverse scales and computes metrics)
    eval_result = evaluator.evaluate(
        predictions_scaled,
        targets_scaled,
        TARGET_FEATURES,
        split_name='test'
    )

    metrics = eval_result['metrics']

    logger.info("\n" + "=" * 100)
    logger.info("FINAL RESULTS")
    logger.info("=" * 100)

    # Overall metrics
    logger.info("\nOVERALL METRICS:")
    overall = metrics['overall']
    logger.info(f"  MAE:                  {overall['mae']:.6f}")
    logger.info(f"  RMSE:                 {overall['rmse']:.6f}")
    logger.info(f"  Skill (vs Persistence): {overall['skill_vs_persistence']:.4f}")

    # Best 5 parameters
    by_target = sorted(metrics['by_target'], key=lambda x: x['mae'])
    logger.info("\nBEST 5 PARAMETERS (lowest MAE):")
    for i, target_metric in enumerate(by_target[:5]):
        logger.info(
            f"  {i+1}. {target_metric['target']:30s} "
            f"MAE={target_metric['mae']:.6f} "
            f"RMSE={target_metric['rmse']:.6f}"
        )

    # Worst 5 parameters
    logger.info("\nWORST 5 PARAMETERS (highest MAE):")
    for i, target_metric in enumerate(by_target[-5:]):
        logger.info(
            f"  {i+1}. {target_metric['target']:30s} "
            f"MAE={target_metric['mae']:.6f} "
            f"RMSE={target_metric['rmse']:.6f}"
        )

    # Horizon degradation
    logger.info("\nHORIZON DEGRADATION:")
    for horizon_metric in metrics['by_horizon']:
        logger.info(
            f"  {horizon_metric['horizon_label']:10s} "
            f"MAE={horizon_metric['mae']:.6f} "
            f"Skill={horizon_metric['skill']:.4f}"
        )

    # Training curves
    logger.info("\nTRAINING HISTORY:")
    logger.info(f"  Initial train loss: {train_result['train_losses'][0]:.6f}")
    logger.info(f"  Final train loss:   {train_result['train_losses'][-1]:.6f}")
    logger.info(f"  Best val loss:      {train_result['best_val_loss']:.6f}")

    # Calculate improvement
    train_improvement = (train_result['train_losses'][0] - train_result['train_losses'][-1]) / train_result['train_losses'][0]
    logger.info(f"  Training improvement: {train_improvement*100:.1f}%")

    logger.info("\n" + "=" * 100)
    logger.info("FILES SAVED")
    logger.info("=" * 100)
    logger.info(f"  Model checkpoint:     {output_dir / 'best_model.pt'}")
    logger.info(f"  Training history:     {output_dir / 'training_history.json'}")
    logger.info(f"  Test metrics:         {output_dir / 'test_metrics_by_target.csv'}")
    logger.info(f"  Horizon metrics:      {output_dir / 'test_metrics_by_horizon.csv'}")
    logger.info(f"  Plots:                {output_dir / 'plots'} (3 PNG files)")

    logger.info("\n" + "=" * 100)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 100)

    # Summary for user
    logger.info("\n📊 SUMMARY:")
    logger.info(f"  • Model trained for {train_result['epochs_trained']} epochs")
    logger.info(f"  • Best validation loss at epoch {train_result['best_epoch']}")
    logger.info(f"  • Test MAE: {metrics['overall']['mae']:.6f}")
    logger.info(f"  • Test Skill: {metrics['overall']['skill_vs_persistence']:.4f}")
    logger.info(f"  • Best parameter: {by_target[0]['target']} (MAE={by_target[0]['mae']:.6f})")
    logger.info(f"  • Worst parameter: {by_target[-1]['target']} (MAE={by_target[-1]['mae']:.6f})")

    return {
        'train_result': train_result,
        'metrics': metrics,
        'output_dir': str(output_dir),
    }


if __name__ == "__main__":
    try:
        result = main()
        logger.info("\n✅ Pipeline completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
