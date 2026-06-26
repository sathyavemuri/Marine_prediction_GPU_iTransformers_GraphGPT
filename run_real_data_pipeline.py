#!/usr/bin/env python3
"""Run MTGNN pipeline on real 120-day marine buoy data."""

import sys
import os
from pathlib import Path
import logging
import json
import numpy as np
import pandas as pd

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "marine_local_mtgnn" / "src"))

from marine_local_mtgnn.config import load_config
from marine_local_mtgnn.pipeline import Pipeline
from marine_local_mtgnn.evaluation.metrics import compute_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run full pipeline and generate 10-day accuracy results."""

    logger.info("=" * 100)
    logger.info("MTGNN PIPELINE: REAL MARINE DATA (120 days)")
    logger.info("=" * 100)

    # Load config
    config_path = project_root / "marine_local_mtgnn" / "configs" / "local_15min_7day.yaml"
    logger.info(f"\nLoading config: {config_path}")
    config = load_config(str(config_path))

    logger.info(f"  Site: {config.site.buoy_id}")
    logger.info(f"  Location: {config.site.latitude}°N, {abs(config.site.longitude)}°W")
    logger.info(f"  Timezone: {config.site.timezone}")
    logger.info(f"  Data: {config.data.raw_csv}")

    # Run pipeline
    logger.info("\nExecuting full pipeline...")
    pipeline = Pipeline(config)
    results = pipeline.run()

    logger.info("\n" + "=" * 100)
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info("=" * 100)

    # Print results summary
    logger.info("\nTraining Summary:")
    logger.info(f"  Epochs trained: {results['training']['epochs_trained']}")
    logger.info(f"  Best validation loss: {results['training']['best_val_loss']:.6f}")

    logger.info("\nTest Metrics:")
    test_metrics = results['test_metrics']
    logger.info(f"  Test MAE: {test_metrics['mae']:.6f}")
    logger.info(f"  Test RMSE: {np.sqrt(test_metrics['mse']):.6f}")
    logger.info(f"  Test Loss: {test_metrics['test_loss']:.6f}")

    logger.info("\nBaseline Selection:")
    for baseline, mae in results['baseline_selection'].items():
        logger.info(f"  {baseline}: MAE = {mae:.6f}")

    logger.info(f"\nOutput directory: {results['output_dir']}")

    return results

if __name__ == "__main__":
    try:
        results = main()
        logger.info("\n[SUCCESS] Pipeline complete!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n[ERROR] Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
