"""Scaling pipeline: fit on training only, transform all splits."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class ScalingPipeline:
    """
    Fit scalers on training data only, transform all splits.

    Two separate scalers:
    1. target_scaler: StandardScaler for 13 targets
    2. known_scaler: StandardScaler for 2 continuous known features

    Prevents test/validation leakage by fitting only on training data.
    """

    def __init__(self, config):
        """Initialize scaling pipeline."""
        self.config = config
        self.target_scaler = StandardScaler()
        self.known_scaler = StandardScaler()
        self.fit_indices = {
            'target_scaler': None,
            'known_scaler': None,
        }
        self.feature_stats = {
            'target': {},
            'known': {},
        }

    def fit(
        self,
        df: pd.DataFrame,
        split_labels: np.ndarray,
        target_features: list,
        known_features: list,
    ):
        """
        Fit scalers on training data only.

        Parameters
        ----------
        df : DataFrame
            Full preprocessed dataframe
        split_labels : array
            Array of split labels: 0=train, 1=val, 2=test
        target_features : list
            13 target feature names
        known_features : list
            2 known feature names (tide_baseline_m, clear_sky_radiation_wm2)
        """
        logger.info("Fitting scalers on training data only...")

        # Get training indices (split_labels == 0)
        train_mask = split_labels == 0
        df_train = df.iloc[train_mask]

        if df_train.shape[0] < 10:
            raise ValueError(f"Not enough training samples: {df_train.shape[0]}")

        # Fit target scaler
        target_data = df_train[target_features].values
        self.target_scaler.fit(target_data)
        self.fit_indices['target_scaler'] = np.where(train_mask)[0]

        logger.info(f"  ✓ Target scaler fit on {len(df_train)} training samples")
        logger.info(f"    Targets: {target_features}")

        # Store target stats
        for i, feat in enumerate(target_features):
            self.feature_stats['target'][feat] = {
                'mean': float(self.target_scaler.mean_[i]),
                'std': float(self.target_scaler.scale_[i]),
            }

        # Fit known scaler
        if known_features:
            known_data = df_train[known_features].values
            self.known_scaler.fit(known_data)
            self.fit_indices['known_scaler'] = np.where(train_mask)[0]

            logger.info(f"  ✓ Known scaler fit on {len(df_train)} training samples")
            logger.info(f"    Known features: {known_features}")

            # Store known stats
            for i, feat in enumerate(known_features):
                self.feature_stats['known'][feat] = {
                    'mean': float(self.known_scaler.mean_[i]),
                    'std': float(self.known_scaler.scale_[i]),
                }

    def transform(
        self,
        df: pd.DataFrame,
        target_features: list,
        known_features: list,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform all features using fitted scalers.

        Parameters
        ----------
        df : DataFrame
            Dataframe to transform
        target_features : list
            13 target feature names
        known_features : list
            2 known feature names

        Returns
        -------
        df_targets_scaled : DataFrame
            Scaled targets [n_rows, 13]
        df_known_scaled : DataFrame
            Scaled known features [n_rows, 2]
        """
        if self.target_scaler.mean_ is None:
            raise ValueError("Scalers not fitted. Call fit() first.")

        # Transform targets
        target_data = df[target_features].values
        target_scaled = self.target_scaler.transform(target_data)
        df_targets_scaled = pd.DataFrame(target_scaled, columns=target_features)

        # Transform known
        known_scaled = None
        df_known_scaled = None
        if known_features:
            known_data = df[known_features].values
            known_scaled = self.known_scaler.transform(known_data)
            df_known_scaled = pd.DataFrame(known_scaled, columns=known_features)

        return df_targets_scaled, df_known_scaled

    def inverse_transform_targets(
        self,
        targets_scaled: np.ndarray,
    ) -> np.ndarray:
        """
        Inverse transform targets back to original scale.

        Parameters
        ----------
        targets_scaled : array
            [n_samples, 13] scaled targets

        Returns
        -------
        targets : array
            [n_samples, 13] original scale
        """
        if self.target_scaler.mean_ is None:
            raise ValueError("Target scaler not fitted.")

        return self.target_scaler.inverse_transform(targets_scaled)

    def inverse_transform_known(
        self,
        known_scaled: np.ndarray,
    ) -> np.ndarray:
        """
        Inverse transform known features back to original scale.

        Parameters
        ----------
        known_scaled : array
            [n_samples, 2] scaled known features

        Returns
        -------
        known : array
            [n_samples, 2] original scale
        """
        if self.known_scaler.mean_ is None:
            raise ValueError("Known scaler not fitted.")

        return self.known_scaler.inverse_transform(known_scaled)

    def save(self, artifacts_dir: Path):
        """
        Save scalers and metadata to disk.

        Parameters
        ----------
        artifacts_dir : Path
            Directory to save artifacts
        """
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save scalers
        target_scaler_path = artifacts_dir / 'target_scaler.joblib'
        joblib.dump(self.target_scaler, target_scaler_path)
        logger.info(f"Saved target scaler: {target_scaler_path}")

        known_scaler_path = artifacts_dir / 'known_scaler.joblib'
        joblib.dump(self.known_scaler, known_scaler_path)
        logger.info(f"Saved known scaler: {known_scaler_path}")

        # Save metadata
        metadata = {
            'target_scaler': {
                'means': self.target_scaler.mean_.tolist(),
                'stds': self.target_scaler.scale_.tolist(),
                'features': list(self.feature_stats['target'].keys()),
            },
            'known_scaler': {
                'means': self.known_scaler.mean_.tolist(),
                'stds': self.known_scaler.scale_.tolist(),
                'features': list(self.feature_stats['known'].keys()),
            },
            'feature_stats': self.feature_stats,
            'fit_indices': {
                'target_scaler': self.fit_indices['target_scaler'].tolist()
                    if self.fit_indices['target_scaler'] is not None else None,
                'known_scaler': self.fit_indices['known_scaler'].tolist()
                    if self.fit_indices['known_scaler'] is not None else None,
            },
        }

        metadata_path = artifacts_dir / 'scaling_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved scaling metadata: {metadata_path}")

    def load(self, artifacts_dir: Path):
        """
        Load scalers from disk.

        Parameters
        ----------
        artifacts_dir : Path
            Directory containing saved artifacts
        """
        artifacts_dir = Path(artifacts_dir)

        # Load scalers
        target_scaler_path = artifacts_dir / 'target_scaler.joblib'
        if target_scaler_path.exists():
            self.target_scaler = joblib.load(target_scaler_path)
            logger.info(f"Loaded target scaler: {target_scaler_path}")
        else:
            logger.warning(f"Target scaler not found: {target_scaler_path}")

        known_scaler_path = artifacts_dir / 'known_scaler.joblib'
        if known_scaler_path.exists():
            self.known_scaler = joblib.load(known_scaler_path)
            logger.info(f"Loaded known scaler: {known_scaler_path}")
        else:
            logger.warning(f"Known scaler not found: {known_scaler_path}")

        # Load metadata
        metadata_path = artifacts_dir / 'scaling_metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.feature_stats = json.load(f).get('feature_stats', {})
            logger.info(f"Loaded scaling metadata: {metadata_path}")

    def get_feature_stats(self) -> Dict:
        """Get statistics of fitted scalers."""
        return {
            'target': self.feature_stats.get('target', {}),
            'known': self.feature_stats.get('known', {}),
        }

    def verify_leakage(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> bool:
        """
        Verify no validation/test data used in fitting.

        Parameters
        ----------
        train_indices : array
            Training data indices
        val_indices : array
            Validation data indices
        test_indices : array
            Test data indices

        Returns
        -------
        is_valid : bool
            True if no leakage detected
        """
        fit_train = set(self.fit_indices['target_scaler'])
        fit_indices = set(self.fit_indices['target_scaler'])
        train_set = set(train_indices)

        # Check if fit indices are subset of training
        if not fit_indices.issubset(train_set):
            logger.error("ERROR: Scaler fit on non-training data!")
            return False

        val_set = set(val_indices)
        test_set = set(test_indices)

        if fit_indices.intersection(val_set):
            logger.error("ERROR: Scaler fit includes validation data!")
            return False

        if fit_indices.intersection(test_set):
            logger.error("ERROR: Scaler fit includes test data!")
            return False

        logger.info("✓ Scaling: No leakage detected")
        return True
