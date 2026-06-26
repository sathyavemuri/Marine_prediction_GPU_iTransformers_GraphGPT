"""PyTorch Dataset for windowed time series forecasting."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ForecastWindowDataset(Dataset):
    """
    Sliding window dataset for marine forecasting.

    Creates windows of:
    - history: [lookback_steps, n_input_features]
    - future_known: [pred_len, n_future_known]
    - targets: [pred_len, n_target_features]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split_labels: np.ndarray,
        split_id: int,
        seq_len: int = 1344,
        pred_len: int = 672,
        stride_steps: int = 4,
        target_cols: List[str] = None,
        known_cols: List[str] = None,
    ):
        """
        Initialize dataset.

        Parameters
        ----------
        df : DataFrame
            Preprocessed data with timestamp + scaled features
        split_labels : array
            [n_samples] with values 0=train, 1=val, 2=test
        split_id : int
            Which split to use (0, 1, or 2)
        seq_len : int
            History length in steps (default 1344 = 14 days)
        pred_len : int
            Forecast horizon in steps (default 672 = 7 days)
        stride_steps : int
            Window stride (default 4 = 1 hour)
        target_cols : list
            13 target feature names
        known_cols : list
            6 known feature names (tide_baseline, clear_sky, hour_sin/cos, dayofyear_sin/cos)
        """
        self.df = df.reset_index(drop=True)
        self.split_labels = split_labels
        self.split_id = split_id
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride_steps = stride_steps

        # Default feature names
        if target_cols is None:
            target_cols = [
                'air_temp_c', 'air_pressure_hpa', 'water_temp_c', 'dew_point_c',
                'salinity_psu', 'wind_u_east_ms', 'wind_v_north_ms',
                'current_u_east_ms', 'current_v_north_ms', 'tidal_residual_m',
                'log_significant_wave_height_m', 'log_zero_crossing_period_s',
                'log_clearness_index',
            ]
        if known_cols is None:
            known_cols = [
                'tide_baseline_m', 'clear_sky_radiation_wm2',
                'hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos',
            ]

        self.target_cols = target_cols
        self.known_cols = known_cols
        self.n_target = len(target_cols)
        self.n_known = len(known_cols)

        # Create windows for this split
        self.windows = self._create_windows()

        logger.info(f"Created {len(self.windows)} windows for split {split_id}")

    def _create_windows(self) -> List[Dict]:
        """
        Create sliding windows for this split.

        Returns
        -------
        windows : list[dict]
            List of window dicts with keys:
            - 'start_idx': start index in full dataframe
            - 'origin_idx': index of forecast origin
            - 'end_idx': end index (exclusive)
        """
        windows = []

        # Get indices for this split
        split_mask = self.split_labels == self.split_id
        split_indices = np.where(split_mask)[0]

        if len(split_indices) == 0:
            logger.warning(f"No data for split {self.split_id}")
            return windows

        # Generate windows within this split
        for origin_idx in split_indices[::self.stride_steps]:
            start_idx = origin_idx - self.seq_len
            end_idx = origin_idx + self.pred_len

            # Check boundaries
            if start_idx < 0:
                continue
            if end_idx > len(self.df):
                continue

            # Ensure all target steps are in the correct split
            target_indices = np.arange(origin_idx, origin_idx + self.pred_len)
            target_splits = self.split_labels[target_indices]

            if not np.all(target_splits == self.split_id):
                # Target horizon spans multiple splits
                continue

            windows.append({
                'start_idx': start_idx,
                'origin_idx': origin_idx,
                'end_idx': end_idx,
            })

        return windows

    def __len__(self) -> int:
        """Return number of windows."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single window.

        Parameters
        ----------
        idx : int
            Window index

        Returns
        -------
        dict
            Keys:
            - 'x_past': [seq_len, n_input] history features
            - 'x_future_known': [pred_len, n_known] known future features
            - 'y_target': [pred_len, n_target] target values
            - 'y_mask': [pred_len, n_target] valid value mask
            - 'origin_timestamp': timestamp of forecast origin
        """
        window = self.windows[idx]
        start_idx = window['start_idx']
        origin_idx = window['origin_idx']
        end_idx = window['end_idx']

        # Extract history (x_past)
        history_data = self.df.iloc[start_idx:origin_idx, 1:]  # Skip timestamp
        x_past = history_data[self.target_cols + self.known_cols].values.astype(np.float32)

        if x_past.shape[0] != self.seq_len:
            raise RuntimeError(
                f"History shape mismatch: {x_past.shape[0]} != {self.seq_len}"
            )

        # Extract future known (x_future_known)
        future_known_data = self.df.iloc[origin_idx:end_idx, 1:]
        x_future_known = future_known_data[self.known_cols].values.astype(np.float32)

        if x_future_known.shape[0] != self.pred_len:
            raise RuntimeError(
                f"Future known shape mismatch: {x_future_known.shape[0]} != {self.pred_len}"
            )

        # Extract targets (y_target)
        target_data = self.df.iloc[origin_idx:end_idx, 1:]
        y_target = target_data[self.target_cols].values.astype(np.float32)

        if y_target.shape[0] != self.pred_len:
            raise RuntimeError(
                f"Target shape mismatch: {y_target.shape[0]} != {self.pred_len}"
            )

        # Create mask (all valid for synthetic data)
        y_mask = (~np.isnan(y_target)).astype(np.float32)

        # Get timestamp
        origin_timestamp = pd.Timestamp(self.df.iloc[origin_idx, 0])
        origin_timestamp_ns = origin_timestamp.value  # nanoseconds since epoch

        return {
            'x_past': torch.FloatTensor(x_past),
            'x_future_known': torch.FloatTensor(x_future_known),
            'y_target': torch.FloatTensor(y_target),
            'y_mask': torch.FloatTensor(y_mask),
            'origin_timestamp': origin_timestamp_ns,
        }

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            'num_windows': len(self.windows),
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_target_features': self.n_target,
            'n_known_features': self.n_known,
            'n_input_features': self.n_target + self.n_known,
            'stride_steps': self.stride_steps,
        }


def create_data_loaders(
    df: pd.DataFrame,
    split_labels: np.ndarray,
    batch_size: int = 16,
    num_workers: int = 2,
    seq_len: int = 1344,
    pred_len: int = 672,
    train_stride: int = 4,
    eval_stride: int = 96,
    target_cols: List[str] = None,
    known_cols: List[str] = None,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train/val/test DataLoaders.

    Parameters
    ----------
    df : DataFrame
        Preprocessed data
    split_labels : array
        Split assignment labels
    batch_size : int
        Batch size
    num_workers : int
        Number of worker processes
    seq_len : int
        Lookback length
    pred_len : int
        Forecast horizon
    train_stride : int
        Stride for training windows (1 hour)
    eval_stride : int
        Stride for validation/test windows (1 day)
    target_cols : list
        Target column names
    known_cols : list
        Known column names

    Returns
    -------
    dict
        DataLoaders for 'train', 'validation', 'test'
    """
    datasets = {}
    loaders = {}

    # Training dataset
    datasets['train'] = ForecastWindowDataset(
        df, split_labels, split_id=0,
        seq_len=seq_len, pred_len=pred_len,
        stride_steps=train_stride,
        target_cols=target_cols, known_cols=known_cols,
    )

    loaders['train'] = torch.utils.data.DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Validation dataset
    datasets['validation'] = ForecastWindowDataset(
        df, split_labels, split_id=1,
        seq_len=seq_len, pred_len=pred_len,
        stride_steps=eval_stride,
        target_cols=target_cols, known_cols=known_cols,
    )

    loaders['validation'] = torch.utils.data.DataLoader(
        datasets['validation'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test dataset
    datasets['test'] = ForecastWindowDataset(
        df, split_labels, split_id=2,
        seq_len=seq_len, pred_len=pred_len,
        stride_steps=eval_stride,
        target_cols=target_cols, known_cols=known_cols,
    )

    loaders['test'] = torch.utils.data.DataLoader(
        datasets['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Log statistics
    logger.info("Dataset Statistics:")
    for split in ['train', 'validation', 'test']:
        stats = datasets[split].get_statistics()
        logger.info(f"  {split.upper()}: {stats['num_windows']} windows")

    return loaders
