"""MarineITransformer: iTransformer-based marine forecasting model."""

import torch
import torch.nn as nn
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class MarineITransformer(nn.Module):
    """
    iTransformer for marine forecasting.

    Inverted Transformer approach: each variable/channel is a token containing
    its full historical time sequence. Attention operates over channels, not time.
    """

    def __init__(
        self,
        seq_len: int = 1344,
        pred_len: int = 672,
        n_input_features: int = 19,
        n_target_features: int = 13,
        n_future_known: int = 6,
        target_indices: list = None,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.20,
        activation: str = "gelu",
        use_instance_norm: bool = True,
    ):
        """
        Initialize MarineITransformer.

        Parameters
        ----------
        seq_len : int
            Input sequence length (history)
        pred_len : int
            Prediction length (forecast horizon)
        n_input_features : int
            Number of input features (19: targets + known)
        n_target_features : int
            Number of target features (13)
        n_future_known : int
            Number of future known features (6)
        target_indices : list
            Indices of target features in input features
        d_model : int
            Model hidden dimension
        n_heads : int
            Number of attention heads
        e_layers : int
            Number of encoder layers
        d_ff : int
            Feed-forward dimension
        dropout : float
            Dropout rate
        activation : str
            Activation function ('gelu' or 'relu')
        use_instance_norm : bool
            Whether to use per-window instance normalization
        """
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_input_features = n_input_features
        self.n_target_features = n_target_features
        self.n_future_known = n_future_known
        self.d_model = d_model
        self.use_instance_norm = use_instance_norm

        if target_indices is None:
            target_indices = list(range(n_target_features))
        self.target_indices = torch.tensor(target_indices, dtype=torch.long)

        # Token embedding: project each channel's time series to d_model
        # Input shape for each token: [seq_len] -> output: [d_model]
        self.token_embedding = nn.Linear(seq_len, d_model)

        # Transformer encoder (operates over channel dimension, not time)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Forecast head: project each target token to prediction horizon
        # Input: [d_model] -> output: [pred_len]
        self.forecast_head = nn.Linear(d_model, pred_len)

        # Known-future covariate head: MLP that conditions on future tide/solar/calendar
        # Input: [n_future_known] -> output: [n_target_features]
        self.future_head = nn.Sequential(
            nn.Linear(n_future_known, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_target_features),
        )

        self.dropout = nn.Dropout(dropout)

        logger.info(f"MarineITransformer initialized: "
                   f"{n_input_features} inputs, {n_target_features} targets, "
                   f"seq_len={seq_len}, pred_len={pred_len}")

    def forward(
        self,
        x_past: torch.Tensor,
        x_future_known: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x_past : Tensor
            [batch, seq_len, n_input_features]
            Scaled input features (targets + known covariates)
        x_future_known : Tensor
            [batch, pred_len, n_future_known]
            Known future covariates (tide, radiation, calendar)

        Returns
        -------
        y_pred : Tensor
            [batch, pred_len, n_target_features]
            Predicted residuals/transformed targets
        """
        batch_size = x_past.shape[0]
        assert x_past.shape == (batch_size, self.seq_len, self.n_input_features)
        assert x_future_known.shape == (batch_size, self.pred_len, self.n_future_known)

        # Instance normalization (per window, per feature)
        if self.use_instance_norm:
            x_norm = self._instance_normalize(x_past)
            # Retain target feature means/stds for inverse normalization later
            target_mean = x_past[:, :, :self.n_target_features].mean(dim=1, keepdim=True)
            target_std = x_past[:, :, :self.n_target_features].std(dim=1, keepdim=True)
        else:
            x_norm = x_past
            target_mean = None
            target_std = None

        # Inverted embedding: each feature becomes a token
        # [batch, seq_len, n_input] -> [batch, n_input, seq_len]
        x_transposed = x_norm.permute(0, 2, 1)

        # Project each token (feature) from time dimension to d_model
        # [batch, n_input, seq_len] -> [batch, n_input, d_model]
        tokens = self.token_embedding(x_transposed)

        # Transformer encoder operates over token dimension (features)
        # [batch, n_input, d_model] -> [batch, n_input, d_model]
        encoded = self.encoder(tokens)

        # Select target tokens only
        # [batch, n_target, d_model]
        device = encoded.device
        target_indices = self.target_indices.to(device)
        target_tokens = encoded[:, target_indices, :]

        # Forecast head: project to prediction horizon
        # [batch, n_target, d_model] -> [batch, n_target, pred_len]
        y_igtransformer = self.forecast_head(target_tokens)

        # Transpose to [batch, pred_len, n_target]
        y_igtransformer = y_igtransformer.permute(0, 2, 1)

        # Future covariate head: add learned conditioning on known future values
        # [batch, pred_len, n_future_known] -> [batch, pred_len, n_target]
        y_future = self.future_head(x_future_known)

        # Combine iTransformer output + future covariate contribution
        y_pred = y_igtransformer + y_future

        # Inverse instance normalization for target features
        if self.use_instance_norm and target_mean is not None:
            y_pred_unnorm = y_pred * (target_std + 1e-8) + target_mean
            return y_pred_unnorm
        else:
            return y_pred

    def _instance_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-window instance normalization.

        Parameters
        ----------
        x : Tensor
            [batch, seq_len, n_features]

        Returns
        -------
        x_norm : Tensor
            [batch, seq_len, n_features], normalized per window/feature
        """
        # Compute statistics over time dimension (dim=1)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8

        # Normalize
        x_norm = (x - mean) / std

        return x_norm

    def get_adjacency(self) -> torch.Tensor:
        """
        Get learned feature adjacency (attention weights).

        Returns
        -------
        adjacency : Tensor
            [n_input_features, n_input_features]
            Normalized attention weights or graph structure
        """
        # This is a placeholder for graph structure learning
        # In a full implementation, could extract attention weights from encoder
        # For now, return identity or learned adjacency matrix if added
        return torch.eye(self.n_input_features)
