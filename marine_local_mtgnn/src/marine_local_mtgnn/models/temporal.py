"""Temporal encoding with dilated convolutions."""

import torch
import torch.nn as nn


class DilatedConvBlock(nn.Module):
    """Residual block with dilated convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        """
        Initialize dilated convolution block.

        Parameters
        ----------
        in_channels : int
            Input channels
        out_channels : int
            Output channels
        kernel_size : int
            Convolution kernel size
        dilation : int
            Dilation rate
        dropout : float
            Dropout rate
        """
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Projection for residual connection if channels differ
        if in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, in_channels, time_steps)

        Returns
        -------
        torch.Tensor
            Output of shape (batch, out_channels, time_steps)
        """
        residual = x if self.proj is None else self.proj(x)

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = out + residual
        return out


class TemporalEncoder(nn.Module):
    """Multi-layer temporal encoder with dilated convolutions."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_blocks: int = 9,
        kernel_size: int = 3,
        dilation_exponential: int = 2,
        dropout: float = 0.0,
    ):
        """
        Initialize temporal encoder.

        Parameters
        ----------
        input_size : int
            Input size (num nodes)
        hidden_size : int
            Hidden channels
        num_blocks : int
            Number of residual blocks
        kernel_size : int
            Convolution kernel size
        dilation_exponential : int
            Base for dilation growth (2 → 1,2,4,8,...)
        dropout : float
            Dropout rate
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks

        # Input projection
        self.input_proj = nn.Conv1d(input_size, hidden_size, 1)

        # Dilated blocks
        blocks = []
        for i in range(num_blocks):
            dilation = dilation_exponential ** i
            blocks.append(
                DilatedConvBlock(
                    hidden_size,
                    hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode temporal patterns.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, nodes, time_steps)

        Returns
        -------
        torch.Tensor
            Encoded of shape (batch, hidden_size, time_steps)
        """
        # Transpose for Conv1d: (batch, nodes, time) → (batch, nodes, time)
        # Conv1d expects (batch, channels, length)
        x = x.permute(0, 2, 1)  # (batch, time, nodes)
        x = x.permute(0, 2, 1)  # Back to (batch, nodes, time)

        # Actually keep as (batch, nodes, time) and treat nodes as channels
        x = x.permute(0, 1, 2)  # (batch, nodes, time)

        # Input projection
        x = self.input_proj(x)  # (batch, hidden, time)

        # Process through blocks
        x = self.blocks(x)  # (batch, hidden, time)

        return x


class TemporalDecoder(nn.Module):
    """Decoder for direct multi-horizon output."""

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        horizon_steps: int,
        target_embedding_dim: int = 8,
        lead_time_embedding_dim: int = 16,
        dropout: float = 0.0,
    ):
        """
        Initialize temporal decoder.

        Parameters
        ----------
        hidden_size : int
            Hidden size from encoder
        output_size : int
            Number of targets (15 for this project)
        horizon_steps : int
            Number of forecast steps (672)
        target_embedding_dim : int
            Embedding dimension for target
        lead_time_embedding_dim : int
            Embedding dimension for lead time
        dropout : float
            Dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.horizon_steps = horizon_steps

        # Target embeddings (one per target)
        self.target_embedding = nn.Embedding(output_size, target_embedding_dim)

        # Lead time embeddings (one per forecast step)
        self.lead_time_embedding = nn.Embedding(horizon_steps, lead_time_embedding_dim)

        # Decoder layers
        embed_total = target_embedding_dim + lead_time_embedding_dim
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + embed_total, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: decode to all targets and horizons.

        Parameters
        ----------
        encoded : torch.Tensor
            Encoded features from temporal encoder
            Shape: (batch, hidden_size, time_steps)

        Returns
        -------
        torch.Tensor
            Forecast of shape (batch, horizon_steps, output_size)
        """
        batch_size = encoded.shape[0]

        # Pool over time (take last timestep)
        context = encoded[:, :, -1]  # (batch, hidden_size)

        # Initialize output
        output = torch.zeros(
            batch_size,
            self.horizon_steps,
            self.output_size,
            device=encoded.device,
        )

        # For each target and horizon, predict value
        for target_idx in range(self.output_size):
            # Target embedding
            target_emb = self.target_embedding(
                torch.tensor(target_idx, device=encoded.device)
            )  # (target_embedding_dim,)
            target_emb = target_emb.unsqueeze(0).expand(batch_size, -1)  # (batch, target_embedding_dim)

            for lead_idx in range(self.horizon_steps):
                # Lead time embedding
                lead_emb = self.lead_time_embedding(
                    torch.tensor(lead_idx, device=encoded.device)
                )  # (lead_time_embedding_dim,)
                lead_emb = lead_emb.unsqueeze(0).expand(batch_size, -1)  # (batch, lead_time_embedding_dim)

                # Concatenate context + embeddings
                features = torch.cat([context, target_emb, lead_emb], dim=1)  # (batch, hidden + embed_total)

                # Predict
                pred = self.decoder(features)  # (batch, 1)
                output[:, lead_idx, target_idx] = pred.squeeze(-1)

        return output
