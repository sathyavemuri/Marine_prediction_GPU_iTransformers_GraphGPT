"""Graph convolutional layers with learned and fixed adjacency."""

import torch
import torch.nn as nn
import numpy as np


class GraphConvolution(nn.Module):
    """Graph convolution with learned weight matrix."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        """
        Initialize graph convolution.

        Parameters
        ----------
        in_features : int
            Input feature dimension
        out_features : int
            Output feature dimension
        bias : bool
            Whether to use bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x' = adj @ x @ W + b.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch, nodes, in_features)
        adj : torch.Tensor
            Adjacency matrix of shape (nodes, nodes)

        Returns
        -------
        torch.Tensor
            Output of shape (batch, nodes, out_features)
        """
        # x: (batch, nodes, in_features)
        # W: (in_features, out_features)
        # adj: (nodes, nodes)

        # Linear transformation
        x = torch.matmul(x, self.weight)  # (batch, nodes, out_features)

        # Apply adjacency
        x = torch.matmul(adj, x)  # (batch, nodes, out_features)

        if self.bias is not None:
            x = x + self.bias

        return x


class AdaptiveAdjacency(nn.Module):
    """Learn sparse adjacency matrix with fixed prior."""

    def __init__(
        self,
        num_nodes: int,
        prior_adj: np.ndarray | None = None,
        top_k: int = 4,
        temperature: float = 1.0,
    ):
        """
        Initialize adaptive adjacency.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in graph
        prior_adj : np.ndarray, optional
            Prior adjacency matrix (sparse, top-k)
        top_k : int
            Number of neighbors to keep (sparsity)
        temperature : float
            Temperature for softmax
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.top_k = top_k
        self.temperature = temperature

        # Initialize with learned parameters
        self.adj_weights = nn.Parameter(torch.ones(num_nodes, num_nodes) / num_nodes)

        # Prior adjacency (fixed, not learned)
        if prior_adj is not None:
            self.register_buffer("prior_adj", torch.FloatTensor(prior_adj))
        else:
            self.register_buffer("prior_adj", torch.eye(num_nodes))

    def forward(self) -> torch.Tensor:
        """
        Compute sparse adjacency matrix.

        Returns
        -------
        torch.Tensor
            Adjacency matrix of shape (num_nodes, num_nodes)
        """
        # Soft attention over top-k
        adj = self.adj_weights / self.temperature

        # Keep only top-k + prior
        if self.top_k > 0:
            # For each node, keep top-k + prior connections
            top_vals, top_idx = torch.topk(adj, self.top_k, dim=1)
            sparse_adj = torch.zeros_like(adj)
            for i in range(self.num_nodes):
                sparse_adj[i, top_idx[i]] = top_vals[i]

            # Add prior adjacency
            sparse_adj = sparse_adj + self.prior_adj
        else:
            sparse_adj = adj + self.prior_adj

        # Normalize rows (out-degree)
        row_sum = sparse_adj.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1  # Avoid division by zero
        sparse_adj = sparse_adj / row_sum

        return sparse_adj


class GraphAttention(nn.Module):
    """Graph attention layer for multi-head attention over adjacency."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        """
        Initialize graph attention.

        Parameters
        ----------
        in_features : int
            Input feature dimension
        out_features : int
            Output feature dimension (per head)
        num_heads : int
            Number of attention heads
        dropout : float
            Dropout rate
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        # Per-head transformations
        self.linear = nn.Linear(in_features, num_heads * out_features)
        self.attention = nn.MultiheadAttention(
            num_heads * out_features,
            num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention over graph structure.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, nodes, in_features)
        adj : torch.Tensor
            Adjacency of shape (nodes, nodes)

        Returns
        -------
        torch.Tensor
            Output of shape (batch, nodes, num_heads*out_features)
        """
        # Linear projection
        x_proj = self.linear(x)  # (batch, nodes, num_heads*out_features)

        # Attention mask from adjacency
        attn_mask = (1 - adj) * -1e9  # Mask out non-adjacent nodes

        # Reshape for attention (seq, batch, features)
        x_proj = x_proj.permute(1, 0, 2)  # (nodes, batch, num_heads*out_features)

        # Apply attention
        attn_out, _ = self.attention(
            x_proj,
            x_proj,
            x_proj,
            attn_mask=attn_mask,
        )

        # Reshape back
        attn_out = attn_out.permute(1, 0, 2)  # (batch, nodes, num_heads*out_features)
        attn_out = self.dropout(attn_out)

        return attn_out
