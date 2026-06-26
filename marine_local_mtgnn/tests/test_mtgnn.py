"""Tests for MTGNN model components."""

import pytest
import torch
import numpy as np

from marine_local_mtgnn.models.graph_conv import GraphConvolution, AdaptiveAdjacency
from marine_local_mtgnn.models.temporal import TemporalEncoder, TemporalDecoder
from marine_local_mtgnn.models.mtgnn import MTGNN


class TestGraphConvolution:
    """Test graph convolution layer."""

    def test_graph_conv_output_shape(self):
        """Test output shape of graph convolution."""
        batch_size = 4
        num_nodes = 19
        in_features = 16
        out_features = 32

        gc = GraphConvolution(in_features, out_features)
        x = torch.randn(batch_size, num_nodes, in_features)
        adj = torch.eye(num_nodes)  # Identity adjacency

        output = gc(x, adj)

        assert output.shape == (batch_size, num_nodes, out_features)

    def test_graph_conv_with_random_adj(self):
        """Test graph convolution with random adjacency."""
        batch_size = 2
        num_nodes = 5
        in_features = 8
        out_features = 16

        gc = GraphConvolution(in_features, out_features)
        x = torch.randn(batch_size, num_nodes, in_features)
        adj = torch.randn(num_nodes, num_nodes)
        adj = adj / adj.sum(dim=1, keepdim=True)  # Row normalize

        output = gc(x, adj)

        assert output.shape == (batch_size, num_nodes, out_features)
        assert torch.all(torch.isfinite(output))


class TestAdaptiveAdjacency:
    """Test adaptive adjacency matrix."""

    def test_adaptive_adj_shape(self):
        """Test adjacency output shape."""
        num_nodes = 19
        aa = AdaptiveAdjacency(num_nodes)

        adj = aa()

        assert adj.shape == (num_nodes, num_nodes)
        assert torch.all(torch.isfinite(adj))

    def test_adaptive_adj_row_stochastic(self):
        """Test that adjacency is row-stochastic (sums to 1)."""
        num_nodes = 10
        aa = AdaptiveAdjacency(num_nodes, top_k=3)

        adj = aa()

        row_sums = adj.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(num_nodes), atol=1e-5)

    def test_adaptive_adj_with_prior(self):
        """Test adaptive adjacency with prior."""
        num_nodes = 5
        prior = np.eye(num_nodes)

        aa = AdaptiveAdjacency(num_nodes, prior_adj=prior, top_k=2)
        adj = aa()

        # Diagonal should be present (from prior + learned)
        assert torch.all(torch.diag(adj) > 0)


class TestTemporalEncoder:
    """Test temporal encoder."""

    def test_encoder_output_shape(self):
        """Test encoder output shape."""
        batch_size = 4
        num_nodes = 19
        time_steps = 672
        hidden_size = 32

        encoder = TemporalEncoder(
            input_size=num_nodes,
            hidden_size=hidden_size,
            num_blocks=3,
        )
        x = torch.randn(batch_size, num_nodes, time_steps)

        output = encoder(x)

        assert output.shape == (batch_size, hidden_size, time_steps)

    def test_encoder_finite_output(self):
        """Test encoder produces finite values."""
        batch_size = 2
        num_nodes = 5
        time_steps = 100
        hidden_size = 16

        encoder = TemporalEncoder(
            input_size=num_nodes,
            hidden_size=hidden_size,
            num_blocks=5,
            dropout=0.1,
        )
        x = torch.randn(batch_size, num_nodes, time_steps)

        output = encoder(x)

        assert torch.all(torch.isfinite(output))


class TestTemporalDecoder:
    """Test temporal decoder."""

    def test_decoder_output_shape(self):
        """Test decoder output shape."""
        batch_size = 4
        hidden_size = 32
        num_targets = 15
        horizon_steps = 672

        decoder = TemporalDecoder(
            hidden_size=hidden_size + 64,  # hidden + skip
            output_size=num_targets,
            horizon_steps=horizon_steps,
        )
        encoded = torch.randn(batch_size, hidden_size + 64, horizon_steps)

        output = decoder(encoded)

        assert output.shape == (batch_size, horizon_steps, num_targets)

    def test_decoder_finite_output(self):
        """Test decoder produces finite values."""
        batch_size = 2
        hidden_size = 16
        num_targets = 5
        horizon_steps = 96

        decoder = TemporalDecoder(
            hidden_size=hidden_size + 32,
            output_size=num_targets,
            horizon_steps=horizon_steps,
            dropout=0.1,
        )
        encoded = torch.randn(batch_size, hidden_size + 32, horizon_steps)

        output = decoder(encoded)

        assert torch.all(torch.isfinite(output))


class TestMTGNN:
    """Test full MTGNN model."""

    def test_mtgnn_output_shape(self):
        """Test MTGNN output shape."""
        batch_size = 4
        num_nodes = 19
        num_targets = 15
        lookback_steps = 672
        horizon_steps = 672

        model = MTGNN(
            num_nodes=num_nodes,
            num_targets=num_targets,
            lookback_steps=lookback_steps,
            horizon_steps=horizon_steps,
            hidden_channels=32,
        )
        x = torch.randn(batch_size, lookback_steps, num_nodes)

        output = model(x)

        assert output.shape == (batch_size, horizon_steps, num_targets)

    def test_mtgnn_finite_output(self):
        """Test MTGNN produces finite values."""
        batch_size = 2
        num_nodes = 19
        num_targets = 15
        lookback_steps = 96
        horizon_steps = 96

        model = MTGNN(
            num_nodes=num_nodes,
            num_targets=num_targets,
            lookback_steps=lookback_steps,
            horizon_steps=horizon_steps,
            hidden_channels=16,
            dropout=0.1,
        )
        x = torch.randn(batch_size, lookback_steps, num_nodes)

        output = model(x)

        assert torch.all(torch.isfinite(output))

    def test_mtgnn_grad_flow(self):
        """Test gradient flow through MTGNN."""
        batch_size = 2
        num_nodes = 5
        num_targets = 3
        lookback_steps = 50
        horizon_steps = 50

        model = MTGNN(
            num_nodes=num_nodes,
            num_targets=num_targets,
            lookback_steps=lookback_steps,
            horizon_steps=horizon_steps,
            hidden_channels=8,
        )

        x = torch.randn(batch_size, lookback_steps, num_nodes, requires_grad=True)
        target = torch.randn(batch_size, horizon_steps, num_targets)

        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Check gradients exist in temporal encoder
        assert x.grad is not None
        assert model.temporal_encoder.input_proj.weight.grad is not None

    def test_mtgnn_with_prior_adj(self):
        """Test MTGNN with prior adjacency matrix."""
        num_nodes = 5
        prior = np.eye(num_nodes)

        model = MTGNN(
            num_nodes=num_nodes,
            num_targets=3,
            lookback_steps=50,
            horizon_steps=50,
            hidden_channels=8,
            prior_adj=prior,
        )

        x = torch.randn(2, 50, num_nodes)
        output = model(x)

        assert output.shape == (2, 50, 3)

    def test_mtgnn_get_adjacency(self):
        """Test extracting learned adjacency."""
        num_nodes = 10

        model = MTGNN(
            num_nodes=num_nodes,
            num_targets=5,
            lookback_steps=50,
            horizon_steps=50,
            hidden_channels=8,
        )

        # Generate dummy forward pass
        _ = model(torch.randn(2, 50, num_nodes))

        # Extract adjacency
        adj = model.get_adjacency()

        assert adj.shape == (num_nodes, num_nodes)
        assert np.all(np.isfinite(adj))
