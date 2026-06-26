"""Graph prior construction: lagged correlation matrix."""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

from ..constants import NODE_NAMES

logger = logging.getLogger(__name__)


class GraphPrior:
    """Build graph prior from lagged correlations in training data."""

    def __init__(self, max_lag_steps: int = 96):
        """
        Initialize graph prior builder.

        Parameters
        ----------
        max_lag_steps : int
            Maximum lag to consider for correlations (default 96 = 1 day at 15-min)
        """
        self.max_lag_steps = max_lag_steps
        self.adjacency = None
        self.correlations = None

    def fit(self, df_train: pd.DataFrame) -> np.ndarray:
        """
        Build graph prior from training data.

        Computes lagged correlation matrix: for each pair of nodes (i,j),
        find maximum absolute correlation at any lag from 0 to max_lag_steps.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data (unlabeled index, columns = NODE_NAMES)

        Returns
        -------
        np.ndarray
            Adjacency matrix of shape (n_nodes, n_nodes)
        """
        n_nodes = len(NODE_NAMES)
        data = df_train.values

        # Standardize
        data_mean = np.nanmean(data, axis=0)
        data_std = np.nanstd(data, axis=0)
        data_std[data_std == 0] = 1  # Avoid division by zero
        data_normalized = (data - data_mean) / data_std

        # Compute lagged correlations
        corr_max = np.zeros((n_nodes, n_nodes))

        for i in range(n_nodes):
            for j in range(n_nodes):
                # Get both time series
                xi = data_normalized[:, i]
                xj = data_normalized[:, j]

                # Handle NaN
                mask = ~(np.isnan(xi) | np.isnan(xj))
                if np.sum(mask) < 10:
                    corr_max[i, j] = 0
                    continue

                xi_clean = xi[mask]
                xj_clean = xj[mask]

                # Compute correlations at different lags
                max_corr = 0
                for lag in range(self.max_lag_steps + 1):
                    if lag == 0:
                        # Contemporaneous correlation
                        corr = np.corrcoef(xi_clean, xj_clean)[0, 1]
                    elif lag < len(xi_clean):
                        # Lagged correlation: xj lags behind xi
                        corr = np.corrcoef(xi_clean[:-lag], xj_clean[lag:])[0, 1]
                    else:
                        continue

                    if not np.isnan(corr):
                        max_corr = max(max_corr, np.abs(corr))

                corr_max[i, j] = max_corr

        self.correlations = corr_max
        self.adjacency = corr_max

        logger.info(f"Built graph prior: {n_nodes}×{n_nodes} adjacency matrix")
        logger.debug(f"Mean correlation strength: {np.mean(corr_max):.4f}")

        return self.adjacency

    def get_sparse_adjacency(self, top_k: int = 4) -> np.ndarray:
        """
        Get sparse adjacency matrix keeping top-k connections per node.

        Parameters
        ----------
        top_k : int
            Number of top connections to keep for each node

        Returns
        -------
        np.ndarray
            Sparse adjacency matrix of shape (n_nodes, n_nodes)
        """
        if self.adjacency is None:
            raise ValueError("Must fit graph prior first")

        n_nodes = len(NODE_NAMES)
        sparse_adj = np.zeros_like(self.adjacency)

        for i in range(n_nodes):
            # Get top-k neighbors for node i (excluding self-connection)
            neighbors = np.argsort(-self.adjacency[i, :])[:top_k + 1]
            neighbors = neighbors[neighbors != i][:top_k]
            sparse_adj[i, neighbors] = self.adjacency[i, neighbors]

        logger.debug(f"Created sparse adjacency (top-{top_k})")

        return sparse_adj

    def save(self, output_dir: str | Path = "outputs") -> None:
        """
        Save graph prior to CSV.

        Parameters
        ----------
        output_dir : str | Path
            Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.adjacency is None:
            raise ValueError("No graph prior to save; must fit first")

        # Save as CSV with node names
        adj_df = pd.DataFrame(
            self.adjacency,
            index=NODE_NAMES,
            columns=NODE_NAMES,
        )

        csv_path = output_dir / "graph_prior_adjacency.csv"
        adj_df.to_csv(csv_path)
        logger.info(f"Saved graph prior: {csv_path}")

        # Save sparse version
        sparse_adj = self.get_sparse_adjacency(top_k=4)
        sparse_df = pd.DataFrame(
            sparse_adj,
            index=NODE_NAMES,
            columns=NODE_NAMES,
        )

        sparse_path = output_dir / "graph_prior_sparse_k4.csv"
        sparse_df.to_csv(sparse_path)
        logger.info(f"Saved sparse graph prior: {sparse_path}")
