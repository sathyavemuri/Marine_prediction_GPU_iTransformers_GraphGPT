"""MTGNN trainer with validation and early stopping."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
from typing import Optional, Tuple
import numpy as np

from ..models import MTGNN
from ..config import Config

logger = logging.getLogger(__name__)


class Trainer:
    """Train MTGNN model with early stopping."""

    def __init__(
        self,
        model: MTGNN,
        config: Config,
        device: str = "cpu",
    ):
        """
        Initialize trainer.

        Parameters
        ----------
        model : MTGNN
            Model to train
        config : Config
            Configuration object
        device : str
            Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Early stopping
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.early_stopping_patience = config.training.early_stopping_patience

        # Training history
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader

        Returns
        -------
        float
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            # Move to device
            history = batch["history"].to(self.device)  # (batch, lookback, nodes)
            targets = batch["targets"].to(self.device)  # (batch, horizon, targets)

            # Forward pass
            predictions = self.model(history)  # (batch, horizon, targets)

            # Loss
            loss = self.criterion(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm,
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate on validation set.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader

        Returns
        -------
        float
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in val_loader:
            history = batch["history"].to(self.device)
            targets = batch["targets"].to(self.device)

            predictions = self.model(history)
            loss = self.criterion(predictions, targets)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str | Path = "outputs",
    ) -> dict:
        """
        Train model with early stopping.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        output_dir : str | Path
            Directory to save checkpoints

        Returns
        -------
        dict
            Training history and final metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training for max {self.config.training.max_epochs} epochs")
        logger.info(f"Early stopping patience: {self.early_stopping_patience}")

        for epoch in range(self.config.training.max_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Log
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1:3d}: "
                    f"train_loss={train_loss:.6f}, "
                    f"val_loss={val_loss:.6f}"
                )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save checkpoint
                self._save_checkpoint(output_dir / "best_model.pt", epoch)
                logger.debug(f"New best validation loss: {val_loss:.6f}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(patience {self.patience_counter}/{self.early_stopping_patience})"
                )
                break

        # Load best model
        best_model_path = output_dir / "best_model.pt"
        if best_model_path.exists():
            self.load_checkpoint(best_model_path)
            logger.info(f"Loaded best model from {best_model_path}")

        # Save final history
        self._save_history(output_dir)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": len(self.train_losses),
        }

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": self.best_val_loss,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from {path}")

    def _save_history(self, output_dir: Path) -> None:
        """Save training history."""
        history = {
            "train_losses": [float(x) for x in self.train_losses],
            "val_losses": [float(x) for x in self.val_losses],
            "best_val_loss": float(self.best_val_loss),
        }

        path = output_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {path}")

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        Evaluate on test set.

        Parameters
        ----------
        test_loader : DataLoader
            Test data loader

        Returns
        -------
        dict
            Test metrics (loss, MAE per target, etc.)
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0
        num_batches = 0

        for batch in test_loader:
            history = batch["history"].to(self.device)
            targets = batch["targets"].to(self.device)

            predictions = self.model(history)

            loss = self.criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        # Stack all predictions and targets
        all_predictions = np.vstack(all_predictions)  # (num_samples, horizon, targets)
        all_targets = np.vstack(all_targets)

        # Compute metrics
        mse = np.mean((all_predictions - all_targets) ** 2)
        mae = np.mean(np.abs(all_predictions - all_targets))

        # Per-target MAE
        per_target_mae = np.mean(np.abs(all_predictions - all_targets), axis=(0, 1))

        metrics = {
            "test_loss": total_loss / num_batches if num_batches > 0 else 0,
            "mse": float(mse),
            "mae": float(mae),
            "per_target_mae": [float(x) for x in per_target_mae],
        }

        return metrics
