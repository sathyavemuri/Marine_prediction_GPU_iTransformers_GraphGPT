"""Training loop for MarineITransformer."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import logging
import json
from typing import Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class HuberLossWeighted(nn.Module):
    """Weighted Huber loss with masks and horizon decay."""

    def __init__(
        self,
        config,
        target_loss_weights: Dict[str, float],
    ):
        """
        Initialize Huber loss.

        Parameters
        ----------
        config : Config
            Configuration object
        target_loss_weights : dict
            Loss weight per target
        """
        super().__init__()
        self.config = config
        # Determine number of targets from the weights dict
        n_targets = len(target_loss_weights)
        self.target_loss_weights = torch.tensor(
            [target_loss_weights.get(f'target_{i}', 1.0) for i in range(n_targets)],
            dtype=torch.float32
        )
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted loss.

        Parameters
        ----------
        predictions : Tensor
            [batch, horizon, n_targets] model output
        targets : Tensor
            [batch, horizon, n_targets] ground truth
        masks : Tensor
            [batch, horizon, n_targets] valid value mask

        Returns
        -------
        loss : Tensor
            Scalar loss
        """
        # Compute per-element loss
        element_loss = self.smoothl1(predictions, targets)  # [batch, horizon, targets]

        # Apply target weights
        device = element_loss.device
        target_weights = self.target_loss_weights.to(device)
        weighted_loss = element_loss * target_weights.unsqueeze(0).unsqueeze(0)

        # Apply horizon decay: 1.0 at t=0, 0.75 at t=horizon
        batch_size, horizon, n_targets = weighted_loss.shape
        horizon_weights = torch.linspace(1.0, 0.75, horizon, device=device)
        weighted_loss = weighted_loss * horizon_weights.unsqueeze(0).unsqueeze(2)

        # Apply mask
        weighted_loss = weighted_loss * masks

        # Average
        denominator = masks.sum() + 1e-8
        loss = weighted_loss.sum() / denominator

        return loss


class Trainer:
    """Training orchestrator for MarineITransformer."""

    def __init__(
        self,
        model: nn.Module,
        config,
        target_loss_weights: Dict[str, float],
        device: str = 'cpu',
    ):
        """
        Initialize trainer.

        Parameters
        ----------
        model : nn.Module
            MarineITransformer model
        config : Config
            Configuration object
        target_loss_weights : dict
            Loss weight per target
        device : str
            'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Loss
        self.loss_fn = HuberLossWeighted(config, target_loss_weights)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
        )

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0

        # History
        self.train_losses = []
        self.val_losses = []

        # Mixed precision
        self.use_amp = config.training.use_amp_if_cuda and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None

        logger.info(f"Trainer initialized on device: {device}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader

        Returns
        -------
        avg_loss : float
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # Move to device
            x_past = batch['x_past'].to(self.device)
            x_future_known = batch['x_future_known'].to(self.device)
            y_target = batch['y_target'].to(self.device)
            y_mask = batch['y_mask'].to(self.device)

            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions = self.model(x_past, x_future_known)
                    loss = self.loss_fn(predictions, y_target, y_mask)

                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                if self.config.training.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(x_past, x_future_known)
                loss = self.loss_fn(predictions, y_target, y_mask)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config.training.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip_norm,
                    )

                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
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
        avg_loss : float
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            # Move to device
            x_past = batch['x_past'].to(self.device)
            x_future_known = batch['x_future_known'].to(self.device)
            y_target = batch['y_target'].to(self.device)
            y_mask = batch['y_mask'].to(self.device)

            # Forward pass
            predictions = self.model(x_past, x_future_known)
            loss = self.loss_fn(predictions, y_target, y_mask)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path,
    ) -> Dict:
        """
        Train for max epochs with early stopping.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        output_dir : Path
            Directory to save checkpoints

        Returns
        -------
        dict
            Training result with history and metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("TRAINING START")
        logger.info("=" * 80)

        max_epochs = self.config.training.epochs
        early_stopping_patience = self.config.training.early_stopping_patience

        for epoch in range(1, max_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Scheduler step
            self.scheduler.step(val_loss)

            # Logging
            logger.info(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

            # Early stopping and checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0

                # Save best model
                checkpoint_path = output_dir / 'best_model.pt'
                self._save_checkpoint(checkpoint_path, epoch, val_loss)
                logger.info(f"  ✓ New best model saved (val_loss: {val_loss:.6f})")

            else:
                self.patience_counter += 1
                logger.info(
                    f"  No improvement ({self.patience_counter}/{early_stopping_patience})"
                )

                if self.patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best epoch: {self.best_epoch}")
        logger.info(f"Best val loss: {self.best_val_loss:.6f}")

        # Save training history
        self._save_history(output_dir)

        return {
            'epochs_trained': epoch,
            'best_epoch': self.best_epoch,
            'best_val_loss': float(self.best_val_loss),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate on test set.

        Parameters
        ----------
        test_loader : DataLoader
            Test data loader

        Returns
        -------
        dict
            Test metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0

        for batch in test_loader:
            # Move to device
            x_past = batch['x_past'].to(self.device)
            x_future_known = batch['x_future_known'].to(self.device)
            y_target = batch['y_target'].to(self.device)
            y_mask = batch['y_mask'].to(self.device)

            # Forward pass
            predictions = self.model(x_past, x_future_known)
            loss = self.loss_fn(predictions, y_target, y_mask)

            total_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_target.cpu().numpy())
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Concatenate all predictions
        predictions_full = np.concatenate(all_predictions, axis=0)
        targets_full = np.concatenate(all_targets, axis=0)

        logger.info(f"Test Loss: {avg_loss:.6f}")
        logger.info(f"Predictions shape: {predictions_full.shape}")
        logger.info(f"Targets shape: {targets_full.shape}")

        return {
            'test_loss': float(avg_loss),
            'predictions': predictions_full,
            'targets': targets_full,
        }

    def _save_checkpoint(self, path: Path, epoch: int, val_loss: float):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, path)

    def _save_history(self, output_dir: Path):
        """Save training history."""
        history_path = output_dir / 'training_history.json'
        history = {
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [float(x) for x in self.val_losses],
            'best_epoch': self.best_epoch,
            'best_val_loss': float(self.best_val_loss),
        }
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history: {history_path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        logger.info(f"Loaded checkpoint from {path}")
