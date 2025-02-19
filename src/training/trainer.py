"""Training utilities for BERT classifier models.

This module provides training and evaluation functionality, including:
- Training loop management with progress tracking
- Model state management and checkpointing
- Metric computation and logging
- Error handling and validation
- Device management
- Learning rate scheduling

The main Trainer class handles model training, evaluation, and state management
while providing proper error handling and logging.

Typical usage:
    ```python
    trainer = Trainer(model, config)
    for epoch in range(num_epochs):
        loss = trainer.train_epoch(train_dataloader, optimizer, scheduler)
        score, metrics = trainer.evaluate(val_dataloader)
    ```

Attributes:
    VALID_METRICS (List[str]): List of supported evaluation metrics

Note:
    All training methods expect data to be properly batched using PyTorch DataLoader.
    Models must be instances of torch.nn.Module and implement forward() appropriately.
"""

from typing import Tuple, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm.auto import tqdm

from ..config.configuration import ModelConfig  # Changed from config to configuration
from ..utils.logging_manager import get_logger  # Changed from setup_logger

logger = get_logger(__name__)  # Changed to get_logger


class TrainerError(Exception):
    """Custom exception for Trainer errors."""


class Trainer:
    """Handles model training and evaluation.

    Manages the training loop, evaluation, model saving/loading, and metric computation.

    Args:
        model: PyTorch model to train.
        config: ModelConfig instance containing training parameters.

    Raises:
        TypeError: If model is not nn.Module or config is not ModelConfig.
    """

    def __init__(self, model: nn.Module, config: ModelConfig) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of nn.Module")
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be an instance of ModelConfig")

        config.validate()  # Validate configuration
        self.model: nn.Module = model
        self.config: ModelConfig = config
        self.device: torch.device = torch.device(config.device)
        self.model.to(self.device)
        self.metric = getattr(config, "metric", "f1")  # Default to f1

    def train_epoch(
        self,
        train_dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        progress_bar: Optional[tqdm] = None,
    ) -> float:
        """Train model for one epoch.

        Args:
            train_dataloader: DataLoader for training data.
            optimizer: Optimizer instance.
            scheduler: Learning rate scheduler.
            progress_bar: Optional progress bar for tracking.

        Returns:
            float: Average loss for the epoch.

        Raises:
            TrainerError: If training fails or dataloader is empty.
        """
        if len(train_dataloader) == 0:
            raise TrainerError("Empty training dataloader")

        total_loss = 0.0
        try:
            self.model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["label"]

                # Move data to the correct device
                outputs = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                )
                loss = nn.CrossEntropyLoss()(outputs, labels.to(self.device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

                if progress_bar:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        except Exception as e:
            raise TrainerError(f"Error during training: {str(e)}") from e

        return total_loss / len(train_dataloader)

    def evaluate(self, eval_dataloader: DataLoader) -> Tuple[float, str]:
        """Evaluate model on validation/test data.

        Args:
            eval_dataloader: DataLoader for evaluation data.

        Returns:
            Tuple containing:
                - float: Primary metric score (f1 or accuracy).
                - str: Detailed classification report.

        Raises:
            TrainerError: If evaluation fails or dataloader is empty.
        """
        if len(eval_dataloader) == 0:
            raise TrainerError("Empty evaluation dataloader")

        try:
            # Verify evaluation mode
            self.model.eval()
            for module in self.model.modules():
                if isinstance(module, (nn.Dropout, nn.BatchNorm1d)):
                    assert not module.training, (
                        f"{type(module).__name__} should be in eval mode"
                    )
            predictions = []
            actual_labels = []
            with torch.no_grad():
                for batch in eval_dataloader:
                    outputs = self.model(
                        input_ids=batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device),
                    )
                    _, preds = torch.max(outputs, dim=1)
                    predictions.extend(preds.cpu().tolist())
                    actual_labels.extend(batch["label"].cpu().tolist())

            # Calculate both metrics but return F1 as primary metric
            if self.metric == "f1":
                primary_score = f1_score(actual_labels, predictions, average="macro")
            else:
                primary_score = accuracy_score(actual_labels, predictions)

            report = classification_report(actual_labels, predictions, zero_division=0)
            return primary_score, report
        except Exception as e:
            raise TrainerError(f"Error during evaluation: {str(e)}") from e

    def save_checkpoint(self, path: str, epoch: int, optimizer: Optimizer) -> None:
        """Save a model training checkpoint.

        Args:
            path: Path to save checkpoint.
            epoch: Current epoch number.
            optimizer: Optimizer instance to save state.

        Raises:
            TrainerError: If saving fails.
        """
        try:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, path)
            logger.info("Saved checkpoint to %s", path)
        except Exception as e:
            raise TrainerError(f"Error saving checkpoint: {str(e)}") from e

    def load_checkpoint(self, path: str, optimizer: Optional[Optimizer] = None) -> int:
        """Load a model training checkpoint.

        Args:
            path: Path to load checkpoint from.
            optimizer: Optional optimizer to load state into.

        Returns:
            int: The epoch number from the checkpoint.

        Raises:
            TrainerError: If loading fails.
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Loaded checkpoint from %s", path)
            return checkpoint["epoch"]
        except Exception as e:
            raise TrainerError(f"Error loading checkpoint: {str(e)}") from e
