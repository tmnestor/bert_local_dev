import logging
from typing import Dict, Any, Tuple, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm.auto import tqdm

from src.config.config import ModelConfig
from src.models.model import BERTClassifier
from src.utils.metrics import compute_metrics
from ..utils.progress_manager import ProgressManager

logger = logging.getLogger(__name__)

class TrainerError(Exception):
    """Custom exception for Trainer errors."""
    pass

class Trainer:
    """Model trainer class"""
    
    def __init__(self, model: BERTClassifier, config: ModelConfig, disable_pbar: bool = False):
        """Initialize trainer with model and configuration"""
        self.model = model
        
        # Set required defaults before anything else
        self._set_config_defaults(config)
        
        self.config = config
        self.device = config.device  # Store device from config
        self.criterion = nn.CrossEntropyLoss()
        self.metric = getattr(config, 'metric', 'f1')  # Add metric attribute
        self.disable_pbar = disable_pbar
        self.epoch_pbar = None
        self.batch_pbar = None
        self._active_pbar = None  # Add tracking of active progress bar
        self.total_steps = 0
        self.current_step = 0
        self.progress = ProgressManager(disable=disable_pbar)
        
        # Validate configuration after ensuring defaults
        config.validate()
    
    def _set_config_defaults(self, config: ModelConfig) -> None:
        """Ensure all required config fields have valid defaults"""
        defaults = {
            'weight_decay': 0.01,
            'learning_rate': 2e-5,
            'warmup_steps': 0,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0
        }
        
        # Always set defaults first
        for key, default_value in defaults.items():
            current_value = getattr(config, key, None)
            if current_value is None:
                setattr(config, key, default_value)
    
    def set_progress_bars(self, epoch_pbar: Optional[tqdm] = None, batch_pbar: Optional[tqdm] = None):
        """Set progress bars to use during training"""
        self.epoch_pbar = epoch_pbar
        self.batch_pbar = batch_pbar
    
    def train_epoch(
        self, 
        train_dataloader: DataLoader, 
        optimizer: Optimizer, 
        scheduler: _LRScheduler
    ) -> float:
        """Train model for one epoch.
        
        Args:
            train_dataloader: DataLoader for training data.
            optimizer: Optimizer instance.
            scheduler: Learning rate scheduler.
            
        Returns:
            float: Average loss for the epoch.

        Raises:
            TrainerError: If training fails or dataloader is empty.
        """
        if len(train_dataloader) == 0:
            raise TrainerError("Empty training dataloader")
            
        # Validate batch size
        if train_dataloader.batch_size < 2:
            raise TrainerError("Batch size must be at least 2 for BatchNorm layers")
            
        total_loss = 0.0
        self.total_steps = len(train_dataloader)
        self.current_step = 0
        
        try:
            self.model.train()
            batch_bar = self.progress.init_batch_bar(len(train_dataloader))
            
            for step, batch in enumerate(train_dataloader, start=1):  # Start counting from 1
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device)
                )
                # Access logits from ModelOutput
                loss = self.criterion(outputs.logits, batch['label'].to(self.device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                
                if not self.disable_pbar:
                    self.progress.update_batch({'loss': f'{loss.item():.4f}'})
                    
            # Always use average loss for consistent reporting
            avg_loss = total_loss / len(train_dataloader)
            
            if self.epoch_pbar and not self.disable_pbar:
                self.epoch_pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
            return avg_loss
                
        except Exception as e:
            raise TrainerError(f"Error during training: {str(e)}") from e
            
        finally:
            if batch_bar:
                batch_bar.close()
                
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
            self.model.eval()
            predictions = []
            actual_labels = []
            
            with torch.no_grad():
                for batch in eval_dataloader:
                    outputs = self.model(
                        input_ids=batch['input_ids'].to(self.device),
                        attention_mask=batch['attention_mask'].to(self.device)
                    )
                    # Access logits from ModelOutput
                    _, preds = torch.max(outputs.logits, dim=1)
                    predictions.extend(preds.cpu().tolist())
                    actual_labels.extend(batch['label'].cpu().tolist())
            
            # Calculate both metrics but return F1 as primary metric
            accuracy = accuracy_score(actual_labels, predictions)
            if self.metric == 'f1':
                primary_score = f1_score(actual_labels, predictions, average='macro')
            else:
                primary_score = accuracy
                
            report = classification_report(actual_labels, predictions, zero_division=0)
            
            return (primary_score, report)
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
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint to {path}")
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
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded checkpoint from {path}")
            return checkpoint['epoch']
        except Exception as e:
            raise TrainerError(f"Error loading checkpoint: {str(e)}") from e
    
    def __del__(self):
        """Cleanup owned progress bars only"""
        if self._active_pbar is not None:
            self._active_pbar.close()
            self._active_pbar = None
