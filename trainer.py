import logging
from typing import Tuple, Optional, Literal
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, f1_score
from config import ModelConfig

logger = logging.getLogger(__name__)

class TrainerError(Exception):
    """Custom exception for Trainer errors."""
    pass

class Trainer:
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
        self.metric = getattr(config, 'metric', 'f1')  # Default to f1

    def train_epoch(
        self, 
        train_dataloader: DataLoader, 
        optimizer: Optimizer, 
        scheduler: _LRScheduler
    ) -> float:
        if len(train_dataloader) == 0:
            raise TrainerError("Empty training dataloader")
            
        total_loss = 0.0
        try:
            self.model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device)
                )
                loss = nn.CrossEntropyLoss()(outputs, batch['label'].to(self.device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                
        except Exception as e:
            raise TrainerError(f"Error during training: {str(e)}") from e
            
        return total_loss / len(train_dataloader)

    def evaluate(self, eval_dataloader: DataLoader) -> Tuple[float, str]:
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
                    _, preds = torch.max(outputs, dim=1)
                    predictions.extend(preds.cpu().tolist())
                    actual_labels.extend(batch['label'].cpu().tolist())
            
            # Calculate both metrics but return F1 as primary metric
            accuracy = accuracy_score(actual_labels, predictions)
            if self.metric == 'f1':
                primary_score = f1_score(actual_labels, predictions, average='macro')
            else:
                primary_score = accuracy
                
            report = classification_report(actual_labels, predictions, zero_division=0)
            logger.info(f"Evaluation metrics - F1: {primary_score:.4f}, Accuracy: {accuracy:.4f}")
            
            return (primary_score, report)
        except Exception as e:
            raise TrainerError(f"Error during evaluation: {str(e)}") from e
