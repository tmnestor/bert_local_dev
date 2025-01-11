import logging
from typing import Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW  # Change this import
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from config import ModelConfig  # Add this import

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model: nn.Module, config: ModelConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
    def train_epoch(self, train_dataloader: DataLoader, optimizer: AdamW, scheduler):
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

    def evaluate(self, eval_dataloader: DataLoader) -> Tuple[float, str]:
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
                
        return (
            accuracy_score(actual_labels, predictions),
            classification_report(actual_labels, predictions, zero_division=0)  # Add zero_division parameter
        )
