from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import torch

@dataclass
class ModelConfig:
    bert_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    num_classes: int = 5
    max_length: int = 128
    batch_size: int = 16
    num_epochs: int = 10  # Number of epochs
    learning_rate: float = 2e-5
    device: str = "cpu"
    data_file: Path = Path("data/bbc-text.csv")
    model_save_path: Path = Path("bert_classifier.pth")
    hidden_dropout: float = 0.1
    n_trials: int = 100

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_classes < 1:
            raise ValueError("num_classes must be positive")
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be positive")
        if not (0.0 < self.learning_rate < 1.0):
            raise ValueError("learning_rate must be between 0 and 1")
        if not (0.0 <= self.hidden_dropout <= 1.0):
            raise ValueError("hidden_dropout must be between 0 and 1")
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        if not torch.cuda.is_available() and self.device.startswith("cuda"):
            raise RuntimeError("CUDA device requested but CUDA is not available")
