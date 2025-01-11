from dataclasses import dataclass
from typing import Optional
from pathlib import Path

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
