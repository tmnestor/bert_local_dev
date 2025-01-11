from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class ModelConfig:
    bert_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    num_classes: int = 5
    max_length: int = 16
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 2e-5
    device: str = "cpu"
    data_file: Path = Path("data/bbc-text.csv")
    model_save_path: Path = Path("bert_classifier.pth")
    hidden_dropout: float = 0.1
