# base.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
from torch import nn

class ModelOutput:
    """Container for model outputs"""
    def __init__(self, logits: torch.Tensor, embeddings: torch.Tensor):
        self.logits = logits
        self.embeddings = embeddings

class TextEncoder(nn.Module):
    """Base class for text encoders"""
    def __init__(self):
        super().__init__()
    
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text into embeddings"""
        raise NotImplementedError
    
    def get_output_dim(self) -> int:
        """Return the output dimension of the encoder"""
        raise NotImplementedError
    
    def freeze(self) -> None:
        """Freeze encoder parameters"""
        raise NotImplementedError

class ClassifierHead(nn.Module, ABC):
    """Abstract base class for classifier heads"""
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass of classifier head"""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration of classifier head"""
        pass