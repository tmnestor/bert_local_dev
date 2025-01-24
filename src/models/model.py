# model.py
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel
from .base import TextEncoder, ClassifierHead, ModelOutput

logger = logging.getLogger(__name__)

class PlaneResNetBlock(nn.Module):
    """A residual block used in the PlaneResNet architecture.
    
    Implements a residual connection with batch normalization and ReLU activation.
    
    Args:
        width: Width of the linear layers in the block.
    """
    def __init__(self, width: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.BatchNorm1d(width)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.block(x)
        return self.relu(out + identity)

class PlaneResNetHead(nn.Module):
    """Classification head using parallel residual network architecture.
    
    Args:
        input_size: Dimension of input features.
        num_classes: Number of output classes.
        num_planes: Number of parallel residual blocks.
        plane_width: Width of each residual block.
    """
    def __init__(self, input_size: int, num_classes: int, num_planes: int, plane_width: int):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, plane_width),
            nn.BatchNorm1d(plane_width),
            nn.ReLU()
        )
        
        # Stack plane blocks
        self.planes = nn.ModuleList([
            PlaneResNetBlock(plane_width) for _ in range(num_planes)
        ])
        
        # Output projection
        self.output = nn.Linear(plane_width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for plane in self.planes:
            x = plane(x)
        return self.output(x)

class BERTClassifier(nn.Module):
    """BERT-based classifier with configurable head architecture"""
    
    def __init__(self, encoder: TextEncoder, head: ClassifierHead):
        super().__init__()
        self.encoder = encoder
        self.head = head
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ModelOutput:
        embeddings = self.encoder.encode(input_ids, attention_mask)
        logits = self.head(embeddings)
        return ModelOutput(logits=logits, embeddings=embeddings)
    
    def freeze_encoder(self) -> None:
        """Freeze the encoder parameters"""
        self.encoder.freeze()
        
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'encoder': self.encoder.__class__.__name__,
            'head': self.head.get_config()
        }