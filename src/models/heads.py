# heads.py
from typing import Dict, Any
import torch
from torch import nn
from .base import ClassifierHead
from .model import PlaneResNetBlock

class StandardClassifierHead(ClassifierHead):
    """Standard MLP classifier head"""
    def __init__(self, input_dim: int, num_classes: int, config: Dict[str, Any]):
        super().__init__()
        layers = []
        current_dim = input_dim  # Start with BERT's output dimension (768)
        
        for _ in range(config['num_layers']):
            layers.extend([
                nn.Linear(current_dim, config['hidden_dim']),
                nn.GELU() if config['activation'] == 'gelu' else nn.ReLU(),
                nn.Dropout(config['dropout_rate'])
            ])
            current_dim = config['hidden_dim']
            
        layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)
    
    def get_config(self) -> Dict[str, Any]:
        return self.config

class PlaneResNetHead(ClassifierHead):
    """Classification head using parallel residual network architecture"""
    def __init__(self, input_dim: int, num_classes: int, num_planes: int, plane_width: int):
        super().__init__()
        
        self.config = {
            'num_planes': num_planes,
            'plane_width': plane_width
        }
        
        # Use BERT's output dimension (768) as input
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, plane_width),
            nn.BatchNorm1d(plane_width),
            nn.ReLU()
        )
        
        self.planes = nn.ModuleList([
            PlaneResNetBlock(plane_width) for _ in range(num_planes)
        ])
        
        self.output = nn.Linear(plane_width, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(embeddings)
        for plane in self.planes:
            x = plane(x)
        return self.output(x)
    
    def get_config(self) -> Dict[str, Any]:
        return self.config