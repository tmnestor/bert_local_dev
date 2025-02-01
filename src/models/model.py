import logging
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from ..utils.logging_manager import get_logger

logger = get_logger(__name__)

class BERTClassifier(nn.Module):
    """BERT-based text classifier with configurable feed-forward head."""
    
    def __init__(self, bert_model_name: str, num_classes: int, classifier_config: Dict[str, Any]) -> None:
        super().__init__()
        self._validate_config(classifier_config)
        self.classifier_config = classifier_config
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
            
        hidden_size = self.bert.config.hidden_size
        self.classifier = self._build_classifier(hidden_size, num_classes)
        
        # Remove initialization logging - it will be handled by optimize.py

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validates the classifier configuration."""
        if 'hidden_dim' not in config:
            raise ValueError("hidden_dim is required in classifier_config")
            
        # Convert single integer to list for backward compatibility
        if isinstance(config['hidden_dim'], int):
            config['hidden_dim'] = [config['hidden_dim']]
            
        # Convert hidden_dim to list if it's a tuple
        if isinstance(config['hidden_dim'], tuple):
            config['hidden_dim'] = list(config['hidden_dim'])
            
        if not isinstance(config['hidden_dim'], (list, tuple)):
            raise ValueError("hidden_dim must be a list or tuple of integers")
            
        if not all(isinstance(dim, int) and dim > 0 for dim in config['hidden_dim']):
            raise ValueError("all hidden_dim values must be positive integers")
            
        if not isinstance(config.get('dropout_rate', 0.1), float) or not 0 <= config['dropout_rate'] <= 1:
            raise ValueError("dropout_rate must be a float between 0 and 1")

    def _build_classifier(self, input_size: int, num_classes: int) -> nn.Module:
        """Build the classifier layers."""
        hidden_dims = self.classifier_config['hidden_dim']
        dropout_rate = self.classifier_config.get('dropout_rate', 0.1)
        activation = self.classifier_config.get('activation', 'gelu')
        
        # Build layer sizes list
        layer_sizes = [input_size] + hidden_dims + [num_classes]
        
        logger.debug("Building classifier with layers: %s", layer_sizes)
        logger.debug("Activation: %s, Dropout: %.3f", activation, dropout_rate)
        
        # Updated activation function mapping
        activation_fn = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU()
        }.get(activation.lower())
        
        if activation_fn is None:
            logger.warning(f"Unknown activation '{activation}', using GELU")
            activation_fn = nn.GELU()
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation_fn)
                layers.append(nn.Dropout(dropout_rate))
        
        return nn.Sequential(*layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        try:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Use mean pooling
            embeddings = self._mean_pooling(outputs, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return self.classifier(embeddings)
        except Exception as e:
            raise RuntimeError(f"Error in forward pass: {str(e)}") from e
            
    def _mean_pooling(self, model_output: Dict[str, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform mean pooling on token embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

