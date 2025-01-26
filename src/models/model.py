import logging
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

logger = logging.getLogger(__name__)

class BERTClassifier(nn.Module):
    """BERT-based text classifier with feed-forward classification head.

    Args:
        bert_model_name: Name or path of pre-trained BERT model.
        num_classes: Number of output classes.
        classifier_config: Configuration dictionary for the classifier head containing:
            - num_layers: Number of hidden layers
            - hidden_dim: Size of hidden layers
            - activation: Activation function name
            - dropout_rate: Dropout probability
    """
    def __init__(self, bert_model_name: str, num_classes: int, classifier_config: Dict[str, Any]) -> None:
        super().__init__()
        self._validate_config(classifier_config)
        self.classifier_config = classifier_config
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
            
        hidden_size = self.bert.config.hidden_size
        self.classifier = self._build_classifier(hidden_size, num_classes)

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validates the classifier configuration."""
        required_keys = ['num_layers', 'hidden_dim', 'activation', 'dropout_rate']
        if not all(key in config for key in required_keys):
            raise ValueError(f"Missing required keys in classifier_config. Required: {required_keys}")
        
        if not isinstance(config['num_layers'], int) or config['num_layers'] < 1:
            raise ValueError("num_layers must be a positive integer")
        
        if not isinstance(config['hidden_dim'], int) or config['hidden_dim'] < 1:
            raise ValueError("hidden_dim must be a positive integer")
        
        valid_activations = [
            'relu', 'leaky_relu', 'elu', 'gelu', 'selu', 
            'mish', 'swish', 'hardswish', 'tanh', 'prelu'
        ]
        if config['activation'] not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}")
        
        if not isinstance(config['dropout_rate'], float) or not 0 <= config['dropout_rate'] <= 1:
            raise ValueError("dropout_rate must be a float between 0 and 1")

    def _get_regularization(self, size: int, dropout_rate: float = 0.1) -> nn.Module:
        """Get regularization layer."""
        return nn.Dropout(dropout_rate)

    def _build_classifier(self, input_size: int, num_classes: int) -> nn.Module:
        layer_sizes = self._calculate_layer_sizes(input_size, self.classifier_config['hidden_dim'], self.classifier_config['num_layers'], num_classes)
        
        logger.info("\nBuilding standard classifier:")
        logger.info("  Input size: %s", layer_sizes[0])
        logger.info("  Hidden layers: %s", len(layer_sizes) - 2)
        logger.info("  Hidden dimension: %s", self.classifier_config['hidden_dim'])
        logger.info("  Activation: %s", self.classifier_config['activation'])
        logger.info("  Regularization: Dropout (rate: %s)", self.classifier_config['dropout_rate'])
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            current_size = layer_sizes[i]
            next_size = layer_sizes[i + 1]
            
            layers.append(nn.Linear(current_size, next_size))
            if i < len(layer_sizes) - 2:  # Hidden layer
                layers.append(self._get_activation(self.classifier_config['activation']))
                layers.append(self._get_regularization(next_size, self.classifier_config['dropout_rate']))
        
        return nn.Sequential(*layers)

    def _calculate_layer_sizes(self, input_size: int, hidden_dim: int, num_layers: int, num_classes: int) -> List[int]:
        """Calculate the sizes for each layer using hidden_dim."""
        if num_layers == 1:
            return [input_size, num_classes]
        
        # For multiple layers, create a smooth progression to hidden_dim
        layer_sizes = [input_size]
        current_size = input_size
        
        if num_layers > 2:
            # Calculate intermediate sizes with geometric progression
            ratio = (hidden_dim / current_size) ** (1.0 / (num_layers - 1))
            for _ in range(num_layers - 1):
                current_size = int(current_size * ratio)
                # Ensure we don't go below hidden_dim
                current_size = max(current_size, hidden_dim)
                layer_sizes.append(current_size)
        else:
            # For 2 layers, use hidden_dim directly
            layer_sizes.append(hidden_dim)
        
        layer_sizes.append(num_classes)
        return layer_sizes

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activation_map = {
            # Basic activations
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
            'selu': nn.SELU,
            # Advanced activations
            'mish': nn.Mish,
            'swish': nn.SiLU,  # SiLU is also known as Swish
            'hardswish': nn.Hardswish,
            'tanh': nn.Tanh,
            'prelu': nn.PReLU
        }
        
        activation_fn = activation_map.get(activation)
        if not activation_fn:
            raise ValueError(f"Unknown activation: {activation}")
            
        # Special handling for PReLU which requires initialization
        if activation == 'prelu':
            return activation_fn()
        return activation_fn()
    
    def _mean_pooling(self, model_output: Dict[str, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            input_ids: Tensor of token ids.
            attention_mask: Tensor of attention mask.

        Returns:
            Tensor of logits for each class.

        Raises:
            RuntimeError: If there's an error during forward pass.
        """
        try:
            # Add more detailed mode logging
            if not self.training:
                # Check and log dropout and batchnorm states
                dropout_states = []
                batchnorm_states = []
                for name, module in self.named_modules():
                    if isinstance(module, nn.Dropout):
                        dropout_states.append(f"{name}: {'enabled' if module.training else 'disabled'}")
                    elif isinstance(module, nn.BatchNorm1d):
                        batchnorm_states.append(f"{name}: {'training' if module.training else 'eval'}")
                
                if not hasattr(self, '_logged_eval_mode'):
                    logger.info("Model evaluation mode states:")
                    logger.info("Dropout layers: %s", ', '.join(dropout_states))
                    logger.info("BatchNorm layers: %s", ', '.join(batchnorm_states))
                    self._logged_eval_mode = True

            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Always use mean pooling for standard architecture
            embeddings = self._mean_pooling(outputs, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return self.classifier(embeddings)
        except Exception as e:
            raise RuntimeError(f"Error in forward pass: {str(e)}") from e

