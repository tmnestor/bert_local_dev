import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

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
    """BERT-based text classifier with configurable classification head.

    Combines a pre-trained BERT model with either a standard feed-forward or
    PlaneResNet classification head.

    Args:
        bert_model_name: Name or path of pre-trained BERT model.
        num_classes: Number of output classes.
        classifier_config: Configuration dictionary for the classifier head.

    The classifier_config dictionary should contain:
        - architecture_type: 'standard' or 'plane_resnet'
        - cls_pooling: Whether to use CLS token pooling (bool)
        For standard architecture:
            - num_layers: Number of hidden layers
            - hidden_dim: Size of hidden layers
            - activation: Activation function name
            - regularization: 'dropout' or 'batchnorm'
            - dropout_rate: Dropout probability if using dropout
        For plane_resnet:
            - num_planes: Number of parallel planes
            - plane_width: Width of each plane

    Raises:
        ValueError: For invalid configuration parameters.
    """
    def __init__(self, bert_model_name: str, num_classes: int, classifier_config: Dict[str, Any]) -> None:
        super().__init__()
        self._validate_config(classifier_config)
        self.classifier_config: Dict[str, Any] = classifier_config
        
        # Initialize MODERNBERT model
        self.bert: AutoModel = AutoModel.from_pretrained(bert_model_name)
        
        # Freeze BERT's weights
        for param in self.bert.parameters():
            param.requires_grad = False
            
        hidden_size = self.bert.config.hidden_size
        self.classifier = self._build_classifier(hidden_size, num_classes, self.classifier_config)

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validates the classifier configuration.

        Args:
            config: Configuration dictionary to validate.

        Raises:
            ValueError: If any configuration parameters are invalid.
        """
        valid_architectures = ['standard', 'plane_resnet']
        if 'architecture_type' not in config:
            config['architecture_type'] = 'standard'  # Default to standard
        
        if config['architecture_type'] not in valid_architectures:
            raise ValueError(f"architecture_type must be one of {valid_architectures}")
            
        # Add cls_pooling with default value if not present
        if 'cls_pooling' not in config:
            config['cls_pooling'] = True  # Default to True for both architectures
            
        if config['architecture_type'] == 'plane_resnet':
            required_keys = ['num_planes', 'plane_width']
            if not all(key in config for key in required_keys):
                raise ValueError(f"Missing required keys for plane_resnet: {required_keys}")
            
            if not isinstance(config['num_planes'], int) or config['num_planes'] < 1:
                raise ValueError("num_planes must be a positive integer")
            
            if not isinstance(config['plane_width'], int) or config['plane_width'] & (config['plane_width'] - 1) != 0:
                raise ValueError("plane_width must be a power of 2")
            
            # Remove regularization choice for plane_resnet - always uses batchnorm
            if 'regularization' in config:
                del config['regularization']
        else:
            required_keys = ['num_layers', 'hidden_dim', 'activation', 'dropout_rate', 'cls_pooling']
            if not all(key in config for key in required_keys):
                raise ValueError(f"Missing required keys in classifier_config. Required: {required_keys}")
            
            if not isinstance(config['num_layers'], int) or config['num_layers'] < 1:
                raise ValueError("num_layers must be a positive integer")
            
            if not isinstance(config['hidden_dim'], int) or config['hidden_dim'] < 1:
                raise ValueError("hidden_dim must be a positive integer")
            
            valid_activations = ['relu', 'leaky_relu', 'elu', 'gelu', 'selu']
            if config['activation'] not in valid_activations:
                raise ValueError(f"activation must be one of {valid_activations}")
            
            if not isinstance(config['dropout_rate'], float) or not 0 <= config['dropout_rate'] <= 1:
                raise ValueError("dropout_rate must be a float between 0 and 1")
            
            # Remove regularization choice for standard - always uses dropout
            if 'regularization' in config:
                del config['regularization']

    def _get_regularization(self, size: int, architecture_type: str, dropout_rate: float = 0.1) -> nn.Module:
        """Get regularization layer based on architecture type."""
        if architecture_type == 'plane_resnet':
            return nn.BatchNorm1d(size)
        else:  # standard architecture
            return nn.Dropout(dropout_rate)

    def _build_classifier(self, input_size: int, num_classes: int, config: Dict[str, Any]) -> nn.Module:
        if config['architecture_type'] == 'plane_resnet':
            logger.info("\nBuilding PlaneResNet classifier:")
            logger.info("  Architecture: %s", config['architecture_type'])
            logger.info("  Input size: %d", input_size)
            logger.info("  Plane width: %d", config['plane_width'])
            logger.info("  Number of planes: %d", config['num_planes'])
            logger.info("  Output size: %d", num_classes)
            logger.info("  Regularization: BatchNorm")  # Added this line
            
            return PlaneResNetHead(
                input_size=input_size,
                num_classes=num_classes,
                num_planes=config['num_planes'],
                plane_width=config['plane_width']
            )
        else:
            layer_sizes = self._calculate_layer_sizes(input_size, config['hidden_dim'], config['num_layers'], num_classes)
            
            logger.info("\nBuilding standard classifier:")
            logger.info("  Architecture: %s", config['architecture_type'])
            logger.info("  Learning rate: %s", config.get('learning_rate', 'default'))
            logger.info("  Weight decay: %s", config.get('weight_decay', 'default'))
            logger.info("  Input size: %s", layer_sizes[0])
            logger.info("  Hidden layers: %s", len(layer_sizes) - 2)
            logger.info("  Hidden dimension: %s", config['hidden_dim'])
            logger.info("  Activation: %s", config['activation'])
            logger.info("  Regularization: Dropout (rate: %s)", config['dropout_rate'])  # Changed this line
            
            layers = []
            for i in range(len(layer_sizes) - 1):
                current_size = layer_sizes[i]
                next_size = layer_sizes[i + 1]
                
                layers.append(nn.Linear(current_size, next_size))
                if i < len(layer_sizes) - 2:  # Hidden layer
                    layers.append(self._get_activation(config['activation']))
                    layers.append(self._get_regularization(next_size, 'standard', config['dropout_rate']))
            
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
        activation_map = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
            'selu': nn.SELU
        }
        return activation_map[activation]()
    
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
            if self.classifier_config['cls_pooling']:
                last_hidden_state = outputs[0]
                embeddings = last_hidden_state[:,0]
            else:
                embeddings = self._mean_pooling(outputs, attention_mask)
                embeddings = F.normalize(embeddings, p=2, dim=1)
            return self.classifier(embeddings)
        except Exception as e:
            raise RuntimeError(f"Error in forward pass: {str(e)}") from e

