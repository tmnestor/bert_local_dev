import logging
from typing import Dict, Any
import torch
from transformers import AutoModel

from .model import BERTClassifier
from .base import TextEncoder, ClassifierHead
from .encoders import BERTEncoder  # Make sure to import BERTEncoder
from .heads import StandardClassifierHead, PlaneResNetHead
from .info import ModelInfo  # Add this import

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class for creating model instances"""
    
    @staticmethod
    def create_encoder(config: Dict[str, Any]) -> BERTEncoder:  # Change return type hint
        """Create encoder instance"""
        # Create BERT model with model name
        bert_model = AutoModel.from_pretrained(config['bert_model_name'])
        return BERTEncoder(base_model=bert_model, cls_pooling=config.get('cls_pooling', True))

    @staticmethod
    def create_head(config: Dict[str, Any], input_dim: int) -> ClassifierHead:
        """Create classifier head"""
        if config['architecture_type'] == 'standard':
            # Handle both nested and flat config structures
            head_config = config.get('config', {})  # Get nested config if it exists
            if not head_config:  # If no nested config, use flat structure
                head_config = {
                    'num_layers': config.get('num_layers', 2),
                    'hidden_dim': config.get('hidden_dim', 256),
                    'activation': config.get('activation', 'gelu'),
                    'dropout_rate': config.get('dropout_rate', 0.1)
                }
            return StandardClassifierHead(
                input_dim=input_dim,
                num_classes=config['num_classes'],
                config=head_config
            )
        elif config['architecture_type'] == 'plane_resnet':
            return PlaneResNetHead(
                input_dim=input_dim,
                num_classes=config['num_classes'],
                num_planes=config.get('num_planes', 8),
                plane_width=config.get('plane_width', 256)
            )
        else:
            raise ValueError(f"Unknown architecture type: {config['architecture_type']}")

    @staticmethod
    def create_model(config: Dict[str, Any]) -> BERTClassifier:
        """Create BERT classifier model with fresh initialization"""
        # Create encoder first to get dimensions
        encoder = ModelFactory.create_encoder(config)
        input_dim = encoder.get_output_dim()
        
        # Create model info
        model_info = ModelInfo.from_config(config, input_dim)
        logger.info("\n%s", model_info.format_info())
        
        # Create head using model info
        head = ModelFactory.create_head(config, model_info.hidden_size)
        
        # Create and return model
        model = BERTClassifier(encoder=encoder, head=head)
        model.info = model_info  # Store info with model
        
        return model