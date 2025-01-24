from typing import Dict, Any
import torch.nn as nn
from ..models.info import ModelInfo  # Add ModelInfo import
from ..models.factory import ModelFactory

def format_model_info(model: nn.Module, config: Dict[str, Any], mode: str = 'train') -> str:
    """Format model architecture information."""
    # Create encoder to get dimensions
    encoder = ModelFactory.create_encoder(config)
    # Create model info
    model_info = ModelInfo.from_config(config, encoder.get_output_dim())
    return model_info.format_info(mode)

def count_parameters(model: nn.Module) -> str:
    """Count trainable parameters in model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return (
        f"\nModel Parameters:"
        f"\n  - Total: {total_params:,}"
        f"\n  - Trainable: {trainable_params:,}"
        f"\n  - Frozen: {total_params - trainable_params:,}"
        f"\n{'=' * 80}"
    )
