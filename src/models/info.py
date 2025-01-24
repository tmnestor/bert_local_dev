from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelInfo:
    """Store model architecture information"""
    encoder_type: str
    bert_model_name: str
    hidden_size: int
    architecture_type: str
    head_dimensions: List[int]
    activation: str = 'gelu'
    dropout_rate: float = 0.1
    num_classes: int = 5
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], hidden_size: int) -> 'ModelInfo':
        """Create ModelInfo from config and known hidden size"""
        arch_type = config.get('architecture_type', 'standard')
        num_classes = config.get('num_classes', 5)
        
        if arch_type == 'standard':
            num_layers = config.get('num_layers', 2)
            hidden_dim = config.get('hidden_dim', 256)
            # Calculate dimensions including input and output
            dimensions = [hidden_size] + [hidden_dim] * num_layers + [num_classes]
        else:  # plane_resnet
            plane_width = config.get('plane_width', 256)
            dimensions = [hidden_size, plane_width, num_classes]
            
        return cls(
            encoder_type='BERTEncoder',
            bert_model_name=config.get('bert_model_name', 'unknown'),
            hidden_size=hidden_size,
            architecture_type=arch_type,
            head_dimensions=dimensions,
            activation=config.get('activation', 'gelu'),
            dropout_rate=config.get('dropout_rate', 0.1),
            num_classes=num_classes
        )
    
    def format_info(self, mode: str = 'train') -> str:
        """Format model information for display"""
        info = [
            "=" * 80,
            "Model Architecture Details",
            "=" * 80,
            "\nEncoder:",
            f"  - Type: {self.encoder_type}",
            f"  - Base Model: {self.bert_model_name}",
            f"  - Hidden Size: {self.hidden_size}",
            f"\nArchitecture: {self.architecture_type.upper()}"
        ]
        
        if self.architecture_type == 'standard':
            info.extend([
                "\nClassifier Head:",
                f"  - Layer Dimensions: {self.head_dimensions}",
                f"  - Activation: {self.activation.upper()}",
                f"  - Dropout Rate: {self.dropout_rate}"
            ])
        else:  # plane_resnet
            info.extend([
                "\nPlaneResNet Head:",
                f"  - Input Size: {self.head_dimensions[0]}",
                f"  - Plane Width: {self.head_dimensions[1]}",
                f"  - Output Size: {self.head_dimensions[-1]}"
            ])
        
        info.append("=" * 80)
        return "\n".join(info)
