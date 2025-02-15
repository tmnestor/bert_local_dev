"""BERT-based text classifier model implementation.

This module provides the core classifier architecture, including:
- BERT encoder integration
- Custom classification head with configurable layers
- Dynamic activation function support
- Dropout regularization
- Forward pass implementation
- State management utilities

The classifier uses a pre-trained BERT model as the encoder and adds
a configurable classification head on top for text classification tasks.

Typical usage:
    ```python
    classifier = BERTClassifier(
        bert_model_name="bert-base-uncased",
        num_classes=5,
        config={
            "hidden_dim": [768, 256],
            "dropout_rate": 0.3,
            "activation": "gelu"
        }
    )
    outputs = classifier(input_ids=input_ids, attention_mask=attention_mask)
    ```

Attributes:
    SUPPORTED_ACTIVATIONS (Dict[str, Type[nn.Module]]): Mapping of activation function
        names to their PyTorch implementations.

Note:
    The BERT encoder should be pre-trained and available either locally
    or through the Hugging Face model hub.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from ..utils.logging_manager import get_logger

logger = get_logger(__name__)


class BERTClassifier(nn.Module):
    """BERT-based text classifier with configurable feed-forward head."""

    def __init__(
        self,
        bert_encoder_path: Optional[str],
        num_classes: int,
        classifier_config: Dict[str, Any],
    ) -> None:
        """Initialize the classifier."""
        super().__init__()
        self._validate_config(classifier_config)
        self.classifier_config = classifier_config

        # Get BERT hidden size from config
        self.bert_hidden_size = classifier_config.get("bert_hidden_size", 384)

        if bert_encoder_path:  # Training mode
            logger.info(
                "Training Mode: Loading BERT encoder from %s", bert_encoder_path
            )
            self.bert = AutoModel.from_pretrained(bert_encoder_path)
            self.bert_hidden_size = self.bert.config.hidden_size
        else:  # Inference mode - Initialize with default model
            logger.info(
                "Inference Mode: Initializing BERT (will be overwritten by checkpoint)"
            )
            self.bert = AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.bert_hidden_size = self.bert.config.hidden_size

        # Build classifier layers with confirmed bert_hidden_size
        self.classifier = self._build_classifier(self.bert_hidden_size, num_classes)

        # Add model architecture logging
        logger.debug("\nModel Architecture:")
        logger.debug("=" * 50)
        logger.debug("BERT hidden size: %d", self.bert_hidden_size)
        logger.debug("Hidden layers: %s", classifier_config["hidden_dim"])
        logger.debug("Activation: %s", classifier_config.get("activation", "gelu"))
        logger.debug("Dropout: %.3f", classifier_config.get("dropout_rate", 0.1))
        logger.debug("Total parameters: %d", sum(p.numel() for p in self.parameters()))
        logger.debug(
            "Trainable parameters: %d",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )
        logger.debug("=" * 50)

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Dict, num_classes: int, device: str = "cpu"
    ) -> "BERTClassifier":
        """Create model instance from checkpoint."""
        config = checkpoint.get("config", {})
        if not config:
            raise ValueError("No configuration found in checkpoint")

        # Create model in inference mode (bert_encoder_path=None)
        model = cls(
            bert_encoder_path=None, num_classes=num_classes, classifier_config=config
        )

        # Load full state dict
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load model state: {str(e)}. "
                "This might be due to incompatible model architectures."
            ) from e

        model.to(device)
        model.eval()

        return model

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validates the classifier configuration."""
        # Support both hidden_dims (new) and hidden_dim (old) for backward compatibility
        if "hidden_dims" in config:
            config["hidden_dim"] = config["hidden_dims"]
        elif "hidden_dim" not in config:
            raise ValueError(
                "hidden_dims or hidden_dim is required in classifier_config"
            )

        # Convert single integer to list for backward compatibility
        if isinstance(config["hidden_dim"], int):
            config["hidden_dim"] = [config["hidden_dim"]]

        # Convert hidden_dim to list if it's a tuple
        if isinstance(config["hidden_dim"], tuple):
            config["hidden_dim"] = list(config["hidden_dim"])

        if not isinstance(config["hidden_dim"], (list, tuple)):
            raise ValueError("hidden_dims must be a list or tuple of integers")

        if not all(isinstance(dim, int) and dim > 0 for dim in config["hidden_dim"]):
            raise ValueError("all hidden_dims values must be positive integers")

        if (
            not isinstance(config.get("dropout_rate", 0.1), float)
            or not 0 <= config["dropout_rate"] <= 1
        ):
            raise ValueError("dropout_rate must be a float between 0 and 1")

    def _build_classifier(self, input_size: int, num_classes: int) -> nn.Module:
        """Build the classifier layers."""
        hidden_dims = self.classifier_config["hidden_dim"]
        dropout_rate = self.classifier_config.get("dropout_rate", 0.1)
        activation = self.classifier_config.get("activation", "gelu")

        # Build layer sizes list
        layer_sizes = [input_size] + hidden_dims + [num_classes]

        # Add newline before the log message
        logger.debug("\nBuilding classifier with layers: %s", layer_sizes)
        logger.debug("Activation: %s, Dropout: %.3f", activation, dropout_rate)

        activation_fn = self._get_activation(activation)

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation_fn)
                layers.append(nn.Dropout(dropout_rate))

        return nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "prelu": nn.PReLU(),
        }
        if name.lower() not in activations:
            logger.warning("Unknown activation '%s', using GELU", name)
            return nn.GELU()
        return activations[name.lower()]

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the model."""
        try:
            if self.bert is None:
                raise RuntimeError(
                    "BERT model not initialized. Load from checkpoint first."
                )

            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Use mean pooling
            embeddings = self._mean_pooling(outputs, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return self.classifier(embeddings)
        except Exception as e:
            raise RuntimeError(f"Error in forward pass: {str(e)}") from e

    def _mean_pooling(
        self, model_output: Dict[str, torch.Tensor], attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Perform mean pooling on token embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
