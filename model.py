from typing import Dict, List, Union, Optional, Any
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name: str, num_classes: int, classifier_config: Dict[str, Any]) -> None:
        super().__init__()
        self._validate_config(classifier_config)
        self.classifier_config: Dict[str, Any] = classifier_config
        self.bert: BertModel = BertModel.from_pretrained(bert_model_name)
        
        # Freeze BERT's weights
        for param in self.bert.parameters():
            param.requires_grad = False
            
        hidden_size = self.bert.config.hidden_size
        self.classifier = self._build_classifier(hidden_size, num_classes, self.classifier_config)

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate classifier configuration."""
        required_keys = ['num_layers', 'activation', 'regularization', 'dropout_rate', 'cls_pooling']
        if not all(key in config for key in required_keys):
            raise ValueError(f"Missing required keys in classifier_config. Required: {required_keys}")
        
        if not isinstance(config['num_layers'], int) or config['num_layers'] < 1:
            raise ValueError("num_layers must be a positive integer")
        
        valid_activations = ['relu', 'leaky_relu', 'elu', 'gelu', 'selu']
        if config['activation'] not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}")
        
        valid_regularizations = ['dropout', 'batchnorm']
        if config['regularization'] not in valid_regularizations:
            raise ValueError(f"regularization must be one of {valid_regularizations}")
        
        if not isinstance(config['dropout_rate'], float) or not 0 <= config['dropout_rate'] <= 1:
            raise ValueError("dropout_rate must be a float between 0 and 1")
        
        if not isinstance(config['cls_pooling'], bool):
            raise ValueError("cls_pooling must be a boolean")

    def _build_classifier(self, input_size: int, num_classes: int, config: Dict[str, Any]) -> nn.Sequential:
        layers = []
        current_size = input_size
        
        for i in range(config['num_layers']):
            next_size = current_size // 2
            layers.append(nn.Linear(current_size, next_size))
            layers.append(self._get_activation(config['activation']))
            layers.append(self._get_regularization(config['regularization'], config['dropout_rate'], next_size))
            current_size = next_size
        
        layers.append(nn.Linear(current_size, num_classes))
        return nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        activation_map = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
            'selu': nn.SELU
        }
        return activation_map[activation]()
    
    def _get_regularization(self, regularization: str, dropout_rate: float, size: int) -> nn.Module:
        if regularization == 'dropout':
            return nn.Dropout(dropout_rate)
        elif regularization == 'batchnorm':
            return nn.BatchNorm1d(size)
        return nn.Identity()

    def _mean_pooling(self, model_output: Dict[str, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        try:
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

