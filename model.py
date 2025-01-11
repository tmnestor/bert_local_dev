import torch
from torch import nn
from transformers import BertModel
from typing import Dict, List, Union

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name: str, num_classes: int, classifier_config: Dict):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Freeze BERT's weights
        for param in self.bert.parameters():
            param.requires_grad = False
            
        hidden_size = self.bert.config.hidden_size
        self.classifier = self._build_classifier(hidden_size, num_classes, classifier_config)

    def _build_classifier(self, input_size: int, num_classes: int, config: Dict) -> nn.Sequential:
        layers = []
        current_size = input_size
        
        activation_map = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
            'selu': nn.SELU
        }
        
        for i in range(config['num_layers']):
            next_size = current_size // 2
            layers.append(nn.Linear(current_size, next_size))
            
            # Add activation
            layers.append(activation_map[config['activation']]())
            
            # Add regularization
            if config['regularization'] == 'dropout':
                layers.append(nn.Dropout(config['dropout_rate']))
            elif config['regularization'] == 'batchnorm':
                layers.append(nn.BatchNorm1d(next_size))
                
            current_size = next_size
        
        # Final classification layer
        layers.append(nn.Linear(current_size, num_classes))
        
        return nn.Sequential(*layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)
