# encoders.py
import logging
from typing import Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoModel
from .base import TextEncoder

logger = logging.getLogger(__name__)

class BERTEncoder(TextEncoder):
    """BERT encoder implementation"""
    def __init__(self, model_name: str = None, base_model = None, cls_pooling: bool = True):
        super().__init__()
        if base_model is not None:
            self.bert = base_model
            # print(f"{self.bert=}") #DELETE
        else:
            self.bert = AutoModel.from_pretrained(model_name)
        self.cls_pooling = cls_pooling
        # Log actual model dimensions
        logger.info(f"BERT Model Config: hidden_size={self.bert.config.hidden_size}")
        
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if self.cls_pooling:
            return outputs[0][:,0]
        return self._mean_pooling(outputs, attention_mask)
    
    def get_output_dim(self) -> int:
        """Get actual output dimension from BERT config"""
        hidden_size = getattr(self.bert.config, 'hidden_size', None)
        if hidden_size is None:
            # Fallback to checking last layer if config doesn't specify
            if hasattr(self.bert, 'pooler') and hasattr(self.bert.pooler, 'dense'):
                hidden_size = self.bert.pooler.dense.in_features
            else:
                raise ValueError("Could not determine BERT output dimension")
        return hidden_size
    
    def freeze(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = False
            
    def _mean_pooling(self, model_output: Dict[str, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return F.normalize(embeddings, p=2, dim=1)