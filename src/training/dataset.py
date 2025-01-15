from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_length: int) -> None:
        """Initialize dataset with input validation"""
        # Validate inputs
        if len(texts) == 0:
            raise ValueError("texts cannot be empty")
        if len(labels) == 0:
            raise ValueError("labels cannot be empty")
        if len(texts) != len(labels):
            raise ValueError(f"texts and labels must have the same length, got {len(texts)} texts and {len(labels)} labels")
        if max_length < 1:
            raise ValueError("max_length must be positive")
        if not isinstance(tokenizer, BertTokenizer):
            raise TypeError("tokenizer must be an instance of BertTokenizer")
        
        # Store validated inputs
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not 0 <= idx < len(self.texts):
            raise IndexError(f"Index {idx} out of range")
            
        try:
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=self.max_length,
                padding='max_length',
                truncation=True
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label)
            }
        except Exception as e:
            raise RuntimeError(f"Error processing item {idx}: {str(e)}") from e
