from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: PreTrainedTokenizer, max_seq_len: int = 512):
        """Initialize dataset with input validation"""
        # Validate inputs
        if len(texts) == 0:
            raise ValueError("texts cannot be empty")
        if len(labels) == 0:
            raise ValueError("labels cannot be empty")
        if len(texts) != len(labels):
            raise ValueError(f"texts and labels must have the same length, got {len(texts)} texts and {len(labels)} labels")
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if not isinstance(tokenizer, PreTrainedTokenizer):
            raise TypeError("tokenizer must be an instance of PreTrainedTokenizer")
        
        # Store validated inputs
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not 0 <= idx < len(self.texts):
            raise IndexError(f"Index {idx} out of range")
            
        try:
            text = str(self.texts[idx])
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        except Exception as e:
            raise RuntimeError(f"Error processing item {idx}: {str(e)}") from e
