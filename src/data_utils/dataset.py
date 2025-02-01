from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

class TextClassificationDataset(Dataset):
    """A PyTorch Dataset for text classification tasks.

    This dataset handles tokenization and preprocessing of text data for BERT-based
    classification models.

    Args:
        texts: List of input text strings to be classified.
        labels: List of integer labels corresponding to each text.
        tokenizer: A PreTrainedTokenizerBase instance for tokenization.
        max_seq_len: Maximum sequence length for tokenization (default: 512).

    Raises:
        ValueError: If texts or labels are empty, or have different lengths.
        ValueError: If max_seq_len is not positive.
        TypeError: If tokenizer is not an instance of PreTrainedTokenizerBase.
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer: PreTrainedTokenizerBase, max_seq_len: int = 512):
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
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError("tokenizer must be an instance of PreTrainedTokenizerBase")
        
        # Store validated inputs
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def __len__(self) -> int:
        """Returns the number of items in the dataset.

        Returns:
            int: The total number of text samples.
        """
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Gets a single tokenized and preprocessed text sample.

        Args:
            idx: Index of the text sample to retrieve.

        Returns:
            Dict containing:
                - input_ids: Tensor of token ids
                - attention_mask: Tensor of attention mask
                - label: Tensor of label id

        Raises:
            IndexError: If idx is out of range.
            RuntimeError: If there's an error processing the text.
        """
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
