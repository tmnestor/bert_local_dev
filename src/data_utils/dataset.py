from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TextClassificationDataset(Dataset):
    """Dataset for text classification tasks.

    Handles text tokenization and truncation/padding to max_seq_len.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int,
    ) -> None:
        """Initialize dataset.

        Args:
            texts: List of input texts
            labels: List of integer labels
            tokenizer: BERT tokenizer to use
            max_seq_len: Maximum sequence length for padding/truncation
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )  # Set device

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized and padded item.

        The tokenizer handles:
        1. Text -> tokens
        2. Truncation to max_seq_len
        3. Padding to max_seq_len
        """
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )

        # Return tensors on CPU; remove .to(self.device)
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
        }
