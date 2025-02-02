import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..utils.logging_manager import get_logger  # Changed from setup_logger

logger = get_logger(__name__)  # Changed to get_logger


def preprocess_text(text: str) -> str:
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    return " ".join(text.split())  # Normalize whitespace


@dataclass
class DataSplit:
    """Data split container for text classification tasks."""

    train_texts: List[str]
    train_labels: List[str]
    val_texts: List[str]
    val_labels: List[str]
    test_texts: Optional[List[str]] = None
    test_labels: Optional[List[str]] = None
    label_encoder: Optional[LabelEncoder] = None
    num_classes: Optional[int] = None

    def __post_init__(self):
        if self.label_encoder and not self.num_classes:
            self.num_classes = len(self.label_encoder.classes_)


class DataSplitter:
    """Handles dataset splitting and persistence."""

    def __init__(self, data_dir: Path):
        """Initialize DataSplitter.

        Args:
            data_dir: Directory containing the data file
        """
        self.data_dir = Path(data_dir)
        self.label_encoder = LabelEncoder()

    def load_splits(self, df: pd.DataFrame) -> DataSplit:
        """Create train/val/test splits from dataframe.
        
        Args:
            df: DataFrame containing 'text' and 'category' columns
        """
        texts = df['text'].tolist()
        labels = self.label_encoder.fit_transform(df['category'])

        # First split out test set (20%)
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Then split remaining into train/val (75%/25% of remaining = 60%/20% of total)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels, test_size=0.25, random_state=42, stratify=train_val_labels
        )

        # Log split sizes
        total = len(texts)
        logger.debug("\nSplit sizes:")
        logger.debug("  Total: %d", total)
        logger.debug("  Train: %d (%.1f%%)", len(train_texts), 100 * len(train_texts) / total)
        logger.debug("  Val:   %d (%.1f%%)", len(val_texts), 100 * len(val_texts) / total)
        logger.debug("  Test:  %d (%.1f%%)", len(test_texts), 100 * len(test_texts) / total)

        return DataSplit(
            train_texts=train_texts,
            val_texts=val_texts,
            test_texts=test_texts,
            train_labels=train_labels,
            val_labels=val_labels,
            test_labels=test_labels,
            label_encoder=self.label_encoder
        )
