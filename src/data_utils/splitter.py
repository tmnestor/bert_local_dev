"""Dataset splitting utilities for text classification tasks.

This module handles the creation and management of dataset splits, providing:
- Consistent train/validation/test splits with stratification
- Persistent storage of splits for reproducibility
- Automatic label encoding
- Split size verification and logging

The module ensures consistent data handling across training, validation,
and testing phases while maintaining proper stratification of labels.

Typical usage:
    ```python
    splitter = DataSplitter(data_dir)
    splits = splitter.load_splits(dataframe)
    train_texts, train_labels = splits.train_texts, splits.train_labels
    ```

Note:
    This module expects input data to have 'text' and 'category' columns,
    where 'category' contains the class labels.
"""

from pathlib import Path
from typing import List, Optional, Tuple
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
    train_ids: List[str]  # Add IDs field
    val_texts: List[str]
    val_labels: List[str]
    val_ids: List[str]  # Add IDs field
    test_texts: Optional[List[str]] = None
    test_labels: Optional[List[str]] = None
    test_ids: Optional[List[str]] = None  # Add IDs field
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
        self.label_encoder = LabelEncoder()  # Created here

    def _load_existing_split(
        self, split_name: str
    ) -> Optional[Tuple[List[str], List[int], List[str]]]:
        """Load existing split if available."""
        split_file = self.data_dir / f"{split_name}.csv"
        if split_file.exists():
            try:
                logger.debug("Found existing %s split: %s", split_name, split_file)
                df = pd.read_csv(split_file)
                texts = df["text"].tolist()
                labels = self.label_encoder.transform(df["category"])

                # Try both column names to handle legacy files
                if "Hash_ID" in df.columns:
                    ids = df["Hash_ID"].tolist()
                else:
                    ids = df["Hash_Id"].tolist()

                return texts, labels, ids
            except (
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                KeyError,
                ValueError,
            ) as e:
                logger.error("Error loading split %s: %s", split_name, str(e))
                return None
        return None

    def load_splits(self, df: pd.DataFrame) -> DataSplit:
        """Create or load train/val/test splits."""
        # Fit label encoder on full dataset first
        self.label_encoder.fit(df["category"])

        source_path = df.name if hasattr(df, "name") else None
        logger.info("Processing data from: %s", source_path)

        # Initialize variables at the top
        train_texts, val_texts, test_texts = None, None, None
        train_labels, val_labels, test_labels = None, None, None
        train_ids, val_ids, test_ids = None, None, None  # Add ID variables
        train_val_texts, train_val_labels, train_val_ids = None, None, None
        texts = df["text"].tolist()
        labels = self.label_encoder.transform(df["category"])

        # Try both column names to handle legacy files
        if "Hash_ID" in df.columns:
            ids = df["Hash_ID"].tolist()
        else:
            ids = df["Hash_Id"].tolist()

        # Try to load existing splits
        train_split = self._load_existing_split("train")
        val_split = self._load_existing_split("val")
        test_split = self._load_existing_split("test")

        if all(split is not None for split in [train_split, val_split, test_split]):
            logger.info("Using existing train/val/test splits")
            return DataSplit(
                train_texts=train_split[0],
                train_labels=train_split[1],
                train_ids=train_split[2],  # Add IDs
                val_texts=val_split[0],
                val_labels=val_split[1],
                val_ids=val_split[2],  # Add IDs
                test_texts=test_split[0],
                test_labels=test_split[1],
                test_ids=test_split[2],  # Add IDs
                label_encoder=self.label_encoder,  # Passed here
            )

        # Create new splits
        if test_split is None:
            (
                train_val_texts,
                test_texts,
                train_val_labels,
                test_labels,
                train_val_ids,
                test_ids,
            ) = train_test_split(
                texts,
                labels,
                ids,  # Add IDs to split
                test_size=0.2,
                random_state=42,
                stratify=labels,
            )
            # Save test split with IDs
            test_df = pd.DataFrame(
                {
                    "Hash_Id": test_ids,  # Changed from Hash_ID
                    "text": test_texts,
                    "category": self.label_encoder.inverse_transform(test_labels),
                }
            )
            test_df.to_csv(self.data_dir / "test.csv", index=False)

        # Create train/val splits
        if train_split is None or val_split is None:
            # Use train_val_texts from test split or full texts if no test split
            texts_to_split = train_val_texts if test_split is None else texts
            labels_to_split = train_val_labels if test_split is None else labels
            ids_to_split = train_val_ids if test_split is None else ids  # Add IDs

            train_texts, val_texts, train_labels, val_labels, train_ids, val_ids = (
                train_test_split(
                    texts_to_split,
                    labels_to_split,
                    ids_to_split,  # Add IDs
                    test_size=0.25,
                    random_state=42,
                    stratify=labels_to_split,
                )
            )

            # Save splits with IDs
            if train_split is None:
                train_df = pd.DataFrame(
                    {
                        "Hash_Id": train_ids,  # Changed from Hash_ID
                        "text": train_texts,
                        "category": self.label_encoder.inverse_transform(train_labels),
                    }
                )
                train_df.to_csv(self.data_dir / "train.csv", index=False)

            if val_split is None:
                val_df = pd.DataFrame(
                    {
                        "Hash_Id": val_ids,  # Changed from Hash_ID
                        "text": val_texts,
                        "category": self.label_encoder.inverse_transform(val_labels),
                    }
                )
                val_df.to_csv(self.data_dir / "val.csv", index=False)

        # Log split sizes
        total = len(texts)
        logger.debug("\nSplit sizes:")
        logger.debug("  Total: %d", total)
        logger.debug(
            "  Train: %d (%.1f%%)", len(train_texts), 100 * len(train_texts) / total
        )
        logger.debug(
            "  Val:   %d (%.1f%%)", len(val_texts), 100 * len(val_texts) / total
        )
        logger.debug(
            "  Test:  %d (%.1f%%)", len(test_texts), 100 * len(test_texts) / total
        )

        return DataSplit(
            train_texts=train_texts,
            val_texts=val_texts,
            test_texts=test_texts,
            train_labels=train_labels,
            val_labels=val_labels,
            test_labels=test_labels,
            train_ids=train_ids,  # Add IDs
            val_ids=val_ids,  # Add IDs
            test_ids=test_ids,  # Add IDs
            label_encoder=self.label_encoder,  # Passed here
        )
