"""Data loading utilities."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..config.configuration import ModelConfig  # Changed from config to configuration
from .dataset import TextClassificationDataset
from .splitter import DataSplitter

logger = logging.getLogger(__name__)

@dataclass
class DataBundle:
    """Container for dataset splits and label encoder."""
    train_texts: Optional[List[str]] = None
    val_texts: Optional[List[str]] = None
    train_labels: Optional[List[int]] = None
    val_labels: Optional[List[int]] = None
    test_texts: Optional[List[str]] = None
    test_labels: Optional[List[int]] = None
    label_encoder: Optional[LabelEncoder] = None

def load_and_preprocess_data(
    config: ModelConfig, validation_mode: bool = False
) -> DataBundle:
    """Load and preprocess data using DataSplitter."""
    logger.info("Loading data from: %s", config.data_file)

    try:
        if not config.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {config.data_file}")

        df = pd.read_csv(config.data_file)
        df.name = config.data_file  # Store full path
        logger.debug("Loaded %d rows from data file", len(df))
    except Exception as e:
        raise RuntimeError(f"Failed to load data file: {str(e)}") from e

    # Check for existing test split first if in validation mode
    test_file = config.data_dir / "test.csv"
    if validation_mode and test_file.exists():
        logger.info("Found existing test split: %s", test_file)
        # Load test data directly
        test_df = pd.read_csv(test_file)
        # Initialize label encoder with all possible classes
        label_encoder = LabelEncoder()
        label_encoder.fit(pd.read_csv(config.data_file)["category"])
        # Transform test labels
        test_texts = test_df["text"].tolist()
        test_labels = label_encoder.transform(test_df["category"])
        return DataBundle(
            test_texts=test_texts,
            test_labels=test_labels,
            label_encoder=label_encoder
        )

    # If no test file or not in validation mode, proceed with normal flow

    # Create data splitter and pass the loaded dataframe
    splitter = DataSplitter(config.data_dir)
    splits = splitter.load_splits(df)

    # Log split information
    logger.info("\nData Split Information:")
    total = len(splits.train_texts) + len(splits.val_texts) + len(splits.test_texts)
    logger.info("Total samples: %d", total)
    logger.info(
        "Training: %d (%.1f%%)",
        len(splits.train_texts),
        100 * len(splits.train_texts) / total,
    )
    logger.info(
        "Validation: %d (%.1f%%)",
        len(splits.val_texts),
        100 * len(splits.val_texts) / total,
    )
    logger.info(
        "Test: %d (%.1f%%)",
        len(splits.test_texts),
        100 * len(splits.test_texts) / total,
    )

    if validation_mode:
        return DataBundle(
            test_texts=splits.test_texts,
            test_labels=splits.test_labels,
            label_encoder=splits.label_encoder
        )
    return DataBundle(
        train_texts=splits.train_texts,
        val_texts=splits.val_texts,
        train_labels=splits.train_labels,
        val_labels=splits.val_labels,
        label_encoder=splits.label_encoder
    )


def create_dataloaders(
    texts: List[str],
    labels: List[int],
    config: ModelConfig,
    batch_size: Optional[int] = None,
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    """Create PyTorch DataLoaders."""
    if batch_size is None:
        batch_size = config.batch_size

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)

    # Handle different input formats
    if isinstance(texts[0], list):  # Multiple sets (train/val)
        train_dataset = TextClassificationDataset(
            texts[0], labels[0], tokenizer, config.max_seq_len
        )
        val_dataset = TextClassificationDataset(
            texts[1], labels[1], tokenizer, config.max_seq_len
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader
    else:  # Single set (test)
        dataset = TextClassificationDataset(
            texts, labels, tokenizer, config.max_seq_len
        )
        return DataLoader(dataset, batch_size=batch_size)
