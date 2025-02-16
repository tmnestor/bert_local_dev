"""Data loading utilities."""

import logging
from dataclasses import dataclass
from pathlib import Path
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
        data_dir = config.data_dir
        # For validation mode, try to load test.csv directly
        if validation_mode and (data_dir / "test.csv").exists():
            logger.info("Loading test split directly")
            df = pd.read_csv(data_dir / "test.csv")
        else:
            df = pd.read_csv(config.data_file)

        df.name = str(config.data_file)
        logger.debug("Loaded %d rows", len(df))
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {str(e)}") from e

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
            test_texts=test_texts, test_labels=test_labels, label_encoder=label_encoder
        )

    # If no test file or not in validation mode, proceed with normal flow

    # Create data splitter and pass the loaded dataframe
    splitter = DataSplitter(config.data_dir)
    splits = splitter.load_splits(df)

    # Split info is now logged by DataSplitter
    if validation_mode:
        return DataBundle(
            test_texts=splits.test_texts,
            test_labels=splits.test_labels,
            label_encoder=splits.label_encoder,
        )
    return DataBundle(
        train_texts=splits.train_texts,
        val_texts=splits.val_texts,
        train_labels=splits.train_labels,
        val_labels=splits.val_labels,
        label_encoder=splits.label_encoder,
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

    # More descriptive error for missing bert_encoder_path
    if not hasattr(config, "bert_encoder_path"):
        raise ValueError(
            "bert_encoder_path must be specified in config. "
            "This should be set by ModelConfig or its subclasses."
        )

    encoder_path = Path(config.bert_encoder_path)
    if not encoder_path.exists():
        raise ValueError(
            f"BERT encoder not found at: {encoder_path}. "
            f"Please ensure it exists in the correct location."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(encoder_path),
            local_files_only=True,
            trust_remote_code=False,  # Additional safety measure
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load local tokenizer: {str(e)}") from e

    # Create datasets with max_seq_len from config
    if isinstance(texts[0], list):  # Multiple sets (train/val)
        train_dataset = TextClassificationDataset(
            texts[0], labels[0], tokenizer, config.max_seq_len
        )
        val_dataset = TextClassificationDataset(
            texts[1], labels[1], tokenizer, config.max_seq_len
        )
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size),
        )
    else:  # Single set (test)
        dataset = TextClassificationDataset(
            texts, labels, tokenizer, config.max_seq_len
        )
        return DataLoader(dataset, batch_size=batch_size)
