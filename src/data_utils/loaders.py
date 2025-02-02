"""Data loading utilities."""

from typing import List, Tuple, Union, Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging

from ..config.config import ModelConfig
from .dataset import TextClassificationDataset
from .splitter import DataSplitter
from ..utils.logging_manager import get_logger

logger = logging.getLogger(__name__)


def load_and_preprocess_data(
    config: ModelConfig, validation_mode: bool = False
) -> Tuple:
    """Load and preprocess data using DataSplitter."""
    logger.debug("Loading data:")
    logger.debug("  Data path: %s", config.data_file)
    logger.debug("  Absolute path: %s", config.data_file.absolute())
    
    # Load CSV file
    try:
        df = pd.read_csv(config.data_file)
        logger.debug("  CSV file loaded successfully")
        logger.debug("  Number of rows: %d", len(df))
    except Exception as e:
        raise RuntimeError(f"Failed to load data file: {str(e)}") from e

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
        return splits.test_texts, splits.test_labels, splits.label_encoder
    return (
        splits.train_texts,
        splits.val_texts,
        splits.train_labels,
        splits.val_labels,
        splits.label_encoder,
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
