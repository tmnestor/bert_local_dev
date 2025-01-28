"""Data loading utilities."""

from pathlib import Path
from typing import List, Tuple, Union, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder  # Add this import

from ..config.config import ModelConfig
from .dataset import TextClassificationDataset
from .splitter import DataSplitter
from .validation import validate_dataset
from ..utils.logging_manager import setup_logger

logger = setup_logger(__name__)

def load_and_preprocess_data(config: ModelConfig, validation_mode: bool = False) -> Union[
    Tuple[List[str], List[str], List[int], List[int], LabelEncoder],  # Training mode
    Tuple[List[str], List[int], LabelEncoder]  # Validation mode
]:
    """Load and preprocess data using DataSplitter."""
    # Validate input data first
    validate_dataset(config.data_file)
    
    data_path = Path(config.data_file)
    splitter = DataSplitter(data_path.parent)  # Pass parent directory
    splits = splitter.load_splits()
    
    if config.num_classes is None:
        config.num_classes = splits.num_classes
    elif config.num_classes != splits.num_classes:
        raise ValueError(
            f"Specified num_classes ({config.num_classes}) "
            f"does not match dataset ({splits.num_classes})"
        )
    
    # Log split information
    logger.info("\nData Split Information:")
    total = len(splits.train_texts) + len(splits.val_texts) + len(splits.test_texts)
    logger.info(f"Total samples: {total}")
    logger.info(f"Training: {len(splits.train_texts)} ({100 * len(splits.train_texts) / total:.1f}%)")
    logger.info(f"Validation: {len(splits.val_texts)} ({100 * len(splits.val_texts) / total:.1f}%)")
    logger.info(f"Test: {len(splits.test_texts)} ({100 * len(splits.test_texts) / total:.1f}%)")
    
    if validation_mode:
        return splits.test_texts, splits.test_labels, splits.label_encoder
    
    return (
        splits.train_texts,
        splits.val_texts,
        splits.train_labels,
        splits.val_labels,
        splits.label_encoder
    )

def create_dataloaders(
    texts: List[str], 
    labels: List[int],
    config: ModelConfig,
    batch_size: int,
    validation_mode: bool = False
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    """Create PyTorch DataLoaders.
    
    Args:
        texts: Text samples or [train_texts, val_texts]
        labels: Labels or [train_labels, val_labels]
        config: ModelConfig instance
        batch_size: Batch size
        validation_mode: If True, create single test DataLoader
        
    Returns:
        Training mode: (train_dataloader, val_dataloader)
        Validation mode: test_dataloader
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.bert_model_name,
        clean_up_tokenization_spaces=True
    )
    
    if validation_mode:
        test_dataset = TextClassificationDataset(
            texts, labels, tokenizer, 
            max_seq_len=config.max_seq_len
        )
        return DataLoader(test_dataset, batch_size=batch_size)
    
    if not isinstance(texts, (list, tuple)) or len(texts) != 2:
        raise ValueError("Expected texts to be [train_texts, val_texts]")
    if not isinstance(labels, (list, tuple)) or len(labels) != 2:
        raise ValueError("Expected labels to be [train_labels, val_labels]")
    
    train_texts, val_texts = texts
    train_labels, val_labels = labels
    
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, config.max_seq_len)
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, tokenizer, config.max_seq_len)
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size)
    )
