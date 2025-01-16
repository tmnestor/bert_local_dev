from typing import Tuple, List, Dict, Any, Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import logging

from ..config.config import ModelConfig 
from ..training.dataset import TextClassificationDataset
from ..utils.logging_manager import setup_logger
from .data_splitter import DataSplitter, DataSplit

# Add logger initialization
logger = setup_logger(__name__)

def load_and_preprocess_data(config: ModelConfig, validation_mode: bool = False) -> Union[
    Tuple[List[str], List[str], List[int], List[int], LabelEncoder],  # Training mode
    Tuple[List[str], List[int], LabelEncoder]  # Validation mode
]:
    """Load and preprocess data using DataSplitter"""
    splitter = DataSplitter(config.data_file.parent)
    
    try:
        # Try to load existing splits first
        splits = splitter.load_splits()
        logger.info("Using existing data splits")
    except FileNotFoundError:
        # Create new splits if they don't exist
        logger.info("Creating new data splits...")
        splits = splitter.create_splits(config.data_file)
    
    if validation_mode:
        logger.info(f"Using test set with {len(splits.test_texts)} samples")
        return splits.test_texts, splits.test_labels, splits.label_encoder
    
    logger.info(f"Split sizes: {len(splits.train_texts)} train, {len(splits.val_texts)} validation")
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
    """Create dataloaders from texts and labels"""
    tokenizer = BertTokenizer.from_pretrained(
        config.bert_model_name,
        clean_up_tokenization_spaces=True
    )
    
    if validation_mode:
        test_dataset = TextClassificationDataset(
            texts, labels, tokenizer, 
            max_seq_len=config.max_seq_len  # Updated from max_length
        )
        return DataLoader(test_dataset, batch_size=batch_size)
    
    # For training, texts and labels are passed as lists containing train and val splits
    if not isinstance(texts, (list, tuple)) or len(texts) != 2:
        raise ValueError("Expected texts to be a list/tuple of [train_texts, val_texts]")
    if not isinstance(labels, (list, tuple)) or len(labels) != 2:
        raise ValueError("Expected labels to be a list/tuple of [train_labels, val_labels]")
    
    train_texts, val_texts = texts
    train_labels, val_labels = labels
    
    # Create datasets - update to use max_seq_len
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, config.max_seq_len)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, config.max_seq_len)
    
    # Create and return dataloaders
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size)
    )

def initialize_progress_bars(n_trials: int, num_epochs: int) -> Tuple[tqdm, tqdm]:
    """Initialize progress bars for training/tuning"""
    trial_pbar = tqdm(total=n_trials, desc='Trials', position=0)
    epoch_pbar = tqdm(total=num_epochs, desc='Epochs', position=1, leave=False)
    return trial_pbar, epoch_pbar

def save_model_state(
    model_state: Dict[str, Any],
    save_path: Path,
    metric_value: float,
    config: Dict[str, Any]
) -> None:
    """Save model checkpoint with metadata"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model_state,
        'config': config,
        'metric_value': metric_value,
        'num_classes': config['num_classes']
    }, save_path)



def log_separator(logger: logging.Logger) -> None:

    logger.info("\n" + "="*80)
