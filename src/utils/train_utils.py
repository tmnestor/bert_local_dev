import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

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
    """Load and preprocess data using DataSplitter.

    Args:
        config: ModelConfig instance containing data parameters.
        validation_mode: If True, return test set only.

    Returns:
        In training mode:
            - train_texts, val_texts: Lists of text samples
            - train_labels, val_labels: Lists of integer labels
            - label_encoder: Fitted LabelEncoder
        In validation mode:
            - test_texts: List of text samples
            - test_labels: List of integer labels
            - label_encoder: Fitted LabelEncoder

    Raises:
        ValueError: If num_classes doesn't match dataset.
        FileNotFoundError: If data file not found.
    """
    splitter = DataSplitter(config.data_file.parent)
    
    try:
        # Try to load existing splits first
        splits = splitter.load_splits()
        logger.info("Using existing data splits")
    except FileNotFoundError:
        # Create new splits if they don't exist
        logger.info("Creating new data splits...")
        splits = splitter.create_splits(config.data_file)
    
    # Set num_classes if not explicitly specified
    if config.num_classes is None:
        config.num_classes = splits.num_classes
    elif config.num_classes != splits.num_classes:
        raise ValueError(f"Specified num_classes ({config.num_classes}) does not match dataset ({splits.num_classes})")
    
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
    validation_mode: bool = False,
    drop_last: bool = True  # Change default to True
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    """Create PyTorch DataLoaders for training or validation.

    Args:
        texts: Text samples or [train_texts, val_texts] in training mode.
        labels: Labels or [train_labels, val_labels] in training mode.
        config: ModelConfig instance.
        batch_size: Batch size for DataLoader.
        validation_mode: If True, create single test DataLoader.
        drop_last: If True, drop the last incomplete batch.

    Returns:
        In training mode: (train_dataloader, val_dataloader)
        In validation mode: test_dataloader

    Raises:
        ValueError: If texts/labels format is invalid.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.bert_model_name,
        clean_up_tokenization_spaces=True
    )
    
    if validation_mode:
        test_dataset = TextClassificationDataset(
            texts, labels, tokenizer, 
            max_seq_len=config.max_seq_len  # Updated from max_length
        )
        return DataLoader(test_dataset, batch_size=batch_size, drop_last=drop_last)
    
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
    
    # Ensure minimum batch size
    effective_batch_size = max(2, batch_size)  # Minimum batch size of 2
    
    # Create and return dataloaders with consistent batch handling
    return (
        DataLoader(
            train_dataset, 
            batch_size=effective_batch_size, 
            shuffle=True, 
            drop_last=True  # Always drop last for training
        ),
        DataLoader(
            val_dataset, 
            batch_size=effective_batch_size, 
            drop_last=False  # Keep all samples for validation
        )
    )

def initialize_progress_bars(n_trials: int, num_epochs: int) -> Tuple[tqdm, tqdm]:
    """Initialize progress bars for training/tuning.

    Args:
        n_trials: Number of optimization trials.
        num_epochs: Number of epochs per trial.

    Returns:
        Tuple of (trial_progress_bar, epoch_progress_bar).
    """
    trial_pbar = tqdm(total=n_trials, desc='Trials', position=0)
    epoch_pbar = tqdm(total=num_epochs, desc='Epochs', position=1, leave=False)
    return trial_pbar, epoch_pbar

def save_model_state(
    model_state: Dict[str, Any],
    save_path: Path,
    metric_value: float,
    config: Dict[str, Any]
) -> None:
    """Save model checkpoint with metadata.

    Args:
        model_state: Model state dict.
        save_path: Path to save checkpoint.
        metric_value: Performance metric value.
        config: Model configuration dict.

    Raises:
        IOError: If saving fails.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model_state,
        'config': config,
        'metric_value': metric_value,
        'num_classes': config['num_classes']
    }, save_path)

def load_best_trial_info(best_trials_dir: Path, study_name: str) -> Optional[dict]:
    """Load information about best trial from optimization."""
    trial_path = best_trials_dir / f"best_trial_{study_name}.pt"
    if not trial_path.exists():
        return None
        
    try:
        # Use weights_only=False since we need to load configuration data
        trial_info = torch.load(trial_path, map_location='cpu', weights_only=False)
        logger.info("Loaded best trial info:")
        logger.info("  Score: %.4f", trial_info['value'])
        logger.info("  Trial: %d", trial_info['number'])
        logger.info("  Model: %s", trial_info['model_path'])
        
        # Ensure architecture_type is present in params
        if 'params' in trial_info and 'architecture_type' not in trial_info['params']:
            trial_info['params']['architecture_type'] = 'standard'  # Set default
            
        return trial_info
    except Exception as e:
        logger.error(f"Failed to load trial info: {e}")
        return None

def log_separator(logger_instance: logging.Logger) -> None:
    logger_instance.info("\n" + "="*80)
