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

def load_and_preprocess_data(config: ModelConfig, validation_mode: bool = False) -> Tuple[List[str], List[int], LabelEncoder]:
    """Load and preprocess data with train/val/test split
    
    Args:
        config: ModelConfig instance
        validation_mode: If True, returns test set, otherwise returns train/val sets
    """
    df = pd.read_csv(config.data_file)
    le = LabelEncoder()
    texts = df['text'].tolist()
    labels = le.fit_transform(df["category"]).tolist()
    
    # First split off test set (20% of data)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Save test split if it doesn't exist
    test_split_path = config.data_file.parent / "test_split.csv"
    if not test_split_path.exists():
        test_df = pd.DataFrame({
            'text': test_texts,
            'category': [le.inverse_transform([l])[0] for l in test_labels]
        })
        test_df.to_csv(test_split_path, index=False)
    
    if validation_mode:
        return test_texts, test_labels, le
    
    # For training, further split train into train/val (80/20 of remaining data)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    return train_texts, val_texts, train_labels, val_labels, le

def create_dataloaders(
    texts: List[str], 
    labels: List[int],
    config: ModelConfig,
    batch_size: int,
    validation_mode: bool = False
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    """Create train/val dataloaders or test dataloader"""
    tokenizer = BertTokenizer.from_pretrained(
        config.bert_model_name,
        clean_up_tokenization_spaces=True
    )
    
    if validation_mode:
        # Create single test dataloader
        test_dataset = TextClassificationDataset(texts, labels, tokenizer, config.max_length)
        return DataLoader(test_dataset, batch_size=batch_size)
    
    # For training, we expect train/val splits
    train_texts, val_texts = texts[0], texts[1]
    train_labels, val_labels = labels[0], labels[1]
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, config.max_length)
    
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
