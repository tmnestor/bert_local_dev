from typing import Tuple, List, Dict, Any
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

def load_and_preprocess_data(config: ModelConfig) -> Tuple[List[str], List[int], LabelEncoder]:
    """Load and preprocess data from config.data_file"""
    df = pd.read_csv(config.data_file)
    le = LabelEncoder()
    texts = df['text'].tolist()
    labels = le.fit_transform(df["category"]).tolist()
    
    # Update config with actual number of classes
    num_classes = len(set(labels))
    if config.num_classes != num_classes:
        config.num_classes = num_classes
    
    return texts, labels, le

def create_dataloaders(
    texts: List[str], 
    labels: List[int],
    config: ModelConfig,
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders"""
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    tokenizer = BertTokenizer.from_pretrained(
        config.bert_model_name,
        clean_up_tokenization_spaces=True
    )
    
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
