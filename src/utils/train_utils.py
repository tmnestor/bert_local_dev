import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import torch
from tqdm.auto import tqdm

from ..utils.logging_manager import get_logger  # Change from setup_logger

logger = get_logger(__name__)  # Change to get_logger

def initialize_progress_bars(n_trials: int, num_batches: int = None) -> Tuple[tqdm, Optional[tqdm]]:
    """Initialize progress bars for training/tuning."""
    print("", flush=True)  # Force line break
    epoch_pbar = tqdm(
        total=n_trials, 
        desc='Epoch 1/5',
        position=0,
        leave=True,
        ncols=80,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
    )
    
    batch_pbar = None
    if num_batches:
        batch_pbar = tqdm(
            total=num_batches,
            desc='Training',
            position=1,
            leave=False,
            ncols=80,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]'
        )
    
    return epoch_pbar, batch_pbar

def save_model_state(
    state_dict: dict,
    path: Path,
    score: float,
    metadata: dict
) -> None:
    """Save model state with complete configuration."""
    save_dict = {
        'model_state_dict': state_dict,
        'config': {
            'classifier_config': metadata['classifier_config'],
            'model_config': {
                'bert_hidden_size': metadata.get('bert_hidden_size', 384),  # Default if not provided
                'num_classes': metadata['num_classes']
            }
        },
        'metric_value': score,
        **{k: v for k, v in metadata.items() if k not in ['classifier_config', 'num_classes']}
    }
    torch.save(save_dict, path)

def log_separator(logger_instance: logging.Logger) -> None:
    logger_instance.info("\n" + "="*80)
