import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from tqdm.auto import tqdm

from ..utils.logging_manager import setup_logger

logger = setup_logger(__name__)

def initialize_progress_bars(n_trials: int, num_epochs: int) -> Tuple[tqdm, tqdm]:
    """Initialize progress bars for training/tuning."""
    # Add line break before progress bars
    print("\n", flush=True)  # Force line break and flush output
    trial_pbar = tqdm(total=n_trials, desc='Trials', position=0)
    epoch_pbar = tqdm(total=num_epochs, desc='Epochs', position=1, leave=False)
    return trial_pbar, epoch_pbar

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
