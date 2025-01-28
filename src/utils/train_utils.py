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

def log_separator(logger_instance: logging.Logger) -> None:
    logger_instance.info("\n" + "="*80)
