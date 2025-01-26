"""Utilities for safe model loading and verification."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

def safe_load_checkpoint(
    checkpoint_path: Path,
    device: Union[str, torch.device],
    expected_keys: Optional[set] = None,
    strict: bool = True,
    weights_only: bool = True
) -> Dict[str, Any]:
    """Safely load a model checkpoint with validation.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load the model onto
        expected_keys: Set of required keys in checkpoint
        strict: Whether to enforce expected keys
        weights_only: Whether to load only weights (safer)
        
    Returns:
        Dictionary containing checkpoint data
        
    Raises:
        ValueError: If checkpoint validation fails
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    def _try_load_with_weights_only():
        # Handle numpy types first
        torch.serialization.add_classes_to_whitelist([
            ('numpy', 'dtype'),
            ('numpy', 'ndarray'),
            ('numpy.core.multiarray', 'scalar'),
            ('numpy', 'bool_'),
            ('numpy', 'float32'),
            ('numpy', 'float64'),
            ('numpy', 'int64'),
            ('numpy', 'int32')
        ])
        # Add numpy functions
        torch.serialization.add_classes_to_whitelist([
            ('numpy.core.multiarray', '_reconstruct')
        ])
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
            mmap=True
        )

    def _try_load_without_weights_only():
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

    try:
        # First attempt: with weights_only=True
        if weights_only:
            try:
                checkpoint = _try_load_with_weights_only()
                logger.info("Successfully loaded checkpoint with weights_only=True")
            except Exception as e:
                logger.warning(f"Failed to load with weights_only=True: {e}")
                logger.info("Falling back to weights_only=False...")
                checkpoint = _try_load_without_weights_only()
                logger.info("Successfully loaded checkpoint with weights_only=False")
        else:
            checkpoint = _try_load_without_weights_only()
            
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise ValueError(f"Checkpoint loading failed: {e}")

    # Validate checkpoint structure
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")

    # Check required keys
    if strict and expected_keys:
        missing = expected_keys - checkpoint.keys()
        if missing:
            raise ValueError(f"Checkpoint missing required keys: {missing}")

    # Log success
    logger.info("\nCheckpoint loaded successfully:")
    logger.info("  File: %s", checkpoint_path)
    logger.info("  Size: %.2f MB", checkpoint_path.stat().st_size / (1024 * 1024))
    logger.info("  Keys: %s", list(checkpoint.keys()))

    return checkpoint

def verify_state_dict(
    state_dict: Dict[str, torch.Tensor],
    model: nn.Module,
    partial: bool = False
) -> bool:
    """Verify that a state dict is compatible with a model.
    
    Args:
        state_dict: Model state dictionary
        model: Target model instance
        partial: Allow partial matches
        
    Returns:
        bool: Whether verification passed
        
    Raises:
        ValueError: If verification fails
    """
    try:
        model_state = model.state_dict()
        
        # Check for missing or unexpected keys
        missing = model_state.keys() - state_dict.keys()
        unexpected = state_dict.keys() - model_state.keys()
        
        if not partial and (missing or unexpected):
            raise ValueError(
                f"State dict mismatch:\n"
                f"Missing keys: {missing}\n"
                f"Unexpected keys: {unexpected}"
            )
            
        # Verify tensor shapes match
        for key in state_dict:
            if key in model_state:
                if state_dict[key].shape != model_state[key].shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: "
                        f"expected {model_state[key].shape}, "
                        f"got {state_dict[key].shape}"
                    )
                    
        return True
        
    except Exception as e:
        logger.error(f"State dict verification failed: {e}")
        raise ValueError(f"Invalid state dict: {e}") from e
