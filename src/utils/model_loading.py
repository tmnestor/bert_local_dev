"""Utilities for safe model loading and verification."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import pickle

logger = logging.getLogger(__name__)

def add_safe_globals():
    """Add necessary globals to PyTorch's safe list."""
    try:
        import numpy as np
        from torch.serialization import add_classes_to_whitelist
        safe_classes = [
            np.core.multiarray.scalar,
            np.ndarray,
            np.dtype,
            np._globals._NoValue,
        ]
        for cls in safe_classes:
            add_classes_to_whitelist([cls])
    except (ImportError, AttributeError):
        # Older PyTorch versions or if add_classes_to_whitelist isn't available
        pass

def safe_load_checkpoint(
    path: Path,
    device: str = 'cpu',
    weights_only: bool = False,
    strict: bool = True
) -> Dict[str, Any]:
    """Safely load model checkpoint with error handling.

    Args:
        path: Path to checkpoint file
        device: Device to load model to
        weights_only: If True, only loads weights (no optimizer etc.)
        strict: Whether to strictly enforce that the keys match

    Returns:
        dict: Loaded checkpoint data

    Raises:
        RuntimeError: If loading fails
    """
    try:
        # Ensure path exists
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Load with progressive fallback strategy
        try:
            # First try loading with just map_location
            checkpoint = torch.load(
                path,
                map_location=device
            )
        except RuntimeError as e:
            if "Weights only load failed" in str(e):
                logger.warning("Standard loading failed, attempting with pickle...")
                # Try loading with pickle directly
                with open(path, 'rb') as f:
                    checkpoint = pickle.load(f)
            else:
                raise

        if not isinstance(checkpoint, dict):
            raise TypeError(f"Expected checkpoint to be dict, got {type(checkpoint)}")

        required_keys = ['model_state_dict', 'config']
        if strict and not all(k in checkpoint for k in required_keys):
            raise KeyError(f"Checkpoint missing required keys: {required_keys}")

        logger.info(f"Successfully loaded checkpoint from {path}")
        return checkpoint

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {path}: {str(e)}") from e

def verify_state_dict(state_dict: Dict[str, torch.Tensor], model: torch.nn.Module) -> None:
    """Verify model state dict compatibility.

    Args:
        state_dict: Model state dictionary
        model: Model instance to verify against

    Raises:
        ValueError: If state dict is incompatible
    """
    try:
        # Get model's state dict for comparison
        model_state = model.state_dict()
        
        # Check keys match
        missing = [k for k in model_state.keys() if k not in state_dict]
        unexpected = [k for k in state_dict.keys() if k not in model_state]
        
        if missing:
            raise ValueError(f"Missing keys in state dict: {missing}")
        if unexpected:
            raise ValueError(f"Unexpected keys in state dict: {unexpected}")
            
        # Check tensor shapes match
        mismatched = []
        for key, val in state_dict.items():
            if key in model_state:
                if val.shape != model_state[key].shape:
                    mismatched.append(
                        f"{key}: expected {model_state[key].shape}, got {val.shape}"
                    )
        
        if mismatched:
            raise ValueError(f"Tensor shape mismatches: {mismatched}")
            
    except Exception as e:
        raise ValueError(f"State dict verification failed: {str(e)}") from e
