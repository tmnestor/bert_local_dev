"""Utilities for safe model loading and verification."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
import pickle
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelCheckpoint:
    """Standardized structure for model checkpoints."""

    model_state_dict: Dict[str, torch.Tensor]
    config: Dict[str, Any]
    num_classes: int
    bert_encoder_path: str
    metric_value: Optional[float] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    optimizer_state_dict: Optional[Dict[str, Any]] = None
    training_details: Optional[Dict[str, Any]] = None


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


def validate_checkpoint(checkpoint: Dict[str, Any]) -> None:
    """Validate checkpoint against schema."""
    required_keys = [
        "model_state_dict",
        "config",
        "num_classes",
        "bert_encoder_path",
    ]  # bert_encoder_path is now required
    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Missing required field: {key}")

    if not isinstance(checkpoint["model_state_dict"], dict):
        raise TypeError("model_state_dict must be a dict")
    if not isinstance(checkpoint["config"], dict):
        raise TypeError("config must be a dict")
    if not isinstance(checkpoint["num_classes"], int):
        raise TypeError("num_classes must be an int")
    if not isinstance(checkpoint["bert_encoder_path"], str):
        raise TypeError("bert_encoder_path must be a string")

    bert_path = Path(checkpoint["bert_encoder_path"])
    if not bert_path.exists():
        raise FileNotFoundError(f"BERT encoder not found at: {bert_path}")


def safe_load_checkpoint(
    path: Path, device: str = "cpu", weights_only: bool = False, strict: bool = True
) -> ModelCheckpoint:
    """Safely load model checkpoint with error handling.

    Args:
        path: Path to checkpoint file
        device: Device to load model to
        weights_only: If True, only loads weights (no optimizer etc.)
        strict: Whether to strictly enforce that the keys match

    Returns:
        ModelCheckpoint: Loaded checkpoint data

    Raises:
        FileNotFoundError: If checkpoint is not found
        RuntimeError: If loading fails
        TypeError: If checkpoint is not a dictionary
        KeyError: If checkpoint is missing required keys
    """
    try:
        # Ensure path exists
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Load with progressive fallback strategy
        try:
            # First try loading with just map_location
            checkpoint = torch.load(path, map_location=device)
        except RuntimeError as e:
            if "Weights only load failed" in str(e):
                logger.warning("Standard loading failed, attempting with pickle...")
                # Try loading with pickle directly
                with open(path, "rb") as f:
                    checkpoint = pickle.load(f)
            else:
                raise

        if not isinstance(checkpoint, dict):
            raise TypeError(f"Expected checkpoint to be dict, got {type(checkpoint)}")

        validate_checkpoint(checkpoint)

        # Create ModelCheckpoint instance
        checkpoint_data = ModelCheckpoint(
            model_state_dict=checkpoint["model_state_dict"],
            config=checkpoint["config"],
            num_classes=checkpoint["num_classes"],
            bert_encoder_path=checkpoint["bert_encoder_path"],
            metric_value=checkpoint.get("metric_value"),
            hyperparameters=checkpoint.get("hyperparameters"),
            optimizer_state_dict=checkpoint.get("optimizer_state_dict"),
            training_details=checkpoint.get("training_details"),
        )

        logger.info(f"Successfully loaded checkpoint from {path}")
        return checkpoint_data

    except FileNotFoundError as e:
        raise e
    except TypeError as e:
        raise TypeError(f"Invalid checkpoint format: {str(e)}") from e
    except KeyError as e:
        raise KeyError(f"Missing key in checkpoint: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {path}: {str(e)}") from e


def save_checkpoint(path: Path, checkpoint: ModelCheckpoint) -> None:
    """Save model checkpoint with standardized format."""
    try:
        # Validate before saving
        validate_checkpoint(checkpoint.__dict__)

        torch.save(checkpoint.__dict__, path)
        logger.info("Saved checkpoint to: %s", path)
    except Exception as e:
        raise RuntimeError(f"Failed to save checkpoint: {str(e)}") from e


def verify_state_dict(
    state_dict: Dict[str, torch.Tensor], model: torch.nn.Module
) -> None:
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


def load_model_checkpoint(
    checkpoint_path: Union[str, Path], default_root: Optional[Path] = None
) -> ModelCheckpoint:
    """Load model checkpoint with flexible path resolution.

    Args:
        checkpoint_path: Either absolute path or relative to best_trials dir
        default_root: Default output root directory containing best_trials

    Returns:
        ModelCheckpoint: Loaded checkpoint data

    Raises:
        FileNotFoundError: If checkpoint cannot be found
        RuntimeError: If checkpoint loading fails
    """
    checkpoint_path = Path(checkpoint_path)

    # Try direct path first
    if checkpoint_path.is_absolute() and checkpoint_path.exists():
        logger.debug("Loading checkpoint from absolute path: %s", checkpoint_path)
    else:
        # Try relative to best_trials directory
        if default_root:
            best_trials_dir = default_root / "best_trials"
            relative_path = best_trials_dir / checkpoint_path.name
            if relative_path.exists():
                checkpoint_path = relative_path
                logger.debug("Loading checkpoint from best_trials: %s", checkpoint_path)
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found at {checkpoint_path} or in {best_trials_dir}"
                )
        else:
            raise ValueError("If using relative path, default_root must be provided")

    try:
        checkpoint = safe_load_checkpoint(checkpoint_path)

        logger.info("Successfully loaded checkpoint: %s", checkpoint_path.name)
        if checkpoint.training_details:
            details = checkpoint.training_details
            logger.debug(
                "Model info - Study: %s, Trial: %d, Score: %.4f",
                details.get("study_name", "unknown"),
                details.get("trial_number", -1),
                details.get("score", float("nan")),
            )

        return checkpoint

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}") from e
