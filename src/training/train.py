"""Training module for BERT text classifier models.

This module handles the end-to-end training process for BERT classifiers including:
- Data loading and preprocessing
- Model initialization and configuration
- Training loop execution
- Checkpoint management
- Progress tracking and logging
- Early stopping
- Performance monitoring

The module supports both direct training and hyperparameter-optimized training,
loading configurations from either command line arguments or optimization results.

Typical usage:
    ```python
    config = ModelConfig.from_args(args)
    train_model(config)
    ```

Attributes:
    CONFIG (Dict): Default configuration loaded from config.yml
    CLASSIFIER_DEFAULTS (Dict): Default classifier architecture settings
    
Note:
    This module expects a BERT encoder to be available at the specified path
    and a properly formatted dataset with 'text' and 'category' columns.
"""

#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Optional
import torch
from transformers import get_linear_schedule_with_warmup

from ..config.configuration import (
    ModelConfig,
    CONFIG,
    CLASSIFIER_DEFAULTS,
)
from ..models.model import BERTClassifier
from .trainer import Trainer
from ..data_utils import load_and_preprocess_data, create_dataloaders
from ..utils.train_utils import initialize_progress_bars
from ..utils.logging_manager import get_logger, setup_logging
from ..tuning.optimize import create_optimizer

logger = get_logger(__name__)  # Change to get_logger


def _log_best_configuration(best_file: Path, best_value: float) -> None:
    """Logs information about the best configuration found.

    Args:
        best_file (Path): Path to the best configuration file.
        best_value (float): Best metric value achieved.

    Note:
        This is an internal helper function for logging purposes.
    """
    logger.info("Loaded best configuration from %s", best_file)
    logger.info("Best trial score: %.4f", best_value)


def load_best_configuration(
    best_trials_dir: Path, study_name: str = None
) -> Optional[dict]:
    """Loads the best model configuration from optimization results.

    Searches through optimization trial results to find the best performing
    configuration based on the evaluation metric.

    Args:
        best_trials_dir (Path): Directory containing trial results.
        study_name (str, optional): Name of the study to filter results.

    Returns:
        Optional[dict]: Best configuration dictionary if found, None otherwise.

    Raises:
        FileNotFoundError: If trial files cannot be found.
        RuntimeError: If trial files cannot be loaded.
    """
    pattern = f"best_trial_{study_name or '*'}.pt"
    trial_files = list(best_trials_dir.glob(pattern))

    if not trial_files:
        logger.warning("No previous optimization results found")
        return None

    # Find the best performing trial
    best_trial = None
    best_value = float("-inf")
    best_file = None

    for file in trial_files:
        try:
            trial_data = torch.load(file, map_location="cpu", weights_only=False)
            if trial_data["value"] > best_value:
                best_value = trial_data["value"]
                best_trial = trial_data
                best_file = file
        except (FileNotFoundError, RuntimeError, KeyError) as e:
            logger.warning("Could not load trial file %s: %s", file, e)
            continue

    if best_trial:
        _log_best_configuration(best_file, best_value)
        # Return the actual configuration
        if "params" in best_trial:
            clf_config = best_trial["params"]
            # Add any missing defaults
            return {
                "hidden_dim": clf_config.get(
                    "hidden_dim", CLASSIFIER_DEFAULTS["hidden_dims"]
                ),
                "activation": clf_config.get(
                    "activation", CLASSIFIER_DEFAULTS["activation"]
                ),
                "dropout_rate": clf_config.get(
                    "dropout_rate", CLASSIFIER_DEFAULTS["dropout_rate"]
                ),
                "lr": clf_config.get("lr", CONFIG["optimizer"]["lr"]),
                "optimizer": clf_config.get(
                    "optimizer", CONFIG["optimizer"]["optimizer_choice"]
                ),
                "weight_decay": clf_config.get(
                    "weight_decay", CONFIG["optimizer"]["weight_decay"]
                ),
                "warmup_ratio": clf_config.get(
                    "warmup_ratio", CONFIG["optimizer"]["warmup_ratio"]
                ),
            }

    logger.info("\nNo previous optimization found. Using default configuration")
    return None


def train_model(model_config: ModelConfig, clf_config: dict = None) -> None:
    """Trains a BERT classifier model with given configuration.

    This function handles the complete training process including:
    - Data loading and preprocessing
    - Model initialization
    - Optimizer and scheduler setup
    - Training loop with validation
    - Model checkpointing
    - Progress tracking and logging

    Args:
        model_config (ModelConfig): Configuration for model training parameters.
        clf_config (dict, optional): Classifier-specific configuration. If None,
            uses defaults from config.yml.

    Raises:
        ValueError: If configuration validation fails.
        RuntimeError: If training encounters an error.
    """
    if clf_config is None:
        # Use default configuration from config.yml
        clf_config = {
            "hidden_dim": CLASSIFIER_DEFAULTS["hidden_dims"],
            "activation": CLASSIFIER_DEFAULTS["activation"],
            "dropout_rate": CLASSIFIER_DEFAULTS["dropout_rate"],
            "optimizer": CONFIG["optimizer"]["optimizer_choice"],
            "lr": CONFIG["optimizer"]["lr"],
            "weight_decay": CONFIG["optimizer"]["weight_decay"],
            "warmup_ratio": CONFIG["optimizer"]["warmup_ratio"],
        }

    # Ensure bert_model_name is absolute
    if not Path(model_config.bert_model_name).is_absolute():
        model_config.bert_model_name = str(model_config.output_root / "bert_encoder")

    logger.info("Using BERT encoder from: %s", model_config.bert_model_name)

    # Single optimizer configuration section
    optimizer_name = clf_config.get("optimizer", "adamw")
    optimizer_params = {
        "lr": clf_config.get("lr", model_config.learning_rate),
        "weight_decay": clf_config.get("weight_decay", 0.01),
    }

    # Add optimizer-specific parameters
    if optimizer_name == "rmsprop":
        optimizer_params.update(
            {
                "momentum": clf_config.get("momentum", CONFIG["optimizer"]["momentum"]),
                "alpha": clf_config.get("alpha", CONFIG["optimizer"]["alpha"]),
            }
        )
    elif optimizer_name == "sgd":
        optimizer_params.update(
            {
                "momentum": clf_config.get("momentum", CONFIG["optimizer"]["momentum"]),
                "nesterov": clf_config.get("nesterov", CONFIG["optimizer"]["nesterov"]),
            }
        )
    elif optimizer_name == "adamw":
        # Ensure betas is a tuple of floats
        betas = CONFIG["optimizer"]["betas"]
        if isinstance(betas, (list, tuple)):
            betas = tuple(float(x) for x in betas)
        optimizer_params.update(
            {
                "betas": betas,
                "eps": float(clf_config.get("eps", CONFIG["optimizer"]["eps"])),
            }
        )

    # Update classifier config with optimizer settings
    clf_config["optimizer_config"] = optimizer_params

    # Load data with DataBundle return type
    data = load_and_preprocess_data(model_config)

    # Set num_classes based on label encoder if not already set
    if model_config.num_classes is None:
        model_config.num_classes = len(data.label_encoder.classes_)

    logger.info(
        "Loaded %d training and %d validation samples (%d classes)",
        len(data.train_texts),
        len(data.val_texts),
        model_config.num_classes,
    )

    # Create dataloaders using utility function with DataBundle attributes
    train_dataloader, val_dataloader = create_dataloaders(
        [data.train_texts, data.val_texts],
        [data.train_labels, data.val_labels],
        model_config,
        model_config.batch_size,
    )

    # Initialize model and trainer
    model = BERTClassifier(
        model_config.bert_model_name, model_config.num_classes, clf_config
    )
    trainer = Trainer(model, model_config)

    optimizer = create_optimizer(optimizer_name, model.parameters(), **optimizer_params)

    total_steps = len(train_dataloader) * model_config.num_epochs
    warmup_steps = int(
        total_steps * clf_config["warmup_ratio"]
    )  # Get from clf_config instead of model_config
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Initialize progress bars using utility function
    epoch_pbar, batch_pbar = initialize_progress_bars(
        model_config.num_epochs,
        len(train_dataloader),  # Pass batch count for batch progress bar
    )

    # Training loop
    best_score = 0.0

    try:
        for epoch in range(model_config.num_epochs):
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{model_config.num_epochs}")

            loss = trainer.train_epoch(
                train_dataloader, optimizer, scheduler, batch_pbar
            )
            score, metrics = trainer.evaluate(val_dataloader)

            epoch_pbar.set_postfix(
                {"loss": f"{loss:.4f}", f"val_{model_config.metric}": f"{score:.4f}"}
            )
            epoch_pbar.update(1)

            if score > best_score:
                best_score = score
                # Create state dictionary with simplified structure
                best_state = {
                    "model_state": model.state_dict(),
                    "config": clf_config.copy(),
                    f"{model_config.metric}_score": score,
                    "trial_number": None,
                    "params": clf_config.copy(),
                    "epoch": epoch,
                    "metrics": metrics,
                }

                # Save using simplified format
                save_dict = {
                    "model_state_dict": best_state["model_state"],
                    "config": clf_config,  # Direct config without nesting
                    "metric_value": score,
                    "study_name": "training",
                    "trial_number": None,
                    "num_classes": model_config.num_classes,
                    "hyperparameters": clf_config,
                    "val_size": 0.2,
                    "metric": model_config.metric,
                }
                torch.save(save_dict, model_config.model_save_path)

                # Add verbosity-specific logging
                if model_config.verbosity >= 1:
                    logger.info(
                        "\nSaved better model with %s=%.4f to: %s",
                        model_config.metric,
                        score,
                        model_config.model_save_path,
                    )

    finally:
        epoch_pbar.close()
        if batch_pbar is not None:  # Only close batch_pbar if it exists
            batch_pbar.close()

        # Add final training status with path
        if model_config.verbosity >= 1:
            logger.info(
                "\nTraining completed. Best %s: %.4f", model_config.metric, best_score
            )
            logger.info("Final model saved to: %s", model_config.model_save_path)

    logger.info("\nTraining completed. Best %s: %.4f", model_config.metric, best_score)
    if model_config.verbosity > 1:  # Add detailed path logging for debug level
        logger.info("Model saved to: %s", model_config.model_save_path.absolute())
        logger.info("Model file exists: %s", model_config.model_save_path.exists())
        logger.info(
            "Model file size: %.2f MB",
            model_config.model_save_path.stat().st_size / (1024 * 1024),
        )


def parse_args() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train BERT Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Essential paths and settings
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(CONFIG["output_root"]),  # Get from CONFIG
        help="Root directory for all operations",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=CONFIG.get("logging", {}).get("verbosity", 1),  # Get from CONFIG
        choices=[0, 1, 2],
        help="Verbosity level",
    )

    # Training parameters
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--num_epochs",
        type=int,
        default=CONFIG["model"]["num_epochs"],  # Get from CONFIG
        help="Number of training epochs",
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=CONFIG["model"]["batch_size"],  # Get from CONFIG
        help="Training batch size",
    )
    train_group.add_argument(
        "--max_seq_len",
        type=int,
        default=CONFIG["model"]["max_seq_len"],  # Get from CONFIG
        help="Maximum sequence length for tokenization",
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=CONFIG["optimizer"]["lr"],  # Get from CONFIG
        help="Learning rate",
    )
    train_group.add_argument(
        "--device",
        type=str,
        default=CONFIG["model"]["device"],  # Get from CONFIG
        choices=["cpu", "cuda"],
        help="Device for training",
    )

    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    config = ModelConfig.from_args(cli_args)
    setup_logging(config)

    # Always use configuration from config.yml
    train_model(config)
