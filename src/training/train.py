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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..config.configuration import (
    ModelConfig,
    CONFIG,
    CLASSIFIER_DEFAULTS,
    load_yaml_config,  # Add this import
)
from ..models.model import BERTClassifier
from .trainer import Trainer
from ..data_utils import load_and_preprocess_data, create_dataloaders
from ..utils.train_utils import initialize_progress_bars
from ..utils.logging_manager import get_logger, setup_logging
from ..tuning.optimize import create_optimizer
from ..utils.model_loading import (
    save_checkpoint,
    ModelCheckpoint,
)  # Import ModelCheckpoint

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


def save_best_model(
    model: BERTClassifier, config: ModelConfig, score: float, path: Path
) -> None:
    """Save best model checkpoint."""

    checkpoint = ModelCheckpoint(
        model_state_dict=model.state_dict(),
        config={
            "bert_hidden_size": CLASSIFIER_DEFAULTS["bert_hidden_size"],
            "hidden_dims": CLASSIFIER_DEFAULTS["hidden_dims"],
            "dropout_rate": CLASSIFIER_DEFAULTS["dropout_rate"],
            "activation": CLASSIFIER_DEFAULTS["activation"],
        },
        num_classes=config.num_classes,
        bert_encoder_path=str(config.bert_encoder_path),
        metric_value=score,
        hyperparameters={
            "optimizer": CONFIG["optimizer"]["optimizer_choice"],
            "lr": CONFIG["optimizer"]["lr"],
            "weight_decay": CONFIG["optimizer"]["weight_decay"],
        },
    )

    try:
        save_checkpoint(path=path, checkpoint=checkpoint)
        logger.info("Saved best model to: %s", path)
    except Exception as e:
        logger.error("Failed to save model: %s", str(e))
        raise


def save_model(model, config, optimizer, metrics, epoch, trial_number=None):
    """Save model checkpoint with complete configuration."""
    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": {
            "hidden_dim": model.classifier_config["hidden_dim"],
            "activation": model.classifier_config["activation"],
            "dropout_rate": model.classifier_config["dropout_rate"],
            "optimizer": optimizer.__class__.__name__.lower(),  # Get actual optimizer type
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0]["weight_decay"],
            "warmup_ratio": config.optimizer.get("warmup_ratio", 0.0),
        },
        # Add complete training details
        "training_details": {
            "study_name": "training",  # or config.study_name if available
            "trial_number": trial_number or 0,
            "score": metrics.get(config.metric, 0.0),
            "total_epochs": config.num_epochs,
            "completed_epochs": epoch + 1,
            "early_stopping": epoch + 1 < config.num_epochs,
            "architecture": {
                "hidden_layers": model.classifier_config["hidden_dim"],
                "activation": model.classifier_config["activation"],
                "dropout_rate": model.classifier_config["dropout_rate"],
            },
            "optimizer": {
                "type": optimizer.__class__.__name__.lower(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "weight_decay": optimizer.param_groups[0]["weight_decay"],
                "warmup_ratio": config.optimizer.get("warmup_ratio", 0.0),
                # Add optimizer-specific parameters
                "betas": getattr(optimizer, "betas", None),  # AdamW
                "eps": getattr(optimizer, "eps", None),  # AdamW
                "momentum": getattr(optimizer, "momentum", None),  # SGD/RMSprop
                "alpha": getattr(optimizer, "alpha", None),  # RMSprop
                "nesterov": getattr(optimizer, "nesterov", None),  # SGD
            },
        },
        "metrics": metrics,
    }

    # Save to file
    save_path = config.model_save_path
    torch.save(save_dict, save_path)
    logger.info(f"Model saved to {save_path}")


def train_model(model_config: ModelConfig, clf_config: dict = None) -> None:
    """Trains a BERT classifier model with given configuration."""
    # Load fresh config
    config_dict = load_yaml_config()
    opt_config = config_dict["optimizer"]  # No need for .get() since we know it exists

    if clf_config is None:
        clf_config = {
            "hidden_dim": config_dict["classifier"]["hidden_dims"],
            "activation": config_dict["classifier"]["activation"],
            "dropout_rate": config_dict["classifier"]["dropout_rate"],
            "bert_hidden_size": config_dict["classifier"]["bert_hidden_size"],
            "optimizer": opt_config["optimizer_choice"],
            "lr": opt_config["lr"],
            "weight_decay": opt_config["weight_decay"],
            "warmup_ratio": opt_config[
                "warmup_ratio"
            ],  # Directly use value from config
        }

    # Get optimizer params from config.yml
    optimizer_name = config_dict["optimizer"]["optimizer_choice"]
    optimizer_params = {
        "lr": config_dict["optimizer"]["lr"],
        "weight_decay": config_dict["optimizer"]["weight_decay"],
    }

    # Get optimizer-specific params directly from config.yml
    if optimizer_name == "rmsprop":
        optimizer_params.update(
            {
                "momentum": config_dict["optimizer"]["momentum"],
                "alpha": config_dict["optimizer"]["alpha"],
            }
        )
    elif optimizer_name == "sgd":
        optimizer_params.update(
            {
                "momentum": config_dict["optimizer"]["momentum"],
                "nesterov": config_dict["optimizer"]["nesterov"],
            }
        )
    elif optimizer_name == "adamw":
        betas = config_dict["optimizer"]["betas"]
        if isinstance(betas, str):
            betas = tuple(float(x.strip()) for x in betas.strip("()").split(","))
        optimizer_params.update(
            {
                "betas": betas,
                "eps": float(config_dict["optimizer"]["eps"]),
            }
        )

    # Add optimizer config to classifier config
    clf_config["optimizer_config"] = optimizer_params

    logger.info("Using BERT encoder from: %s", model_config.bert_encoder_path)

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

    # Always use local BERT encoder for training
    bert_encoder_path = model_config.output_root / "bert_encoder"
    if not bert_encoder_path.exists():
        raise ValueError(f"BERT encoder not found at {bert_encoder_path}")

    logger.info("Using local BERT encoder from: %s", bert_encoder_path)

    # Initialize model for training
    model = BERTClassifier(
        bert_encoder_path=str(model_config.bert_encoder_path),
        num_classes=model_config.num_classes,
        classifier_config=clf_config,
    )

    trainer = Trainer(model, model_config)

    logger.info(
        "Creating %s optimizer with params: %s", optimizer_name, optimizer_params
    )
    optimizer = create_optimizer(optimizer_name, model.parameters(), **optimizer_params)

    total_steps = len(train_dataloader) * model_config.num_epochs
    warmup_steps = int(total_steps * opt_config["warmup_ratio"])

    # Create linear warmup scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Initialize progress bars using utility function
    epoch_pbar, batch_pbar = initialize_progress_bars(
        model_config.num_epochs,
        len(train_dataloader),  # Pass batch count for batch progress bar
    )

    # Initialize lists to store training and validation metrics
    training_metrics = []
    validation_metrics = []

    # Training loop
    best_score = 0.0

    try:
        for epoch in range(model_config.num_epochs):
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{model_config.num_epochs}")

            loss = trainer.train_epoch(
                train_dataloader, optimizer, scheduler, batch_pbar
            )
            score, metrics = trainer.evaluate(val_dataloader)

            # Calculate training F1 score
            train_score, train_metrics = trainer.evaluate(train_dataloader)

            # Calculate validation loss
            val_loss = trainer.train_epoch(val_dataloader, optimizer, scheduler)

            epoch_pbar.set_postfix(
                {"loss": f"{loss:.4f}", f"val_{model_config.metric}": f"{score:.4f}"}
            )
            epoch_pbar.update(1)

            # Store training and validation metrics
            training_metrics.append(
                {
                    "epoch": epoch + 1,
                    "metric": "loss",
                    "value": loss,
                    "dataset": "training",
                }
            )
            training_metrics.append(
                {
                    "epoch": epoch + 1,
                    "metric": model_config.metric,
                    "value": train_score,
                    "dataset": "training",
                }
            )
            validation_metrics.append(
                {
                    "epoch": epoch + 1,
                    "metric": model_config.metric,
                    "value": score,
                    "dataset": "validation",
                }
            )
            validation_metrics.append(
                {
                    "epoch": epoch + 1,
                    "metric": "loss",
                    "value": val_loss,
                    "dataset": "validation",
                }
            )

            if score > best_score:
                best_score = score
                save_best_model(
                    model, model_config, score, model_config.model_save_path
                )

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

    # Create learning curves plot
    plot_learning_curves(
        training_metrics, validation_metrics, model_config.evaluation_dir
    )

    logger.info("\nTraining completed. Best %s: %.4f", model_config.metric, best_score)
    if model_config.verbosity > 1:  # Add detailed path logging for debug level
        logger.info("Model saved to: %s", model_config.model_save_path.absolute())
        logger.info("Model file exists: %s", model_config.model_save_path.exists())
        logger.info(
            "Model file size: %.2f MB",
            model_config.model_save_path.stat().st_size / (1024 * 1024),
        )


def plot_learning_curves(training_metrics, validation_metrics, output_dir):
    """Plot learning curves using Seaborn."""
    # Combine training and validation metrics into a single DataFrame
    all_metrics = training_metrics + validation_metrics
    metrics_df = pd.DataFrame(all_metrics)

    # Use Seaborn to create the learning curves plot
    plt.figure(figsize=(12, 6))
    sns.set(style="darkgrid")
    sns.lineplot(
        data=metrics_df,
        x="epoch",
        y="value",
        hue="metric",
        style="dataset",
        markers=True,
        dashes=False,
    )

    # Customize the plot
    plt.title("Learning Curves", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.legend(title="Legend", loc="best")

    # Set x-ticks to integer values
    plt.xticks(range(1, len(set(metrics_df["epoch"])) + 1))

    plt.tight_layout()

    # Save the plot
    plot_path = output_dir / "learning_curves.png"
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Learning curves plot saved to: {plot_path}")


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

    # Set default tensor type to MPS if available
    if config.device == "mps" and torch.backends.mps.is_available():
        torch.set_default_tensor_type(torch.FloatTensor)

    # Always use configuration from config.yml
    train_model(config)
