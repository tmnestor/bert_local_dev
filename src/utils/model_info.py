"""Utilities for displaying model information."""

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from ..utils.logging_manager import get_logger

logger = get_logger(__name__)


def get_optimizer_info(opt_config: dict) -> dict:
    """Get optimizer-specific parameters."""
    opt_type = opt_config.get("type", opt_config.get("optimizer", "adamw"))

    # Base optimizer info
    info = {
        "type": opt_type,
        "learning_rate": opt_config.get("lr", opt_config.get("learning_rate")),
        "weight_decay": opt_config.get("weight_decay"),
        "warmup_ratio": opt_config.get("warmup_ratio"),
    }

    # Add optimizer-specific parameters
    if opt_type == "adamw":
        info.update(
            {
                "betas": opt_config.get("betas", "(0.9, 0.999)"),
                "eps": opt_config.get("eps", 1e-8),
            }
        )
    elif opt_type == "rmsprop":
        info.update(
            {
                "momentum": opt_config.get("momentum", 0.0),
                "alpha": opt_config.get("alpha", 0.99),
            }
        )
    elif opt_type == "sgd":
        info.update(
            {
                "momentum": opt_config.get("momentum", 0.0),
                "nesterov": opt_config.get("nesterov", False),
            }
        )

    return info


def display_model_info(checkpoint_path: Path) -> None:
    """Display model architecture and training details."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Get training details if they exist, otherwise use config
        if "training_details" in checkpoint:
            details = checkpoint["training_details"]
            arch = details["architecture"]
            opt = details["optimizer"]

            # Get complete optimizer info
            opt_info = get_optimizer_info(opt)

            separator = "=" * 80
            print(f"\nModel Configuration")
            print(separator)
            print(f"Study: {details['study_name']}")
            print(f"Trial: {details['trial_number']} (score: {details['score']:.4f})")
            print(
                f"Epochs: {details['completed_epochs']}/{details['total_epochs']}"
                + (" (early stopping)" if details["early_stopping"] else "")
            )
        else:
            # Fallback to config for older checkpoints
            config = checkpoint["config"]
            separator = "=" * 80
            print(f"\nModel Configuration")
            print(separator)
            print("Using checkpoint:", checkpoint_path.name)
            arch = {
                "hidden_layers": config.get("hidden_dim"),
                "activation": config.get("activation"),
                "dropout_rate": config.get("dropout_rate"),
            }
            # Get complete optimizer info from config
            opt_info = get_optimizer_info(config)

        print("\nModel Architecture:")
        print(f"  Hidden layers: {arch['hidden_layers']}")
        print(f"  Activation: {arch['activation']}")
        print(f"  Dropout rate: {arch['dropout_rate']:.4f}")
        print("\nOptimizer Settings:")
        print(f"  Type: {opt_info['type']}")
        print(f"  Learning rate: {opt_info['learning_rate']:.6f}")
        print(f"  Weight decay: {opt_info['weight_decay']:.6f}")
        print(f"  Warmup ratio: {opt_info['warmup_ratio']:.4f}")

        # Print optimizer-specific parameters
        if opt_info["type"] == "adamw":
            print(f"  Betas: {opt_info['betas']}")
            print(f"  Epsilon: {opt_info['eps']:.2e}")
        elif opt_info["type"] == "rmsprop":
            print(f"  Momentum: {opt_info['momentum']:.4f}")
            print(f"  Alpha: {opt_info['alpha']:.4f}")
        elif opt_info["type"] == "sgd":
            print(f"  Momentum: {opt_info['momentum']:.4f}")
            print(f"  Nesterov: {opt_info['nesterov']}")
        print(separator)

    except Exception as e:
        logger.error(f"Failed to load model info: {str(e)}")


def list_model_files(trials_dir: Path) -> None:
    """List all model files in the trials directory."""
    model_files = list(trials_dir.glob("best_*.pt"))
    if not model_files:
        print(f"No model files found in {trials_dir}")
        return

    print(f"\nFound {len(model_files)} model file(s) in {trials_dir}:")
    print("=" * 80)

    for file in sorted(model_files):
        try:
            checkpoint = torch.load(file, map_location="cpu")
            if "training_details" in checkpoint:
                details = checkpoint["training_details"]
                print(f"\nFile: {file.name}")
                print(f"Study: {details['study_name']}")
                print(
                    f"Trial: {details['trial_number']} (score: {details['score']:.4f})"
                )
                print(
                    f"Epochs: {details['completed_epochs']}/{details['total_epochs']}"
                    + (" (early stopping)" if details["early_stopping"] else "")
                )
            else:
                # Fallback for older checkpoints
                print(f"\nFile: {file.name}")
                print("(Older checkpoint format - detailed info not available)")
            print("-" * 80)
        except Exception as e:
            print(f"Error loading {file.name}: {str(e)}")


def main():
    """Command-line interface for model inspection."""
    parser = argparse.ArgumentParser(description="Inspect BERT classifier checkpoints")
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Root directory containing best_trials",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model file to inspect (relative to best_trials dir)",
    )

    args = parser.parse_args()
    trials_dir = args.output_root / "best_trials"

    if not trials_dir.exists():
        print(f"Error: best_trials directory not found at {trials_dir}")
        return

    if args.model:
        model_path = trials_dir / args.model
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}")
            return
        display_model_info(model_path)
    else:
        # List all models if no specific model specified
        list_model_files(trials_dir)


if __name__ == "__main__":
    main()
