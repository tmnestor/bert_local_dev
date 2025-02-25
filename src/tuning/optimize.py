#!/usr/bin/env python

"""Hyperparameter optimization for BERT classifier using Optuna and Population Based Training.

This module handles hyperparameter optimization using Optuna with PBT integration.
It provides functionality for:
- Setting up optimization trials with Optuna
- Managing hyperparameter population with PBT
- Dynamic batch size adjustment based on memory usage
- Early stopping with multiple criteria
- Trial tracking and logging
- Model state saving and loading
- Progress visualization

The main optimization loop uses a combination of Optuna's TPE sampler and
Population Based Training to efficiently explore the hyperparameter space while
allowing for dynamic adaptation during training.

Typical usage:
    ```python
    config = ModelConfig.from_args(args)
    best_params = run_optimization(
        config,
        experiment_name="bert_opt",
        n_trials=100,
    )
    ```

Note:
    This module requires Optuna, PyTorch, and the custom BERT classifier implementation.
"""

import argparse
import random
import time
import warnings
from collections import deque
from functools import partial
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

import logging  # Add this import
import numpy as np
import psutil
import optuna
import torch
from optuna._experimental import ExperimentalWarning
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, QMCSampler
from sklearn.model_selection import train_test_split
from torch import optim
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from ..config.configuration import (
    ModelConfig,
    CONFIG,
    CLASSIFIER_DEFAULTS,
)  # Change imports to use configuration
from ..data_utils import create_dataloaders, load_and_preprocess_data
from ..models.model import BERTClassifier
from ..training.trainer import Trainer
from ..utils.logging_manager import (
    get_logger,
    setup_logging,
)  # Change from setup_logger
from ..utils.train_utils import log_separator
from ..utils.model_loading import (
    save_checkpoint,
    ModelCheckpoint,
)  # Import ModelCheckpoint

# Silence specific Optuna warnings
warnings.filterwarnings("ignore", category=ExperimentalWarning)

logger = get_logger(__name__)  # Change to get_logger


class PBTManager:
    """Population Based Training manager for hyperparameter optimization.

    This class implements the Population Based Training algorithm, managing a population
    of trials and their hyperparameters. It handles exploration and exploitation of the
    parameter space based on trial performance.

    Attributes:
        population_size (int): Size of the population to maintain.
        exploit_threshold (float): Threshold for determining when to exploit better solutions.
        population (List[Dict]): List of dictionaries containing trial information.
        generation (int): Current generation number.
    """

    def __init__(self, population_size: int = 4, exploit_threshold: float = 0.2):
        self.population_size = population_size
        self.exploit_threshold = exploit_threshold
        self.population: List[Dict] = []
        self.generation = 0

    def should_explore(self, score: float) -> bool:
        """Determine if a trial should explore new hyperparameters."""
        if len(self.population) < self.population_size:
            return False

        # Sort population by score
        sorted_pop = sorted(self.population, key=lambda x: x["score"], reverse=True)

        # Find position of current score
        for idx, member in enumerate(sorted_pop):
            if score <= member["score"]:
                # Explore if in bottom percentile
                return idx >= len(sorted_pop) * (1 - self.exploit_threshold)
        return True

    def explore(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate new parameters by perturbing current ones."""
        new_params = current_params.copy()

        # Perturb continuous parameters
        for key in ["lr", "weight_decay", "dropout_rate", "warmup_ratio"]:
            if key in new_params:
                # Perturb by random factor between 0.8 and 1.2
                new_params[key] *= random.uniform(0.8, 1.2)

        return new_params

    def exploit(
        self, score: float, current_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Copy parameters from better performing trials.

        Args:
            score: Current trial's score
            current_params: Current trial's parameters to potentially replace

        Returns:
            Optional[Dict[str, Any]]: Parameters from a better performing trial or None
        """
        if len(self.population) < 2:
            return None

        # Find better performing trials
        better_trials = [
            p
            for p in self.population
            if p["score"] > score and p["params"] != current_params
        ]  # Avoid self-copy
        if not better_trials:
            return None

        # Copy from a random better trial
        chosen = random.choice(better_trials)
        return chosen["params"].copy()

    def update_population(self, score: float, params: Dict[str, Any], trial_num: int):
        """Update population with new trial results."""
        self.population.append({"score": score, "params": params, "trial": trial_num})

        # Keep only the best population_size members
        if len(self.population) > self.population_size:
            self.population.sort(key=lambda x: x["score"], reverse=True)
            self.population = self.population[: self.population_size]


def create_optimizer(
    optimizer_name: str, model_params: Iterator[torch.nn.Parameter], **kwargs
) -> torch.optim.Optimizer:
    """Creates a PyTorch optimizer with proper parameter mapping.

    Args:
        optimizer_name (str): Name of the optimizer to create (adam, adamw, sgd, rmsprop).
        model_params (Iterator[torch.nn.Parameter]): Model parameters to optimize.
        **kwargs: Optimizer-specific configuration parameters.

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.

    Raises:
        ValueError: If optimizer_name is not one of the supported optimizers.
    """

    optimizers = {
        "adam": {"class": optim.Adam, "params": {}},
        "adamw": {"class": optim.AdamW, "params": {}},
        "sgd": {"class": optim.SGD, "params": {}},
        "rmsprop": {"class": optim.RMSprop, "params": {}},
    }

    if optimizer_name.lower() not in optimizers:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Available optimizers: {list(optimizers.keys())}"
        )

    optimizer_info = optimizers[optimizer_name.lower()]
    optimizer_class = optimizer_info["class"]
    optimizer_params = optimizer_info["params"].copy()

    # Add optimizer-specific parameters from kwargs
    optimizer_params.update(kwargs)

    return optimizer_class(model_params, **optimizer_params)


def get_optimizer_config(trial, optimizer_name, lr):
    """Get optimizer-specific configuration."""
    optimizer_config = {
        "lr": lr,
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
    }

    if optimizer_name == "sgd":
        optimizer_config.update(
            {
                "momentum": trial.suggest_float("momentum", 0.0, 0.99),
                "nesterov": trial.suggest_categorical("nesterov", [True, False]),
            }
        )
    elif optimizer_name == "adamw":
        optimizer_config.update(
            {
                "betas": (
                    trial.suggest_float("beta1", 0.5, 0.9999),
                    trial.suggest_float("beta2", 0.9, 0.9999),
                ),
                "eps": trial.suggest_float("eps", 1e-8, 1e-6, log=True),
            }
        )
    elif optimizer_name == "rmsprop":
        optimizer_config.update(
            {
                "momentum": trial.suggest_float("momentum", 0.0, 0.99),
                "alpha": trial.suggest_float("alpha", 0.8, 0.99),
            }
        )

    return optimizer_config


def _create_study(
    name: str,
    storage: Optional[str] = None,
    sampler_config: Dict[str, Any] = None,
    pruner_config: Dict[str, Any] = None,
    random_seed: Optional[int] = None,
) -> optuna.Study:
    """Create an Optuna study for hyperparameter optimization."""
    # Silence optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if random_seed is None:
        random_seed = int(time.time())

    # Sampler configuration
    sampler_type = sampler_config.get("name", "tpe") if sampler_config else "tpe"
    sampler_kwargs = sampler_config.get("kwargs", {}) if sampler_config else {}
    sampler_kwargs["seed"] = random_seed

    # Remove n_startup_trials as it's not valid for RandomSampler
    sampler_kwargs.pop("n_startup_trials", None)
    # Remove n_ei_candidates as it's not valid for RandomSampler
    sampler_kwargs.pop("n_ei_candidates", None)

    sampler = {
        "tpe": TPESampler(**sampler_kwargs),
        "random": RandomSampler(**sampler_kwargs),
        "cmaes": CmaEsSampler(**sampler_kwargs),
        "qmc": QMCSampler(qmc_type="sobol", **sampler_kwargs),
    }.get(sampler_type, TPESampler(**sampler_kwargs))

    # Pruner configuration
    pruner_type = (
        pruner_config.get("name", "hyperband") if pruner_config else "hyperband"
    )
    pruner_kwargs = pruner_config.get("kwargs", {}) if pruner_config else {}

    pruner = {
        "hyperband": HyperbandPruner(**pruner_kwargs),
        "median": MedianPruner(),  # Remove pruner_kwargs
        "none": NopPruner(),
    }.get(pruner_type, HyperbandPruner(**pruner_kwargs))

    return optuna.create_study(
        study_name=name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def run_optimization(
    model_config: ModelConfig,
    timeout: Optional[int] = None,
    experiment_name: str = "bert_optimization",
    storage: Optional[str] = None,
    random_seed: Optional[int] = None,
    n_trials: Optional[int] = None,
) -> Dict[str, Any]:
    """Run hyperparameter optimization for the BERT classifier."""
    # BERT encoder path is now handled by ModelConfig's __post_init__
    # Just log the path we're using
    logger.info("Using BERT encoder from: %s", model_config.bert_encoder_path)

    # Only show detailed info if verbosity > 0
    if model_config.verbosity > 0:
        log_separator(logger)
        logger.info("Starting optimization")
        logger.info("Number of trials: %s", n_trials or model_config.n_trials)
        logger.info("\nLoading data...")

    # Load data with new DataBundle return type
    data = load_and_preprocess_data(model_config)

    if model_config.num_classes is None:
        model_config.num_classes = len(data.label_encoder.classes_)

    # Combine texts and labels
    texts = []
    texts.extend(data.train_texts)
    texts.extend(data.val_texts)

    labels = []
    labels.extend(data.train_labels)
    labels.extend(data.val_labels)

    logger.info(
        "Loaded %d total samples (%d classes)", len(texts), model_config.num_classes
    )

    # Initialize progress bar
    trial_pbar = tqdm(
        total=n_trials or model_config.n_trials,
        desc="Trials",
        position=0,
        leave=True,
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
    )

    # Get pruner and sampler settings from config
    pruner_config = CONFIG.get("pruner", {})
    sampler_config = CONFIG.get("sampler", {})

    # Create study and run optimization
    study = _create_study(
        experiment_name, storage, sampler_config, pruner_config, random_seed
    )
    study.set_user_attr("best_value", 0.0)
    global_best_info = {
        "score": float("-inf"),
        "model_info": {
            "trial_number": None,
            "params": {},
            "model_state": None,
            "config": {},
        },
    }

    # Initialize PBT manager
    pbt_manager = PBTManager(population_size=4)

    try:
        study.optimize(
            partial(
                objective,
                model_config=model_config,
                texts=texts,
                labels=labels,
                best_model_info=global_best_info,
                trial_pbar=trial_pbar,
                pbt_manager=pbt_manager,  # Pass PBT manager to objective
            ),
            n_trials=n_trials or model_config.n_trials,
            timeout=timeout,
            callbacks=[
                # Add new callback for logging best info
                lambda study, trial: (
                    log_current_best(global_best_info, experiment_name, model_config)
                    if trial.value
                    and trial.value > study.user_attrs.get("best_value", float("-inf"))
                    else None
                ),
                save_trial_callback(experiment_name, model_config, global_best_info),
                lambda study, trial: pbt_manager.update_population(  # Add PBT callback
                    trial.value or float("-inf"), trial.params, trial.number
                ),
            ],
            gc_after_trial=True,
            catch=(optuna.TrialPruned,),  # Add this to handle pruned trials gracefully
        )
    finally:
        trial_pbar.close()

        # Update the final best trial printing format
        if (
            global_best_info is not None
            and "model_info" in global_best_info
            and global_best_info["model_info"] is not None
        ):
            info = global_best_info["model_info"]

            print("\nBest Trial Configuration")
            print("=" * 80)
            print(f"Study: {experiment_name}")
            print(
                f"Trial: {info['trial_number']} (score: {global_best_info['score']:.4f})"
            )
            print(
                f"Epochs: {info['epoch'] + 1}/{model_config.num_epochs} (early stopping)"
            )  # Add epochs info
            print("\nModel Architecture:")
            print(f"  Hidden layers: {info['config']['hidden_dim']}")
            print(f"  Activation: {info['config']['activation']}")
            print(f"  Dropout rate: {info['config']['dropout_rate']:.4f}")
            print("\nOptimizer Settings:")
            print(f"  Type: {info['config']['optimizer']}")
            print(f"  Learning rate: {info['config']['lr']:.6f}")
            print(f"  Weight decay: {info['config']['weight_decay']:.6f}")
            print(f"  Warmup ratio: {info['config']['warmup_ratio']:.4f}")
            print("=" * 80)

            save_best_trial(
                global_best_info["model_info"], experiment_name, model_config
            )

    return study.best_trial.params


def save_best_trial(
    best_model_info: Dict[str, Any], trial_study_name: str, model_config: ModelConfig
) -> None:
    """Save the best trial model and configuration."""
    model_config.best_trials_dir.mkdir(exist_ok=True, parents=True)
    final_model_path = (
        model_config.best_trials_dir / f"best_model_{trial_study_name}.pt"
    )

    try:
        # Debug logging
        logger.info("Saving trial with performance:")
        logger.info(
            "Trial Score in best_model_info: %s",
            best_model_info.get(f"{model_config.metric}_score"),
        )
        logger.info("Model State Dict Size: %d", len(best_model_info["model_state"]))

        metric_key = f"{model_config.metric}_score"
        save_dict = {
            "model_state_dict": best_model_info["model_state"],
            "config": best_model_info["config"],  # Direct config without nesting
            "metric_value": best_model_info[metric_key],
            "study_name": trial_study_name,
            "trial_number": best_model_info["trial_number"],
            "num_classes": model_config.num_classes,
            "hyperparameters": best_model_info["params"],
            "val_size": 0.2,
            "metric": model_config.metric,
            "bert_encoder_path": str(model_config.bert_encoder_path),  # Add this line
        }
        torch.save(save_dict, final_model_path)
        logger.info(
            "Best trial metric (%s): %s", metric_key, best_model_info[metric_key]
        )
        logger.info("Saved best model to %s", final_model_path)
        logger.info("Best %s: %.4f", model_config.metric, best_model_info[metric_key])
    except Exception as e:
        logger.error("Failed to save best trial: %s", str(e))
        raise IOError(f"Failed to save best trial: {str(e)}") from e


# First save location: During optimization in the objective function
def setup_training_components(
    model_config, classifier_config, optimizer_name, optimizer_config, train_dataloader
):
    """Set up model, trainer, optimizer and scheduler."""
    model = BERTClassifier(
        bert_encoder_path=str(model_config.bert_encoder_path),  # Use bert_encoder_path
        num_classes=model_config.num_classes,
        classifier_config=classifier_config,
    )
    trainer = Trainer(model, model_config)

    optimizer = create_optimizer(optimizer_name, model.parameters(), **optimizer_config)

    total_steps = len(train_dataloader) * model_config.num_epochs
    warmup_steps = int(
        total_steps * classifier_config["warmup_ratio"]
    )  # Using config value
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    return model, trainer, optimizer, scheduler


def calculate_hidden_sizes(
    bert_hidden_size: int, num_classes: int, num_layers: int
) -> List[int]:
    """Calculate hidden layer sizes that decrease geometrically from BERT size to near num_classes.

    Args:
        bert_hidden_size: Size of BERT hidden layer
        num_classes: Number of output classes
        num_layers: Number of hidden layers to generate

    Returns:
        List of hidden layer sizes
    """
    if num_layers == 0:
        return []

    # Start with double BERT size and decrease geometrically
    start_size = 2 * bert_hidden_size
    min_size = max(num_classes * 2, 64)  # Don't go smaller than this

    # Calculate ratio for geometric decrease
    ratio = (min_size / start_size) ** (1.0 / (num_layers + 1))

    sizes = []
    current_size = start_size
    for _ in range(num_layers):
        current_size = int(current_size * ratio)
        # Round to nearest even number and ensure minimum size
        current_size = max(min_size, 2 * (current_size // 2))
        sizes.append(current_size)

    return sizes


class MemoryManager:
    """Manage memory during optimization."""

    def __init__(self):
        self.memory_threshold = 0.85  # 85% memory usage threshold
        self.batch_size_limits = {"min": 8, "max": 64, "current": 32}

    def check_memory(self) -> float:
        """Get current memory usage as a percentage.

        Returns:
            float: Memory usage percentage (0-100) for the current process.

        Note:
            Uses psutil.Process().memory_percent() which returns the process's
            current memory utilization as a percentage of total system memory.
        """
        return psutil.Process().memory_percent()

    def adjust_batch_size(self, current_memory: float) -> int:
        """Dynamically adjust batch size based on memory usage."""
        if current_memory > self.memory_threshold:
            self.batch_size_limits["current"] = max(
                self.batch_size_limits["current"] // 2, self.batch_size_limits["min"]
            )
        return self.batch_size_limits["current"]


def get_trial_config(
    trial: optuna.Trial, model_config: ModelConfig
) -> Tuple[int, Dict, str, Dict]:
    """Get trial configuration parameters."""
    # Add memory-aware batch size selection
    memory_manager = MemoryManager()
    current_memory = memory_manager.check_memory()
    max_batch = memory_manager.adjust_batch_size(current_memory)

    # Use dynamic batch size options
    batch_sizes = [size for size in [8, 16, 32, 64] if size <= max_batch]
    batch_size = trial.suggest_categorical("batch_size", batch_sizes)

    # Get number of hidden layers
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 4)

    # Calculate hidden dimensions based on BERT size and number of classes
    hidden_dims = calculate_hidden_sizes(
        bert_hidden_size=CLASSIFIER_DEFAULTS[
            "bert_hidden_size"
        ],  # Now CLASSIFIER_DEFAULTS is defined
        num_classes=model_config.num_classes,
        num_layers=num_hidden_layers,
    )

    # Updated activation function options with only supported activations
    activation_functions = [
        "relu",  # Standard ReLU
        "gelu",  # Gaussian Error Linear Unit
        "silu",  # Sigmoid Linear Unit (Swish)
        "elu",  # Exponential Linear Unit
        "tanh",  # Hyperbolic Tangent
        "leaky_relu",  # Leaky ReLU
        "prelu",  # Parametric ReLU
    ]

    activation = trial.suggest_categorical("activation", activation_functions)

    classifier_config = {
        "hidden_dim": hidden_dims,
        "activation": activation,  # Now using suggested activation
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.6),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
    }

    optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "sgd", "rmsprop"])
    learning_rate = trial.suggest_float(
        "lr", 1e-6, 5e-3, log=True
    )  # Changed parameter name

    optimizer_config = get_optimizer_config(trial, optimizer_name, learning_rate)
    classifier_config.update(
        {
            "lr": learning_rate,  # Changed from learning_rate
            "optimizer": optimizer_name,
            "optimizer_config": optimizer_config.copy(),
        }
    )

    return batch_size, classifier_config, optimizer_name, optimizer_config


def log_current_best(
    best_info: Dict[str, Any], study_name: str, model_config: ModelConfig
) -> None:
    """Log current best trial information in a structured format."""
    if not best_info.get("model_info"):
        return

    separator = "=" * 80
    info = best_info["model_info"]

    log_msg = [
        "\nCurrent Best Trial Configuration",
        separator,
        f"Study: {study_name}",
        f"Trial: {info['trial_number']} (score: {best_info['score']:.4f})",
        f"Epochs: {info['epoch'] + 1}/{model_config.num_epochs} (early stopping)",  # Add epochs info
        "\nModel Architecture:",
        f"  Hidden layers: {info['config']['hidden_dim']}",
        f"  Activation: {info['config']['activation']}",
        f"  Dropout rate: {info['config']['dropout_rate']:.4f}",
        "\nOptimizer Settings:",
        f"  Type: {info['config']['optimizer']}",
        f"  Learning rate: {info['config']['lr']:.6f}",
        f"  Weight decay: {info['config']['weight_decay']:.6f}",
        f"  Warmup ratio: {info['config']['warmup_ratio']:.4f}",
        separator,
    ]

    logger.info("\n".join(log_msg))


def log_trial_config(
    trial_num: int,
    classifier_config: Dict[str, Any],
    total_trials: int,
    score: Optional[float] = None,
) -> None:
    """Log trial configuration in a clean, structured format."""
    hidden_dims = classifier_config["hidden_dim"]
    opt_config = classifier_config["optimizer_config"]
    activation = classifier_config.get("activation", "gelu")

    logger.info("\nTrial %d of %d:", trial_num + 1, total_trials)
    logger.info("=" * 50)
    logger.info("Architecture:")
    logger.info("  Hidden layers: %s", hidden_dims)
    logger.info("  Activation: %s", activation)
    logger.info("  Dropout rate: %.3f", classifier_config["dropout_rate"])
    logger.info("  Weight decay: %.2e", classifier_config["weight_decay"])
    logger.info("  Warmup ratio: %.2f", classifier_config["warmup_ratio"])

    if classifier_config["optimizer"] == "rmsprop":
        logger.info("  Momentum: %.3f", opt_config["momentum"])
        logger.info("  Alpha: %.3f", opt_config["alpha"])
    elif classifier_config["optimizer"] == "sgd":
        logger.info("  Momentum: %.3f", opt_config["momentum"])
        logger.info("  Nesterov: %s", opt_config.get("nesterov", False))

    if score is not None:
        logger.info("\nScore:")
        logger.info("  f1: %.4f", score)
    logger.info("=" * 50)


def log_trial_summary(
    trial_num: int, trial_config: Dict[str, Any], score: float, total_trials: int
) -> None:
    """Log a clean summary of trial configuration and performance."""
    # Only log if we have a score (i.e., at end of trial)
    if score is None:
        return

    tqdm.write("\n" + "=" * 50)
    tqdm.write(f"Trial {trial_num + 1} of {total_trials}:")
    tqdm.write("=" * 50)
    tqdm.write("Architecture:")
    tqdm.write(f"  Hidden layers: {trial_config['hidden_dim']}")
    tqdm.write(f"  Activation: {trial_config['activation']}")
    tqdm.write(f"  Dropout rate: {trial_config['dropout_rate']:.3f}")
    tqdm.write("")
    tqdm.write(f"Optimizer: {trial_config['optimizer']}")
    tqdm.write(
        f"  Learning rate: {trial_config['lr']:.2e}"
    )  # Changed from learning_rate to lr
    tqdm.write(f"  Weight decay: {trial_config['weight_decay']:.2e}")
    tqdm.write(f"  Warmup ratio: {trial_config['warmup_ratio']:.2f}")

    opt_config = trial_config["optimizer_config"]
    if trial_config["optimizer"] == "rmsprop":  # Changed from config to trial_config
        tqdm.write(f"  Momentum: {opt_config['momentum']:.3f}")
        tqdm.write(f"  Alpha: {opt_config['alpha']:.3f}")
    elif trial_config["optimizer"] == "sgd":  # Changed from config to trial_config
        tqdm.write(f"  Momentum: {opt_config['momentum']:.3f}")
        tqdm.write(f"  Nesterov: {opt_config.get('nesterov', False)}")

    tqdm.write("\nScore:")
    tqdm.write(f"  f1: {score:.4f}")
    tqdm.write("=" * 50 + "\n")


class EarlyStoppingManager:
    """Enhanced early stopping with moving averages and trend detection."""

    def __init__(
        self,
        patience: int = 5,
        min_epochs: int = 5,
        improvement_threshold: float = 0.001,
        smoothing_window: int = 3,
        trend_window: int = 5,
    ):
        self.patience = patience
        self.min_epochs = min_epochs
        self.improvement_threshold = improvement_threshold
        self.smoothing_window = smoothing_window
        self.trend_window = trend_window

        # State tracking
        self.best_score = float("-inf")
        self.best_epoch = 0
        self.scores: Deque[float] = deque(maxlen=trend_window)
        self.no_improve_count = 0

    def should_stop(self, epoch: int, score: float) -> Tuple[bool, str]:
        """Determine if training should stop based on multiple criteria."""
        if epoch < self.min_epochs:
            return False, "Continuing - below minimum epochs"

        self.scores.append(score)
        smooth_score = self._get_smooth_score()

        # Update best score if improved significantly
        if smooth_score > self.best_score + self.improvement_threshold:
            self.best_score = smooth_score
            self.best_epoch = epoch
            self.no_improve_count = 0
            return False, "New best score"

        self.no_improve_count += 1

        # Check multiple stopping conditions
        reasons = []

        # 1. No improvement for too long
        if self.no_improve_count >= self.patience:
            reasons.append(f"No improvement for {self.patience} epochs")

        # 2. Performance regression
        if len(self.scores) >= 3:
            avg_recent = np.mean(list(self.scores)[-3:])
            if avg_recent < self.best_score * 0.8:  # 20% drop
                reasons.append("Significant performance regression")

        # 3. Detect negative trend
        if len(self.scores) >= self.trend_window:
            trend = self._calculate_trend()
            if trend < -0.01:  # Negative trend threshold
                reasons.append("Negative performance trend")

        return bool(reasons), " & ".join(reasons)

    def _get_smooth_score(self) -> float:
        """Calculate smoothed score using moving average."""
        if len(self.scores) < self.smoothing_window:
            return list(self.scores)[-1]
        return np.mean(list(self.scores)[-self.smoothing_window :])

    def _calculate_trend(self) -> float:
        """Calculate the trend of recent scores using linear regression."""
        if len(self.scores) < self.trend_window:
            return 0.0
        x = np.arange(len(self.scores))
        y = np.array(list(self.scores))
        return np.polyfit(x, y, 1)[0]  # Return slope


def objective(
    trial: optuna.Trial,
    model_config: ModelConfig,
    texts: List[str],
    labels: List[int],
    best_model_info: Dict[str, Any],
    trial_pbar: Optional[tqdm] = None,
    pbt_manager: Optional[PBTManager] = None,
) -> float:
    """Optimization objective function with PBT."""
    trial_best_score = 0.0
    classifier_config = None
    current_params = None
    trial_best_state = None  # Initialize at the top with other variables

    try:
        # Get the logger for the model
        model_logger = logging.getLogger("src.models.model")
        original_level = model_logger.level  # Save original level
        model_logger.setLevel(logging.WARNING)  # Suppress INFO messages

        batch_size, classifier_config, optimizer_name, optimizer_config = (
            get_trial_config(trial, model_config)
        )
        current_params = trial.params.copy()

        # Create train/val split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=trial.number, stratify=labels
        )

        # Create dataloaders
        train_dataloader, val_dataloader = create_dataloaders(
            [train_texts, val_texts],
            [train_labels, val_labels],
            model_config,
            batch_size,
        )

        # Setup initial training components
        model, trainer, optimizer, scheduler = setup_training_components(
            model_config,
            classifier_config,
            optimizer_name,
            optimizer_config,
            train_dataloader,
        )

        # Initialize early stopping
        early_stopping = EarlyStoppingManager(
            patience=max(3, min(8, trial.number // 2)),
            min_epochs=max(5, model_config.num_epochs // 4),
        )

        # Training loop with PBT
        for epoch in range(model_config.num_epochs):
            try:
                # Train for one epoch
                trainer.train_epoch(train_dataloader, optimizer, scheduler)
                score, metrics = trainer.evaluate(val_dataloader)
                trial.report(score, epoch)

                # PBT: Check if we should exploit better configurations
                if pbt_manager and epoch > 0:  # Skip first epoch
                    if pbt_manager.should_explore(score):
                        # Get better params from population
                        better_params = pbt_manager.exploit(score, current_params)
                        if better_params:
                            # Update trial parameters with better ones
                            current_params = better_params.copy()
                            # Explore around the better parameters
                            current_params = pbt_manager.explore(current_params)

                            # Update model configuration with new parameters
                            for key, value in current_params.items():
                                if key in classifier_config:
                                    classifier_config[key] = value

                            # Re-initialize training components with new config
                            model, trainer, optimizer, scheduler = (
                                setup_training_components(
                                    model_config,
                                    classifier_config,
                                    current_params.get("optimizer", optimizer_name),
                                    optimizer_config,
                                    train_dataloader,
                                )
                            )
                            logger.info("PBT: Updated configuration in epoch %d", epoch)

                # Update best state if improved
                if score > trial_best_score:
                    trial_best_score = score
                    trial_best_state = {
                        "model_state": {
                            k: v.cpu().clone() for k, v in model.state_dict().items()
                        },
                        "config": classifier_config.copy(),
                        f"{model_config.metric}_score": trial_best_score,
                        "trial_number": trial.number,
                        "params": current_params.copy(),
                        "epoch": epoch,
                        "metrics": metrics,
                        "bert_encoder_path": str(
                            model_config.bert_encoder_path
                        ),  # Add this line
                    }

                # Early stopping checks
                should_stop, reason = early_stopping.should_stop(epoch, score)
                if should_stop:
                    logger.info("\nEarly stopping trial %d: %s", trial.number, reason)
                    break

                # Check for pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.error("Error in epoch %d: %s", epoch, str(e))
                raise

        # Update progress and save best result
        if trial_pbar:
            trial_pbar.update(1)
        if trial_best_state and trial_best_score > best_model_info["score"]:
            best_model_info["score"] = trial_best_score
            best_model_info["model_info"] = trial_best_state

        return trial_best_score

    except optuna.TrialPruned:
        raise
    except Exception as e:
        if trial_pbar:
            trial_pbar.update(1)
        logger.error("Trial %d failed: %s", trial.number, str(e), exc_info=True)
        raise
    finally:
        # Restore original logging level
        model_logger.setLevel(original_level)


def save_trial_callback(
    trial_study_name: str, model_config: ModelConfig, best_model_info: Dict[str, Any]
):
    """Create a callback for saving trial information."""

    def callback(study: optuna.Study, trial: optuna.Trial):
        if trial.value and trial.value > study.user_attrs.get(
            "best_value", float("-inf")
        ):
            study.set_user_attr("best_value", trial.value)
            model_config.best_trials_dir.mkdir(exist_ok=True, parents=True)

            # Get model info
            model_info = best_model_info.get("model_info", {})

            # Create detailed training info
            training_details = {
                "study_name": trial_study_name,
                "trial_number": trial.number,
                "score": trial.value,
                "total_epochs": model_config.num_epochs,
                "completed_epochs": model_info.get("epoch", 0) + 1,
                "early_stopping": True
                if model_info.get("epoch", 0) + 1 < model_config.num_epochs
                else False,
                "architecture": {
                    "hidden_layers": model_info.get("config", {}).get("hidden_dim"),
                    "activation": model_info.get("config", {}).get("activation"),
                    "dropout_rate": model_info.get("config", {}).get("dropout_rate"),
                },
                "optimizer": {
                    "type": model_info.get("config", {}).get("optimizer"),
                    "learning_rate": model_info.get("config", {}).get("lr"),
                    "weight_decay": model_info.get("config", {}).get("weight_decay"),
                    "warmup_ratio": model_info.get("config", {}).get("warmup_ratio"),
                },
            }

            # Save everything including bert_encoder_path
            best_trial_info = ModelCheckpoint(
                model_state_dict=model_info.get("model_state"),
                config=model_info.get("config", {}),
                num_classes=model_config.num_classes,
                bert_encoder_path=str(model_config.bert_encoder_path),
                metric_value=trial.value,
                hyperparameters=trial.params,
                training_details=training_details,
            )

            torch.save(
                best_trial_info.__dict__,
                model_config.best_trials_dir / f"best_trial_{trial_study_name}.pt",
            )

    return callback


def load_best_configuration(best_trials_dir: Path, exp_name: str = None) -> dict:
    """Load best model configuration from optimization results.

    Args:
        best_trials_dir: Directory containing trial results.
        exp_name: Optional experiment name filter.

    Returns:
        dict: Best configuration or None if not found.
    """
    pattern = f"best_trial_{exp_name or '*'}.pt"
    trial_files = list(best_trials_dir.glob(pattern))

    if not trial_files:
        logger.warning("No previous optimization results found")
        return None

    # Find the best performing trial
    best_trial = None
    best_value = float("-inf")

    for file in trial_files:
        trial_data = torch.load(file, map_location="cpu", weights_only=False)
        if trial_data["value"] > best_value:
            best_value = trial_data["value"]
            best_trial = trial_data

    return best_trial


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BERT Classifier Hyperparameter Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Get defaults from CONFIG
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(CONFIG["output_root"]),  # From CONFIG
        help="Root directory for all operations",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="bert_optimization",
        help="Name for the optimization study",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=CONFIG.get("optimization", {}).get("n_trials", 100),  # From CONFIG
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=CONFIG.get("logging", {}).get("verbosity", 1),  # From CONFIG
        choices=[0, 1, 2],
        help="Verbosity level",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=CONFIG["model"]["max_seq_len"],  # From CONFIG
        help="Maximum sequence length for tokenization",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = ModelConfig.from_args(args)
    setup_logging(config)

    try:
        best_params = run_optimization(
            config,
            experiment_name=args.study_name,
            n_trials=args.n_trials,
        )
        logger.info("Best parameters: %s", best_params)

    except Exception as e:
        logger.error("Error during optimization: %s", str(e), exc_info=True)
        raise
