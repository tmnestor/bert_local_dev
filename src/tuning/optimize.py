#!/usr/bin/env python
import argparse
import random
import time
import warnings
from collections import deque
from functools import partial
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple


import numpy as np
import psutil
import optuna
import torch
from optuna._experimental import ExperimentalWarning
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from torch import optim
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from ..config.config import ModelConfig
from ..config.defaults import (  # Update imports
    CLASSIFIER_DEFAULTS,  # Add this import
)
from ..data_utils import create_dataloaders, load_and_preprocess_data
from ..models.model import BERTClassifier
from ..training.trainer import Trainer
from ..utils.logging_manager import (
    get_logger,
    setup_logging,
)  # Change from setup_logger
from ..utils.train_utils import log_separator

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
    # Map common parameter names to optimizer-specific names
    param_mapping = {
        "lr": "lr",  # Changed from learning_rate
        "weight_decay": "weight_decay",
        "momentum": "momentum",
        "beta1": "betas[0]",
        "beta2": "betas[1]",
    }

    # Convert parameters using mapping
    optimizer_kwargs = {}
    for key, value in kwargs.items():
        if key in param_mapping:
            mapped_key = param_mapping[key]
            if "[" in mapped_key:  # Handle nested params like betas
                base_key, idx = mapped_key.split("[")
                idx = int(idx.rstrip("]"))
                if base_key not in optimizer_kwargs:
                    optimizer_kwargs[base_key] = [0.9, 0.999]  # Default AdamW betas
                optimizer_kwargs[base_key][idx] = value
            else:
                optimizer_kwargs[mapped_key] = value
        else:
            optimizer_kwargs[key] = value

    optimizers = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
    }

    if optimizer_name.lower() not in optimizers:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Available optimizers: {list(optimizers.keys())}"
        )

    return optimizers[optimizer_name.lower()](model_params, **optimizer_kwargs)


def _create_study(
    name: str,
    storage: Optional[str] = None,
    sampler_type: str = "tpe",
    random_seed: Optional[int] = None,
) -> optuna.Study:
    """Create an Optuna study for hyperparameter optimization."""
    # Silence optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if random_seed is None:
        random_seed = int(time.time())

    sampler = {
        "tpe": TPESampler(
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True,
            warn_independent_sampling=False,  # Suppress warnings
            seed=random_seed,
        ),
        "random": optuna.samplers.RandomSampler(seed=random_seed),
        "cmaes": optuna.samplers.CmaEsSampler(n_startup_trials=10, seed=random_seed),
        "qmc": optuna.samplers.QMCSampler(qmc_type="sobol", seed=random_seed),
    }.get(sampler_type, TPESampler(seed=random_seed))

    # Adjust pruner to be less aggressive
    pruner = HyperbandPruner(
        min_resource=3,  # Increased from 1
        max_resource=15,  # Increased from 10
        reduction_factor=2,  # Decreased from 3
    )

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
    # Only show detailed info if verbosity > 0
    if model_config.verbosity > 0:
        log_separator(logger)
        logger.info("Starting optimization")
        logger.info("Number of trials: %s", n_trials or model_config.n_trials)
        logger.info("\nLoading data...")

    # Load data - Fix unpacking by capturing all returned values
    train_texts, val_texts, train_labels, val_labels, label_encoder = (
        load_and_preprocess_data(model_config)
    )

    if model_config.num_classes is None:
        model_config.num_classes = len(label_encoder.classes_)

    # Combine train and val sets for optimization
    texts = train_texts + val_texts
    labels = train_labels + val_labels

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

    # Create study and run optimization
    study = _create_study(experiment_name, storage, model_config.sampler, random_seed)
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
                save_trial_callback(experiment_name, model_config, global_best_info),
                lambda study, trial: pbt_manager.update_population(  # Add PBT callback
                    trial.value or float("-inf"), trial.params, trial.number
                ),
            ],
            gc_after_trial=True,
        )
    finally:
        trial_pbar.close()

        # Fix the condition and null check
        if (
            global_best_info is not None
            and "model_info" in global_best_info
            and global_best_info["model_info"] is not None
        ):
            print("\nBest Trial Configuration:")
            print("=" * 50)
            print(f"Trial Number: {global_best_info['model_info']['trial_number']}")
            print(f"Score: {global_best_info['score']:.4f}")
            print("\nHyperparameters:")
            for key, value in global_best_info["model_info"]["params"].items():
                print(f"  {key}: {value}")
            print("=" * 50)

            # Always save best model regardless of verbosity
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
        model_config.bert_model_name, model_config.num_classes, classifier_config
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

    # Add activation function as a trial parameter
    activation = trial.suggest_categorical(
        "activation", ["relu", "gelu", "silu", "tanh"]
    )

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


def get_optimizer_config(trial, optimizer_name, lr):  # Changed parameter name
    """Get optimizer-specific configuration."""
    optimizer_config = {
        "lr": lr,  # Changed from learning_rate
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


def log_current_best(best_info: Dict[str, Any]) -> None:
    """Log the current best trial configuration in a standardized format."""
    if not best_info["model_info"]:
        return

    info = best_info["model_info"]
    params = info["params"]
    clf_config = info["config"]

    logger.info("\nCurrent Best Trial Configuration:")
    logger.info("=" * 50)
    logger.info("Trial Number: %d", info["trial_number"])
    logger.info("Score: %.4f", best_info["score"])
    logger.info("\nHyperparameters:")
    logger.info("  batch_size: %d", params["batch_size"])
    logger.info("  hidden_layers: %s", clf_config["hidden_dim"])
    logger.info("  dropout_rate: %.4f", clf_config["dropout_rate"])
    logger.info("  weight_decay: %.6f", clf_config["weight_decay"])
    logger.info("  warmup_ratio: %.2f", clf_config["warmup_ratio"])
    logger.info("  optimizer: %s", clf_config["optimizer"])
    logger.info("  lr: %.6f", clf_config["lr"])  # Changed from learning_rate to lr
    logger.info("=" * 50)


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
    """Optimization objective function for a single trial.

    Handles the training and evaluation of a single hyperparameter configuration trial,
    including Population Based Training updates if enabled.

    Args:
        trial (optuna.Trial): Current optimization trial.
        model_config (ModelConfig): Model configuration object.
        texts (List[str]): List of input texts.
        labels (List[int]): List of corresponding labels.
        best_model_info (Dict[str, Any]): Dictionary tracking best model information.
        trial_pbar (Optional[tqdm]): Progress bar for trial tracking.
        pbt_manager (Optional[PBTManager]): Population Based Training manager.

    Returns:
        float: Trial score (metric value).

    Raises:
        optuna.TrialPruned: If trial is pruned.
        Exception: If training fails.
    """
    trial_best_score = 0.0
    classifier_config = None

    try:
        batch_size, classifier_config, optimizer_name, optimizer_config = (
            get_trial_config(trial, model_config)
        )

        # Only show trial progress if verbosity > 0
        if model_config.verbosity > 0:
            log_trial_summary(
                trial.number, classifier_config, None, model_config.n_trials
            )

        # Create train/val split here for each trial
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

        # Setup training components
        model, trainer, optimizer, scheduler = setup_training_components(
            model_config,
            classifier_config,
            optimizer_name,
            optimizer_config,
            train_dataloader,
        )

        # Initialize early stopping manager - replaces unused patience variables
        early_stopping = EarlyStoppingManager(
            patience=max(3, min(8, trial.number // 2)),
            min_epochs=max(5, model_config.num_epochs // 4),
            improvement_threshold=0.001,
            smoothing_window=3,
            trend_window=5,
        )

        # Training loop
        trial_best_state = None

        # Main training loop
        for epoch in range(model_config.num_epochs):
            try:
                trainer.train_epoch(train_dataloader, optimizer, scheduler)
                score, metrics = trainer.evaluate(val_dataloader)
                trial.report(score, epoch)

                # Check for improvement and update best state
                if score > trial_best_score:
                    trial_best_score = score
                    trial_best_state = {
                        "model_state": {
                            k: v.cpu().clone() for k, v in model.state_dict().items()
                        },
                        "config": classifier_config.copy(),
                        f"{model_config.metric}_score": trial_best_score,
                        "trial_number": trial.number,
                        "params": trial.params.copy(),
                        "epoch": epoch,
                        "metrics": metrics,
                    }

                # Early stopping checks
                should_stop, reason = early_stopping.should_stop(epoch, score)
                if should_stop:
                    # Add newline before early stopping message
                    logger.info("\nEarly stopping triggered: %s", reason)
                    break

                # Optuna pruning check
                if trial.should_prune():
                    raise optuna.TrialPruned(f"Trial pruned at epoch {epoch}")

                # Check if we should perform PBT operations
                if pbt_manager and pbt_manager.should_explore(score):
                    # Either exploit or explore
                    if random.random() < 0.5:
                        new_params = pbt_manager.exploit(score, trial.params)
                        if new_params:
                            # Update trial parameters
                            for key, value in new_params.items():
                                trial.params[key] = value
                            # Recreate optimizer with new parameters
                            optimizer_config = get_optimizer_config(
                                trial,
                                optimizer_name,
                                new_params.get(
                                    "lr"
                                ),  # Changed from learning_rate to lr
                            )
                            optimizer = create_optimizer(
                                optimizer_name, model.parameters(), **optimizer_config
                            )
                    else:
                        # Explore by perturbing current parameters
                        new_params = pbt_manager.explore(trial.params)
                        trial.params.update(new_params)
                        optimizer_config = get_optimizer_config(
                            trial,
                            optimizer_name,
                            new_params.get("lr"),  # Changed from learning_rate to lr
                        )
                        optimizer = create_optimizer(
                            optimizer_name, model.parameters(), **optimizer_config
                        )

            except Exception as e:
                logger.error("Error in epoch %d: %s", epoch, str(e))
                raise

        # Update progress and log final result
        if trial_pbar:
            trial_pbar.update(1)
            # Only show per-trial results if verbosity > 0
            if model_config.verbosity > 0:
                log_trial_summary(
                    trial.number,
                    classifier_config,
                    trial_best_score,
                    model_config.n_trials,
                )

        # Save best state if this is the best trial so far
        if trial_best_state and trial_best_score > best_model_info["score"]:
            best_model_info["score"] = trial_best_score
            best_model_info["model_info"] = trial_best_state

        return trial_best_score

    except Exception as e:
        if trial_pbar:
            trial_pbar.update(1)
        if model_config.verbosity > 0:  # Only log error details if not in minimal mode
            logger.error("Trial %d failed: %s", trial.number, str(e), exc_info=True)
        raise


def save_trial_callback(
    trial_study_name: str, model_config: ModelConfig, best_model_info: Dict[str, Any]
):
    """Create a callback for saving trial information.

    Args:
        trial_study_name: Name of the optimization study.
        model_config: Model configuration instance.
        best_model_info: Dictionary tracking best model state.

    Returns:
        Callable: Callback function for Optuna.
    """

    def callback(study: optuna.Study, trial: optuna.Trial):
        if trial.value and trial.value > study.user_attrs.get(
            "best_value", float("-inf")
        ):
            study.set_user_attr("best_value", trial.value)
            model_config.best_trials_dir.mkdir(exist_ok=True, parents=True)
            # Ensure model_info exists
            model_info = best_model_info.get("model_info", {})
            best_trial_info = {
                "trial_number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "study_name": trial_study_name,
                "best_model_score": best_model_info["score"],
                "model_state": model_info.get("model_state"),
                "config": model_info.get("config", {}),
            }
            torch.save(
                best_trial_info,
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
    """Parse command line arguments for optimization.

    Returns:
        Namespace containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="BERT Classifier Hyperparameter Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ModelConfig.add_argparse_args(parser)

    optim_group = parser.add_argument_group("Optimization")
    optim_group.add_argument(
        "--timeout", type=int, default=None, help="Optimization timeout in seconds"
    )
    optim_group.add_argument(
        "--study_name",
        type=str,
        default="bert_optimization",
        help="Base name for the Optuna study",
    )
    optim_group.add_argument(
        "--storage", type=str, default=None, help="Database URL for Optuna storage"
    )
    optim_group.add_argument(
        "--seed", type=int, default=None, help="Random seed for sampler"
    )

    return parser.parse_args()


def log_best_configuration(best_info: Dict[str, Any]) -> None:
    """Log details about the best trial configuration."""
    logger.info("\nBest Trial Configuration:")
    logger.info("=" * 50)
    logger.info("Trial Number: %d", best_info["model_info"]["trial_number"])
    logger.info("Score: %.4f", best_info["score"])
    logger.info("\nHyperparameters:")
    for key, value in best_info["model_info"]["params"].items():
        logger.info("  %s: %s", key, value)
    logger.info("=" * 50)


if __name__ == "__main__":
    args = parse_args()
    config = ModelConfig.from_args(args)
    setup_logging(config)  # Initialize logging configuration first

    try:
        trials_per_exp = args.trials_per_experiment or args.n_trials
        for experiment_id in range(args.n_experiments):
            study_name = f"{args.study_name}_exp{experiment_id}"
            seed = args.seed + experiment_id if args.seed is not None else None

            best_params = run_optimization(
                config,
                timeout=args.timeout,
                experiment_name=study_name,
                storage=args.storage,
                random_seed=seed,
                n_trials=trials_per_exp,
            )
            logger.info("Experiment %d completed", experiment_id + 1)
            logger.info("Best parameters: %s", best_params)

        logger.info("\nAll experiments completed successfully")
    except Exception as e:
        logger.error("Error during optimization: %s", str(e), exc_info=True)
        raise
