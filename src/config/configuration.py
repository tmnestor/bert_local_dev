"""Unified configuration management for BERT classifier.

This module provides complete configuration management including:
1. YAML configuration loading with anchor support
2. Command line argument processing
3. Configuration validation
4. Directory structure management
5. Type-safe configuration classes
"""

import argparse
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .base_config import BaseConfig

# Setup logging
logger = logging.getLogger(__name__)

# Define valid metrics
VALID_METRICS = {"accuracy", "f1", "precision", "recall"}


def join_path_tag(loader: yaml.SafeLoader, node: yaml.Node) -> str:
    """Custom YAML tag handler for !join."""
    parts = loader.construct_sequence(node)
    return str(Path(*[str(p) for p in parts]))


def load_yaml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML with custom tag support."""
    if config_path is None:
        config_path = Path.cwd() / "config.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"No configuration file found at {config_path}")

    # Register custom YAML tag
    yaml.SafeLoader.add_constructor("!join", join_path_tag)

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}") from e

    # Convert paths and special values
    _process_config_values(config)

    return config


def _process_config_values(config: Dict[str, Any]) -> None:
    """Process configuration values after loading."""
    # First process paths
    if "output_root" in config:
        config["output_root"] = Path(config["output_root"])

    # Process data files
    if "data" in config and "files" in config["data"]:
        data_root = config["output_root"] / config["dirs"]["data"]
        for key, path in config["data"]["files"].items():
            config["data"]["files"][key] = str(data_root / Path(path).name)
            logger.debug("Resolved path for %s: %s", key, config["data"]["files"][key])

    # Process optimizer settings based on chosen optimizer
    if "optimizer" in config:
        optimizer_choice = config["optimizer"].get("optimizer_choice", "adamw")
        logger.debug("Processing optimizer config for: %s", optimizer_choice)

        # IMPORTANT: First preserve common optimizer parameters
        common_params = {
            "optimizer_choice": optimizer_choice,
            "lr": config["optimizer"]["lr"],
            "weight_decay": config["optimizer"]["weight_decay"],
            "warmup_ratio": config["optimizer"][
                "warmup_ratio"
            ],  # Always preserve warmup_ratio
        }

        # Process optimizer-specific parameters
        if optimizer_choice == "sgd":
            common_params.update(
                {
                    "momentum": config["optimizer"]["momentum"],
                    "nesterov": config["optimizer"]["nesterov"],
                }
            )
        elif optimizer_choice == "rmsprop":
            common_params.update(
                {
                    "momentum": config["optimizer"]["momentum"],
                    "alpha": config["optimizer"]["alpha"],
                }
            )
        elif optimizer_choice == "adamw":
            betas = config["optimizer"]["betas"]
            if isinstance(betas, str):
                betas = tuple(float(x.strip()) for x in betas.strip("()").split(","))
            common_params.update(
                {"betas": betas, "eps": float(config["optimizer"]["eps"])}
            )

        # Replace optimizer config with processed values
        config["optimizer"] = common_params

        logger.debug("Final optimizer config: %s", config["optimizer"])


# Load config after function definitions
CONFIG = load_yaml_config()


def resolve_path(path: Path, root_dir: Path) -> Path:
    """Resolves a path relative to a root directory."""
    if not path.is_absolute():
        return root_dir / path
    return path


@dataclass
class ModelConfig(BaseConfig):
    """Strongly typed configuration for model training.

    Attributes:
        bert_encoder_path (Path): Path to the BERT encoder model.
        bert_model_name (str): Name or path of the BERT model.
        num_classes (Optional[int]): Number of output classes.
        max_seq_len (int): Maximum sequence length for tokenization.
        batch_size (int): Training batch size.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate.
        device (str): Device for training (cpu or cuda).
        dropout_rate (float): Dropout rate.
        metric (str): Evaluation metric.
        metrics (List[str]): List of metrics to compute.
        data_file (Path): Path to the main data file.
        n_trials (Optional[int]): Number of optimization trials.
        n_experiments (int): Number of optimization experiments.
        trials_per_experiment (Optional[int]): Trials per experiment.
        sampler (str): Sampler for Optuna.
        output_root (Path): Root directory for all outputs.
        verbosity (int): Verbosity level.
    """

    bert_encoder_path: Path = field(
        default_factory=lambda: Path(CONFIG["model_paths"]["bert_encoder"]),
        metadata={"help": "Path to the BERT encoder model"},
    )
    bert_model_name: str = field(
        default_factory=lambda: CONFIG["model"].get(
            "bert_model_name", "./bert_encoder"
        ),
        metadata={"help": "Name or path of the BERT model"},
    )
    num_classes: Optional[int] = field(
        default_factory=lambda: CONFIG["classifier"].get("num_classes"),
        metadata={"help": "Number of output classes"},
    )
    max_seq_len: int = field(
        default_factory=lambda: CONFIG["model"]["max_seq_len"],
        metadata={"help": "Maximum sequence length for tokenization"},
    )
    batch_size: int = field(
        default_factory=lambda: CONFIG["model"]["batch_size"],
        metadata={"help": "Training batch size"},
    )
    num_epochs: int = field(
        default_factory=lambda: CONFIG["model"]["num_epochs"],
        metadata={"help": "Number of training epochs"},
    )
    learning_rate: float = field(
        default_factory=lambda: CONFIG["optimizer"]["lr"],
        metadata={"help": "Learning rate"},
    )
    device: str = field(
        default_factory=lambda: CONFIG["model"]["device"],
        metadata={"help": "Device for training (cpu or cuda)"},
    )
    dropout_rate: float = field(
        default_factory=lambda: CONFIG["classifier"]["dropout_rate"],
        metadata={"help": "Dropout rate"},
    )
    metric: str = field(
        default_factory=lambda: CONFIG["model"]["metric"],
        metadata={"help": "Evaluation metric"},
    )
    metrics: List[str] = field(
        default_factory=lambda: CONFIG["model"]["metrics"],
        metadata={"help": "List of metrics to compute"},
    )
    data_file: Path = field(
        default_factory=lambda: Path(CONFIG["data"]["files"]["default"]),
        metadata={"help": "Path to the main data file"},
    )
    n_trials: Optional[int] = field(
        default_factory=lambda: CONFIG.get("optimization", {}).get("n_trials", 100),
        metadata={"help": "Number of optimization trials"},
    )
    n_experiments: int = field(
        default_factory=lambda: CONFIG.get("optimization", {}).get("n_experiments", 1),
        metadata={"help": "Number of optimization experiments"},
    )
    trials_per_experiment: Optional[int] = None
    sampler: str = field(
        default_factory=lambda: CONFIG.get("optimization", {}).get("sampler", "tpe"),
        metadata={"help": "Sampler for Optuna"},
    )
    output_root: Path = field(
        default_factory=lambda: Path(CONFIG["output_root"]),
        metadata={"help": "Root directory for all outputs"},
    )
    verbosity: int = field(
        default_factory=lambda: CONFIG.get("logging", {}).get("verbosity", 1),
        metadata={"help": "Verbosity level"},
    )

    def __post_init__(self):
        """Initialize paths after creation."""
        # Initialize directories
        self.best_trials_dir = self.output_root / "best_trials"
        self.evaluation_dir = self.output_root / "evaluation_results"
        self.logs_dir = self.output_root / "logs"
        self.data_dir = self.output_root / "data"
        self.model_save_path = self.best_trials_dir / "best_model.pt"

        # Always load data_file from config.yml
        config = load_yaml_config()
        self.data_file = Path(config["data"]["files"]["default"])

        # Resolve data_file relative to output_root
        self.data_file = resolve_path(self.data_file, self.output_root / "data")

        # Resolve bert_encoder_path relative to output_root if not absolute
        self.bert_encoder_path = resolve_path(self.bert_encoder_path, self.output_root)

        # Create necessary directories
        self.best_trials_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ModelConfig":
        """Create config from argparse namespace."""
        # Convert args to dictionary
        config_dict = {
            field.name: getattr(args, field.name, None)
            for field in fields(cls)
            if hasattr(args, field.name)
        }

        # Create instance and let post_init handle path resolution
        return cls(**config_dict)

    @classmethod
    def add_argparse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add model configuration arguments to parser."""
        # Model settings
        model_group = parser.add_argument_group("Model Configuration")
        model_group.add_argument(
            "--bert_model_name",
            type=str,
            default=CONFIG["model"].get("bert_model_name", "./bert_encoder"),
            help="Path to BERT model",
        )
        model_group.add_argument(
            "--num_classes",
            type=int,
            default=CONFIG["classifier"].get("num_classes"),
            help="Number of output classes",
        )
        model_group.add_argument(
            "--max_seq_len",
            type=int,
            default=CONFIG["model"]["max_seq_len"],
            help="Maximum sequence length",
        )
        model_group.add_argument(
            "--batch_size",
            type=int,
            default=CONFIG["model"]["batch_size"],
            help="Batch size",
        )

        # Training settings
        train_group = parser.add_argument_group("Training Configuration")
        train_group.add_argument(
            "--learning_rate",
            type=float,
            default=CONFIG["optimizer"]["lr"],
            help="Learning rate",
        )
        train_group.add_argument(
            "--num_epochs",
            type=int,
            default=CONFIG["model"]["num_epochs"],
            help="Number of epochs",
        )
        train_group.add_argument(
            "--device",
            type=str,
            default=CONFIG["model"]["device"],
            choices=["cpu", "cuda"],
            help="Device for training",
        )

        # Path settings
        path_group = parser.add_argument_group("Paths")
        path_group.add_argument(
            "--output_root",
            type=Path,
            default=Path(CONFIG["output_root"]),
            help="Root directory for outputs",
        )
        path_group.add_argument(
            "--data_file",
            type=Path,
            default=CONFIG["data"]["files"]["default"],
            help="Input data file",
        )

        # Other settings
        other_group = parser.add_argument_group("Other")
        other_group.add_argument(
            "--verbosity",
            type=int,
            default=CONFIG.get("logging", {}).get("verbosity", 1),
            choices=[0, 1, 2],
            help="Verbosity level",
        )


@dataclass
class ValidationConfig(ModelConfig):
    """Configuration for model validation."""

    test_file: Optional[Path] = field(default=None)
    model_path: Optional[Path] = field(default=None)
    # ...rest of ValidationConfig implementation...


@dataclass
class EvaluationConfig(ModelConfig):
    """Configuration for model evaluation."""

    best_model: Path = field(default=None)
    output_dir: Path = field(init=False)
    metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "f1", "precision", "recall"]
    )
    n_folds: int = field(default=7)  # Add k-fold parameter

    DEFAULT_METRICS = ["accuracy", "f1", "precision", "recall"]

    def __post_init__(self):
        """Initialize paths after parent initialization."""
        # First do parent initialization to set up directories
        super().__post_init__()
        # Then set output_dir to evaluation_dir
        self.output_dir = self.evaluation_dir

    @classmethod
    def add_argparse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add evaluation-specific command line arguments."""
        # First add parent's arguments
        super().add_argparse_args(parser)

        # Add evaluation-specific arguments
        eval_group = parser.add_argument_group("Evaluation")
        eval_group.add_argument(
            "--best_model",
            type=Path,
            required=True,
            help="Path to trained model checkpoint",
        )
        eval_group.add_argument(
            "--metrics",
            nargs="+",
            type=str,
            default=cls.DEFAULT_METRICS,
            choices=cls.DEFAULT_METRICS,
            help="Metrics to compute",
        )
        eval_group.add_argument(
            "--n_folds", type=int, default=7, help="Number of cross-validation folds"
        )


@dataclass
class PredictionConfig(ModelConfig):
    """Configuration for prediction tasks."""

    best_model: Path = field(default=None)
    output_dir: Path = field(init=False)

    def __post_init__(self):
        """Initialize paths after parent initialization."""
        super().__post_init__()  # This will set bert_encoder_path correctly
        # Set up prediction-specific paths
        self.output_dir = self.predictions_dir = self.output_root / "predictions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Ensure bert_encoder_path is absolute
        if not self.bert_encoder_path.is_absolute():
            self.bert_encoder_path = self.output_root / self.bert_encoder_path
        logger.debug("Using BERT encoder from: %s", self.bert_encoder_path)

    @classmethod
    def add_argparse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add prediction-specific command line arguments."""
        # First add parent's arguments
        super().add_argparse_args(parser)

        # Add prediction-specific arguments
        predict_group = parser.add_argument_group("Prediction")
        predict_group.add_argument(
            "--best_model",
            type=str,
            required=True,
            help="Model file name (relative to best_trials_dir)",
        )
        predict_group.add_argument(
            "--output_file",
            type=str,
            default="predictions.csv",
            help="Output filename for predictions",
        )


def get_config(args: Optional[argparse.Namespace] = None) -> ModelConfig:
    """Central configuration factory function.

    This is the main entry point for getting configuration instances.
    It handles:
    1. Loading YAML config
    2. Processing command line args
    3. Environment variables
    4. Validation
    """
    # Load base configuration
    config = load_yaml_config()

    # Create config instance
    if args:
        instance = ModelConfig.from_args(args)
    else:
        # Ensure data_file path is set from config
        if "data" in config and "files" in config["data"]:
            config["data_file"] = config["data"]["files"]["default"]
        instance = ModelConfig.from_dict(config)

    # Validate after paths are resolved
    instance.validate()

    return instance


# Export commonly used defaults
CONFIG = load_yaml_config()
DIR_DEFAULTS = {"output_root": CONFIG["output_root"], "dirs": CONFIG["dirs"]}
DATA_DEFAULTS = {"files": CONFIG["data"]["files"]}
MODEL_DEFAULTS = {k: v for k, v in CONFIG["model"].items()}
CLASSIFIER_DEFAULTS = {k: v for k, v in CONFIG["classifier"].items()}
OPTIMIZER_DEFAULTS = {k: v for k, v in CONFIG["optimizer"].items()}
