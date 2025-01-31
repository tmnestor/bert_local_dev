import argparse
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml

import torch

from .base_config import BaseConfig
from .defaults import (
    MODEL_DEFAULTS, 
    CLASSIFIER_DEFAULTS,
    DIR_DEFAULTS,
    MODEL_PATHS,
    DATA_DEFAULTS  # Add this import
)

# Add logger at module level
logger = logging.getLogger(__name__)
VALID_METRICS = {'accuracy', 'f1', 'precision', 'recall'}

@dataclass
class ModelConfig(BaseConfig):
    # Class-level defaults for argparse
    DEFAULT_DATA_FILE = Path(DATA_DEFAULTS['default_file'])
    DEFAULT_NUM_EPOCHS = MODEL_DEFAULTS['num_epochs']
    DEFAULT_BATCH_SIZE = MODEL_DEFAULTS['batch_size']
    DEFAULT_LEARNING_RATE = MODEL_DEFAULTS['learning_rate']
    DEFAULT_DEVICE = MODEL_DEFAULTS['device']
    DEFAULT_HIDDEN_DROPOUT = MODEL_DEFAULTS['hidden_dropout']
    DEFAULT_MAX_SEQ_LEN = MODEL_DEFAULTS['max_seq_len']
    DEFAULT_METRIC = MODEL_DEFAULTS['metric']
    DEFAULT_METRICS = MODEL_DEFAULTS['metrics']

    # Instance fields with defaults from config.yml
    bert_model_name: str = './bert_encoder'
    num_classes: Optional[int] = None
    max_seq_len: int = MODEL_DEFAULTS['max_seq_len']
    batch_size: int = MODEL_DEFAULTS['batch_size']
    num_epochs: int = MODEL_DEFAULTS['num_epochs']
    learning_rate: float = MODEL_DEFAULTS['learning_rate']
    device: str = MODEL_DEFAULTS['device']
    hidden_dropout: float = MODEL_DEFAULTS['hidden_dropout']
    metric: str = MODEL_DEFAULTS['metric']
    metrics: List[str] = field(
        default_factory=lambda: MODEL_DEFAULTS['metrics']
    )
    data_file: Path = Path(DATA_DEFAULTS['default_file'])
    n_trials: Optional[int] = field(default=100)
    n_experiments: int = 1
    trials_per_experiment: Optional[int] = field(default=None)
    sampler: str = 'tpe'
    output_root: Path = DIR_DEFAULTS['output_root']
    
    # Remove default values for paths that should be under output_root
    model_save_path: Path = field(init=False)  # Will be set in post_init
    best_trials_dir: Path = field(init=False)
    evaluation_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    
    def __post_init__(self):
        """Initialize directory paths after initialization."""
        self._init_directories()
        # Ensure model_save_path is under output_root/best_trials
        self.model_save_path = self.best_trials_dir / "bert_classifier.pth"
        
    def _init_directories(self):
        """Initialize directory configurations from file or defaults."""
        # First try project root, then environment variable
        config_paths = [
            Path.cwd() / 'directories.yml',  # Project root
            Path(os.environ.get('BERT_DIR_CONFIG', 'config/directories.yml'))  # Fallback
        ]
        
        config_file = next((p for p in config_paths if p.exists()), None)
        
        try:
            if (config_file):
                logger.info(f"Loading directory configuration from: {config_file}")
                with open(config_file) as f:
                    dir_config = yaml.safe_load(f)
                output_root = Path(dir_config.get('output_root', DIR_DEFAULTS['output_root']))
                dirs = dir_config.get('dirs', DIR_DEFAULTS['dirs'])
                model_paths = dir_config.get('model_paths', {'bert_encoder': MODEL_PATHS['bert_encoder']})
            else:
                logger.info("No configuration file found, using defaults")
                output_root = DIR_DEFAULTS['output_root']
                dirs = DIR_DEFAULTS['dirs']
                model_paths = {'bert_encoder': MODEL_PATHS['bert_encoder']}
                
            self.output_root = output_root
            # Setup standard directories (remove checkpoint_dir)
            self.best_trials_dir = output_root / dirs['best_trials']
            self.evaluation_dir = output_root / dirs['evaluation']
            self.logs_dir = output_root / dirs['logs']
            self.data_dir = output_root / dirs['data']
            self.models_dir = output_root / dirs['models']
            
            # Update data_file to be under data_dir if it's not an absolute path
            if not Path(self.data_file).is_absolute():
                self.data_file = self.data_dir / self.data_file.name
            
            # Add debug logging for BERT encoder path
            logger.info("Configuring BERT encoder path:")
            logger.info("  Output root: %s", output_root)
            logger.info("  Model paths from config: %s", model_paths)
            logger.info("  BERT encoder relative path: %s", model_paths['bert_encoder'])
            
            # Setup BERT encoder path - can be absolute or relative to output_root
            bert_path = Path(model_paths['bert_encoder'])
            if not bert_path.is_absolute():
                bert_path = output_root / bert_path
            self.bert_model_name = str(bert_path)
            logger.info("  BERT model path: %s", self.bert_model_name)
            logger.info("  Path exists: %s", Path(self.bert_model_name).exists())
            
        except Exception as e:
            logger.warning(f"Failed to load directory config: {e}")
            self._init_default_directories()
            
    def _init_default_directories(self):
        """Initialize directories with default values."""
        self.best_trials_dir = self.output_root / DIR_DEFAULTS['dirs']['best_trials']
        self.evaluation_dir = self.output_root / DIR_DEFAULTS['dirs']['evaluation']
        self.logs_dir = self.output_root / DIR_DEFAULTS['dirs']['logs']
        self.data_dir = self.output_root / DIR_DEFAULTS['dirs']['data']

    def _validate_metrics(self, metrics: List[str]) -> None:
        """Validate metrics configuration"""
        if metrics is None:
            self.metrics = ["accuracy", "f1", "precision", "recall"]  # Set default if None
            return
            
        if not isinstance(metrics, list):
            raise ValueError("metrics must be a list")
            
        invalid_metrics = set(metrics) - VALID_METRICS
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. Valid options are: {VALID_METRICS}")

    def _validate_metric(self, metric: str) -> None:
        """Validate primary metric"""
        if self.metrics is None:
            self._validate_metrics(None)  # Initialize defaults if needed
            
        if metric not in VALID_METRICS:
            raise ValueError(f"Primary metric '{metric}' must be one of {VALID_METRICS}")

    @classmethod
    def add_argparse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add ModelConfig arguments to an ArgumentParser"""
        # Training settings
        training = parser.add_argument_group('Training Configuration')
        training.add_argument('--num_epochs', type=int, default=cls.DEFAULT_NUM_EPOCHS, 
                            help='Number of training epochs')
        training.add_argument('--batch_size', type=int, default=cls.DEFAULT_BATCH_SIZE,
                            help='Training batch size')
        training.add_argument('--learning_rate', type=float, default=cls.DEFAULT_LEARNING_RATE,
                            help='Learning rate')
        training.add_argument('--hidden_dropout', type=float, default=cls.DEFAULT_HIDDEN_DROPOUT,
                            help='Hidden layer dropout rate')

        # Model settings
        model = parser.add_argument_group('Model Configuration')
        model.add_argument('--bert_model_name', type=str, default=cls.bert_model_name,
                          help='Name or path of the pre-trained BERT model')
        model.add_argument('--num_classes', type=int, default=cls.num_classes,
                          help='Number of output classes')
        model.add_argument('--max_seq_len', type=int, default=cls.DEFAULT_MAX_SEQ_LEN,  # Changed from max_length to max_seq_len
                          help='Maximum sequence length for BERT tokenizer')

        # System settings
        system = parser.add_argument_group('System Configuration')
        system.add_argument('--device', type=str, default=cls.DEFAULT_DEVICE,
                          choices=['cpu', 'cuda'], help='Device to use for training')
        system.add_argument('--n_trials', type=int, default=cls.n_trials,
                          help='Total number of trials (used when trials_per_experiment not set)')
        system.add_argument('--sampler', type=str, default=cls.sampler,
                          choices=['tpe', 'random', 'cmaes', 'qmc'],
                          help='Optuna sampler to use')
        system.add_argument('--metric', type=str, default=cls.DEFAULT_METRIC,
                          choices=['f1', 'accuracy'],
                          help='Metric to use for model assessment')

        # File paths
        paths = parser.add_argument_group('File Paths')
        paths.add_argument('--data_file', type=Path, default=cls.DEFAULT_DATA_FILE,
                          help='Path to input data file')
        # Remove default value for model_save_path, will be set based on output_root
        paths.add_argument('--model_save_path', type=Path,
                          help='Path to save the trained model')
        # Remove default value for best_trials_dir, will be set based on output_root
        paths.add_argument('--best_trials_dir', type=Path,
                          help='Directory to save best trial models and results')

        # Experiment settings
        experiment = parser.add_argument_group('Experiment Configuration')
        experiment.add_argument('--n_experiments', type=int, default=cls.n_experiments,
                              help='Number of experiments to run')
        experiment.add_argument('--trials_per_experiment', type=int, default=cls.trials_per_experiment,
                              help='Number of trials per experiment. If not set, uses n_trials')
        
        # Add directory configuration arguments
        dirs = parser.add_argument_group('Directory Configuration')
        dirs.add_argument('--output_root', type=Path, 
                         default=DIR_DEFAULTS['output_root'],
                         help='Root directory for all outputs')
        dirs.add_argument('--dir_config', type=Path,
                         default='config/directories.yml',
                         help='Path to directory configuration YAML')
        
        # Remove bert_encoder_path argument
        model_paths = parser.add_argument_group('Model Paths')
        # No additional arguments needed here now

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ModelConfig':
        """Create config from argparse namespace"""
        config_dict = {f.name: getattr(args, f.name, None) for f in fields(cls)}
        return cls.from_dict(config_dict)

    def validate(self) -> None:
        """Extended validation with parent validation"""
        # Validate metrics first to ensure they're initialized
        self._validate_metrics(self.metrics)
        # Then validate the primary metric
        self._validate_metric(self.metric)
        # Finally call parent validation
        super().validate()
        
        # Model-specific validation
        self._validate_training_params()
        self._validate_system_params()
        self._validate_experiment_params()
        self._validate_model_paths()

    def _validate_training_params(self) -> None:
        """Validate training-related parameters"""
        if self.num_classes is not None and self.num_classes < 1:
            raise ValueError("if specified, num_classes must be positive")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be positive")
        if not (0.0 < self.learning_rate < 1.0):
            raise ValueError("learning_rate must be between 0 and 1")
        if not (0.0 <= self.hidden_dropout <= 1.0):
            raise ValueError("hidden_dropout must be between 0 and 1")

    def _validate_system_params(self) -> None:
        """Validate system-related parameters"""
        if not torch.cuda.is_available() and self.device.startswith("cuda"):
            raise RuntimeError("CUDA device requested but CUDA is not available")
        if self.metric not in ['f1', 'accuracy']:
            raise ValueError("metric must be either 'f1' or 'accuracy'")

    def _validate_experiment_params(self) -> None:
        """Validate experiment-related parameters"""
        if self.n_experiments < 1:
            raise ValueError("n_experiments must be positive")
        
        # Handle trials_per_experiment - it's allowed to be None
        if self.trials_per_experiment is not None:
            if self.trials_per_experiment < 1:
                raise ValueError("trials_per_experiment must be positive when set")
            # Update n_trials based on trials_per_experiment if it's set
            self.n_trials = self.n_experiments * self.trials_per_experiment
        elif self.n_trials is None:
            # Set default n_trials if neither is specified
            self.n_trials = 100
            
        # Final validation of n_trials
        if self.n_trials < 1:
            raise ValueError("n_trials must be positive")
    
    def _validate_model_paths(self) -> None:
        """Validate model paths."""
        if not Path(self.bert_model_name).exists():
            raise FileNotFoundError(
                f"BERT model not found at: {self.bert_model_name}\n"
                "Please specify correct path in directories.yml or via --bert_model_name"
            )

@dataclass
class ValidationConfig(ModelConfig):
    """Configuration for model validation."""
    test_file: Optional[Path] = field(default=None)
    output_dir: Path = field(init=False)  # Remove default value, will be set in post_init
    model_path: Optional[Path] = field(default=None)
    threshold: float = 0.5
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])
    save_predictions: bool = True
    
    def __post_init__(self):
        """Initialize paths after parent initialization."""
        # First call parent's initialization for directory setup
        super().__post_init__()
        # Then set output_dir to be under output_root/evaluation
        self.output_dir = self.evaluation_dir
    
    DEFAULT_METRICS = ["accuracy", "f1", "precision", "recall"]
    
    @classmethod
    def add_argparse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add validation-specific command line arguments"""
        # First add base model arguments
        super().add_argparse_args(parser)
        
        # Then add validation-specific arguments
        validation = parser.add_argument_group('Validation Configuration')
        validation.add_argument('--test_file', type=Path, default=None,
                              help='Path to test data CSV (default: automatic test split)')
        validation.add_argument('--output_dir', type=Path, default=cls.output_dir,
                              help='Directory to save validation results')
        validation.add_argument('--threshold', type=float, default=cls.threshold,
                              help='Classification threshold')
        validation.add_argument('--metrics', nargs='+', type=str, 
                              default=cls.DEFAULT_METRICS,
                              choices=cls.DEFAULT_METRICS,
                              help='Metrics to compute')
        validation.add_argument('--save_predictions', type=bool, default=cls.save_predictions,
                              help='Whether to save predictions to file')

    def _resolve_paths(self) -> None:
        """Resolve model and test file paths"""
        # Try to find best model if none specified
        if self.model_path is None:
            self.model_path = self._find_best_model()
        if self.model_path is None:
            raise ValueError("No model path specified and no best model found in best_trials_dir")
        
        # If no test file specified, look for auto-split
        if self.test_file is None:
            test_split = self.data_file.parent / "test_split.csv"
            if test_split.exists():
                self.test_file = test_split
            else:
                raise ValueError("No test file specified and no test split found")

    def validate(self) -> None:
        """Validate validation configuration"""
        # First validate base configuration
        super().validate()
        
        # Resolve paths before validation
        self._resolve_paths()
        
        # Then validate validation-specific fields
        if self.threshold < 0 or self.threshold > 1:
            raise ValueError("threshold must be between 0 and 1")
        if not self.metrics:
            raise ValueError("at least one metric must be specified")
        if not isinstance(self.save_predictions, bool):
            raise ValueError("save_predictions must be a boolean")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Finally validate paths exist
        self._validate_path_fields()

    def _validate_path_fields(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.test_file.exists():
            raise FileNotFoundError(f"Test file not found: {self.test_file}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _find_best_model(self) -> Optional[Path]:
        """Find best model from either training or tuning"""
        if not self.best_trials_dir.exists():
            logger.warning("Best trials directory not found: %s", self.best_trials_dir)
            return None
            
        # First try tuning models
        model_files = list(self.best_trials_dir.glob("best_model_*.pt"))
        if model_files:
            best_model = None
            best_score = float('-inf')
            
            for file in model_files:
                try:
                    checkpoint = torch.load(file, map_location='cpu', weights_only=False)
                    # Look for metric value consistently
                    score = checkpoint.get('metric_value', float('-inf'))
                    if score > best_score:
                        best_score = score
                        best_model = file
                        logger.info("Found better model with score %f: %s", score, file)
                except (IOError, RuntimeError, torch.serialization.pickle.UnpicklingError) as e:
                    logger.warning("Couldn't load %s: %s", file, str(e))
                    continue
                    
            if best_model:
                return best_model
                
        # Try training checkpoint as fallback
        checkpoint = self.best_trials_dir / "bert_classifier.pth"
        if checkpoint.exists():
            logger.info("Found training checkpoint: %s", checkpoint)
            return checkpoint
            
        logger.warning("No valid model found in best trials directory")
        return None

@dataclass
class EvaluationConfig(ModelConfig):
    """Configuration for model evaluation"""
    best_model: Path = field(default=None)
    output_dir: Path = field(init=False)
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])
    
    DEFAULT_METRICS = ["accuracy", "f1", "precision", "recall"]
    
    def __post_init__(self):
        """Initialize paths after parent initialization."""
        # First do parent initialization to set up directories
        super().__post_init__()
        
        # Set output_dir to evaluation_dir
        self.output_dir = self.evaluation_dir
        
        # Convert relative best_model path to absolute using output_root if needed
        if self.best_model and not self.best_model.is_absolute():
            # Look for best_model relative to output_root first
            best_model_path = self.output_root / self.best_model
            if best_model_path.exists():
                self.best_model = best_model_path
            else:
                # Try relative to best_trials_dir
                best_model_path = self.best_trials_dir / self.best_model.name
                if best_model_path.exists():
                    self.best_model = best_model_path
                else:
                    logger.warning(f"Best model not found at either:\n"
                                 f"  {best_model_path}\n"
                                 f"  {self.best_trials_dir / self.best_model.name}")
    
    @classmethod
    def add_argparse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add evaluation-specific command line arguments"""
        super().add_argparse_args(parser)
        
        eval_group = parser.add_argument_group('Evaluation')
        eval_group.add_argument('--best_model', type=Path, required=True,
                              help='Path to trained model checkpoint')
        eval_group.add_argument('--output_dir', type=Path,
                              help='Directory to save evaluation results')
        # Fix metrics default value to use class constant
        eval_group.add_argument('--metrics', nargs='+', type=str,
                              default=cls.DEFAULT_METRICS,  # Use class constant
                              choices=cls.DEFAULT_METRICS,
                              help='Metrics to compute')

    def validate(self) -> None:
        """Validate evaluation configuration"""
        super().validate()
        if not self.best_model:
            raise ValueError("best_model path must be specified")
        if not self.best_model.exists():
            raise FileNotFoundError(f"Model file not found: {self.best_model}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

def load_best_configuration(best_trials_dir: Path, study_name: str = None) -> Optional[dict]:
    """Load best model configuration from optimization results"""
    # ...existing code...
    
    if best_trial:
        _log_best_configuration(best_file, best_value)
        # Return the actual configuration without architecture type checks
        if 'params' in best_trial:
            config = best_trial['params']
            # Add default values for missing configuration keys
            config.update({
                'hidden_dim': config.get('hidden_dim', [256, 218]),
                'dropout_rate': config.get('dropout_rate', 0.1),
            })
            return config

# ...existing code...
