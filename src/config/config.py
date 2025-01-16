from dataclasses import dataclass, fields, field
from typing import Optional, List
from pathlib import Path
import torch
import argparse
import logging
import pickle

from .base_config import BaseConfig

# Add logger at module level
logger = logging.getLogger(__name__)

VALID_METRICS = {'accuracy', 'f1', 'precision', 'recall'}

@dataclass
class ModelConfig(BaseConfig):
    bert_model_name: str = './bert_encoder'  # Update default value
    num_classes: int = 5
    max_seq_len: int = 512  # Changed from max_length
    batch_size: int = 16
    num_epochs: int = 10  # Number of epochs
    learning_rate: float = 2e-5
    device: str = "cpu"
    data_file: Path = Path("data/bbc-text.csv")
    best_trials_dir: Path = Path("best_trials")  # Base directory
    model_save_path: Path = Path("best_trials/bert_classifier.pth")  # Final model path
    hidden_dropout: float = 0.1
    n_trials: Optional[int] = field(default=100)  # Change to use field with default
    n_experiments: int = 1  # Number of experiments to run
    trials_per_experiment: Optional[int] = field(default=None)  # Make explicitly Optional with field
    sampler: str = 'tpe'  # Default sampler is 'tpe'
    metric: str = 'f1'  # Single metric for training/optimization
    metrics: List[str] = field(  # Multiple metrics for evaluation
        default_factory=lambda: ["accuracy", "f1", "precision", "recall"]
    )

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
        training.add_argument('--num_epochs', type=int, default=cls.num_epochs, 
                            help='Number of training epochs')
        training.add_argument('--batch_size', type=int, default=cls.batch_size,
                            help='Training batch size')
        training.add_argument('--learning_rate', type=float, default=cls.learning_rate,
                            help='Learning rate')
        training.add_argument('--hidden_dropout', type=float, default=cls.hidden_dropout,
                            help='Hidden layer dropout rate')

        # Model settings
        model = parser.add_argument_group('Model Configuration')
        model.add_argument('--bert_model_name', type=str, default=cls.bert_model_name,
                          help='Name or path of the pre-trained BERT model')
        model.add_argument('--num_classes', type=int, default=cls.num_classes,
                          help='Number of output classes')
        model.add_argument('--max_seq_len', type=int, default=cls.max_seq_len,  # Changed from max_length to max_seq_len
                          help='Maximum sequence length for BERT tokenizer')

        # System settings
        system = parser.add_argument_group('System Configuration')
        system.add_argument('--device', type=str, default=cls.device,
                          choices=['cpu', 'cuda'], help='Device to use for training')
        system.add_argument('--n_trials', type=int, default=cls.n_trials,
                          help='Total number of trials (used when trials_per_experiment not set)')
        system.add_argument('--sampler', type=str, default=cls.sampler,
                          choices=['tpe', 'random', 'cmaes', 'qmc'],
                          help='Optuna sampler to use')
        system.add_argument('--metric', type=str, default=cls.metric,
                          choices=['f1', 'accuracy'],
                          help='Metric to use for model assessment')

        # File paths
        paths = parser.add_argument_group('File Paths')
        paths.add_argument('--data_file', type=Path, default=cls.data_file,
                          help='Path to input data file')
        paths.add_argument('--model_save_path', type=Path, default=cls.model_save_path,
                          help='Path to save the trained model')
        # Add best trials directory argument
        paths.add_argument('--best_trials_dir', type=Path, default=cls.best_trials_dir,
                          help='Directory to save best trial models and results')

        # Experiment settings
        experiment = parser.add_argument_group('Experiment Configuration')
        experiment.add_argument('--n_experiments', type=int, default=cls.n_experiments,
                              help='Number of experiments to run')
        experiment.add_argument('--trials_per_experiment', type=int, default=cls.trials_per_experiment,
                              help='Number of trials per experiment. If not set, uses n_trials')

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

    def _validate_training_params(self) -> None:
        """Validate training-related parameters"""
        if self.num_classes < 1:
            raise ValueError("num_classes must be positive")
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

@dataclass
class ValidationConfig(ModelConfig):
    """Configuration for model validation."""
    test_file: Optional[Path] = field(default=None)  # This can be None if using auto-split
    output_dir: Path = field(default=Path("validation_results"))
    model_path: Optional[Path] = field(default=None)  # Add this line
    threshold: float = 0.5
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])
    save_predictions: bool = True
    
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
        """Find best model from Optuna trials"""
        if not self.best_trials_dir.exists():
            logger.warning("Best trials directory not found: %s", self.best_trials_dir)
            return None
            
        # First try to find best model file
        model_files = list(self.best_trials_dir.glob("best_model_*.pt"))
        if not model_files:
            # If no best model, try checkpoint file
            checkpoint = self.best_trials_dir / "bert_classifier.pth"
            if checkpoint.exists():
                logger.info("Found checkpoint file: %s", checkpoint)
                return checkpoint
            logger.warning("No model files found in best trials directory")
            return None
            
        # Find the best performing model
        best_model = None
        best_score = float('-inf')
        
        for model_file in model_files:
            try:
                checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
                score = checkpoint.get('f1_score', 
                       checkpoint.get('accuracy_score',
                       checkpoint.get('metric_value',
                       float('-inf'))))
                if score > best_score:
                    best_score = score
                    best_model = model_file
                    logger.info("Found better model with score %f: %s", score, model_file)
            except (RuntimeError, FileNotFoundError, pickle.UnpicklingError) as e:
                logger.warning("Couldn't load %s: %s", model_file, str(e))
                continue
                
        if best_model is None:
            logger.warning("No valid model found in best trials directory")
        else:
            logger.info("Selected best model: %s", best_model)
            
        return best_model

@dataclass
class EvaluationConfig(ModelConfig):
    """Configuration for model evaluation"""
    best_model: Path = field(default=None)
    output_dir: Path = field(default=Path("evaluation_results"))
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])
    
    @classmethod
    def add_argparse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add evaluation-specific command line arguments"""
        super().add_argparse_args(parser)
        
        eval_group = parser.add_argument_group('Evaluation')
        eval_group.add_argument('--best_model', type=Path, required=True,
                              help='Path to trained model checkpoint')
        eval_group.add_argument('--output_dir', type=Path, 
                              default=cls.output_dir,
                              help='Directory to save evaluation results')
        eval_group.add_argument('--metrics', nargs='+', type=str,
                              default=cls.metrics,
                              choices=["accuracy", "f1", "precision", "recall"],
                              help='Metrics to compute')

    def validate(self) -> None:
        """Validate evaluation configuration"""
        super().validate()
        if not self.best_model:
            raise ValueError("best_model path must be specified")
        if not self.best_model.exists():
            raise FileNotFoundError(f"Model file not found: {self.best_model}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
