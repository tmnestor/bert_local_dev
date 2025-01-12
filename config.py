from dataclasses import dataclass, fields
from typing import Optional, Any
from pathlib import Path
import torch
import argparse

@dataclass
class ModelConfig:
    bert_model_name: str = './all-MiniLM-L6-v2'
    num_classes: int = 5
    max_length: int = 128
    batch_size: int = 16
    num_epochs: int = 10  # Number of epochs
    learning_rate: float = 2e-5
    device: str = "cpu"
    data_file: Path = Path("data/bbc-text.csv")
    best_trials_dir: Path = Path("best_trials")  # Base directory
    model_save_path: Path = Path("best_trials/bert_classifier.pth")  # Final model path
    hidden_dropout: float = 0.1
    n_trials: int = None  # Change default to None
    n_experiments: int = 1  # Number of experiments to run
    trials_per_experiment: int = None  # Trials per experiment, if None uses n_trials
    sampler: str = 'tpe'  # Default sampler is 'tpe'
    metric: str = 'f1'  # Default to F1 score for model assessment

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
        model.add_argument('--max_length', type=int, default=cls.max_length,
                          help='Maximum sequence length')

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
        """Create a ModelConfig instance from parsed arguments"""
        # Get all field names from the dataclass
        field_names = {f.name for f in fields(cls)}
        # Filter args.__dict__ to only include fields that exist in ModelConfig
        config_args = {k: v for k, v in vars(args).items() if k in field_names}
        return cls(**config_args)

    def validate(self) -> None:
        """Validate configuration parameters."""
        # Create best_trials_dir if it doesn't exist
        self.best_trials_dir.mkdir(parents=True, exist_ok=True)
        
        # Create parent directory for model_save_path if it doesn't exist
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Existing validations
        if self.num_classes < 1:
            raise ValueError("num_classes must be positive")
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be positive")
        if not (0.0 < self.learning_rate < 1.0):
            raise ValueError("learning_rate must be between 0 and 1")
        if not (0.0 <= self.hidden_dropout <= 1.0):
            raise ValueError("hidden_dropout must be between 0 and 1")
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        if not torch.cuda.is_available() and self.device.startswith("cuda"):
            raise RuntimeError("CUDA device requested but CUDA is not available")
        
        # Validate experiment settings and calculate n_trials
        if self.n_experiments < 1:
            raise ValueError("n_experiments must be positive")
        if self.trials_per_experiment is not None and self.trials_per_experiment < 1:
            raise ValueError("trials_per_experiment must be positive")
            
        # Calculate total trials if not explicitly set
        if self.n_trials is None:
            if self.trials_per_experiment is not None:
                self.n_trials = self.n_experiments * self.trials_per_experiment
            else:
                self.n_trials = 100  # Default fallback value
        elif self.n_trials < 1:
            raise ValueError("n_trials must be positive")
        
        # Validate metric
        if self.metric not in ['f1', 'accuracy']:
            raise ValueError("metric must be either 'f1' or 'accuracy'")
