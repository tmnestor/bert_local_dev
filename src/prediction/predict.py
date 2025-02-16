"""BERT classifier prediction module for making and saving predictions.

This module provides prediction functionality for trained BERT classifiers:
- Model loading and configuration
- Batch prediction generation
- Confidence score calculation
- Results saving with original data

Typical usage:
    ```python
    config = PredictionConfig.from_args(args)
    predictor = Predictor.from_config(config)
    predictions_df = predictor.predict(save_predictions=True)
    ```
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from ..config.configuration import BaseConfig, load_yaml_config
from ..data_utils import create_dataloaders, load_and_preprocess_data
from ..evaluation.evaluator import ModelEvaluator, suppress_evaluation_warnings
from ..utils.logging_manager import get_logger, setup_logging
from ..utils.model_info import display_model_info  # Add import
from dataclasses import dataclass, field

logger = get_logger(__name__)


@dataclass
class PredictionConfig(BaseConfig):
    """Configuration specific to prediction tasks."""

    output_root: Path = field(default=None)
    data_file: Path = field(default=None)
    best_model: Path = field(default=None)
    output_file: str = field(default="predictions.csv")
    batch_size: int = field(default=32)
    device: str = field(default="cpu")
    verbosity: int = field(default=1)
    num_classes: int = field(default=None)  # Add num_classes field
    max_seq_len: int = field(init=False)  # Add this field

    # Non-init fields
    output_dir: Path = field(init=False)
    best_trials_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    data_dir: Path = field(init=False)  # Add data_dir

    def __post_init__(self):
        """Initialize after creation."""
        # Load defaults from config.yml
        config = load_yaml_config()

        # Convert string paths to Path objects
        if isinstance(self.output_root, str):
            self.output_root = Path(self.output_root)
        if isinstance(self.data_file, str):
            self.data_file = Path(self.data_file)
        if isinstance(self.best_model, str):
            self.best_model = Path(self.best_model)

        # Set defaults from config if not provided
        if self.output_root is None:
            self.output_root = Path(config["output_root"])
        if self.data_file is None:
            self.data_file = Path(config["data"]["files"]["predict"])
        if self.batch_size is None:
            self.batch_size = config["model"]["batch_size"]
        if self.device is None:
            self.device = config["model"]["device"]

        # Set model-specific defaults from config.yml
        if self.num_classes is None:
            self.num_classes = config["classifier"]["num_classes"]
            if self.num_classes is None:
                raise ValueError("num_classes must be set in config.yml")

        # Set max_seq_len from config.yml
        self.max_seq_len = config["model"]["max_seq_len"]

        # Setup all required directories
        self.output_dir = self.output_root / "predictions"
        self.best_trials_dir = self.output_root / "best_trials"
        self.logs_dir = self.output_root / "logs"
        self.data_dir = self.output_root / "data"  # Initialize data_dir

        # Create all directories
        for dir_path in [
            self.output_dir,
            self.best_trials_dir,
            self.logs_dir,
            self.data_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Resolve paths relative to output_root
        if not self.data_file.is_absolute():
            self.data_file = self.output_root / "data" / self.data_file
        if not self.best_model.is_absolute():
            self.best_model = self.best_trials_dir / self.best_model

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PredictionConfig":
        """Create config from argparse namespace."""
        # Extract arguments into a dictionary
        config_dict = {}
        for arg in [
            "data_file",
            "best_model",
            "output_file",
            "batch_size",
            "device",
            "verbosity",
            "output_root",
        ]:
            if hasattr(args, arg):
                config_dict[arg] = getattr(args, arg)

        return cls(**config_dict)

    @classmethod
    def add_argparse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add prediction-specific arguments to parser."""
        parser.add_argument(
            "--output_root",
            type=str,
            required=True,
            help="Root directory for all operations",
        )
        parser.add_argument(
            "--data_file",
            type=str,
            help="Path to input data file for predictions (default from config.yml)",
        )
        parser.add_argument(
            "--best_model",
            type=str,
            required=True,
            help="Path to trained model checkpoint",
        )
        parser.add_argument(
            "--output_file",
            type=str,
            default="predictions.csv",
            help="Output file name for predictions",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            help="Batch size for prediction (default from config.yml)",
        )
        parser.add_argument(
            "--num_classes",
            type=int,
            help="Number of output classes (default from config.yml)",
        )


def save_minimal_predictions(df: pd.DataFrame, output_path: Path) -> None:
    """Save predictions with only required columns and specific names."""
    minimal_df = pd.DataFrame(
        {
            "Hash_Id": df["Hash_Id"],
            "Cleaned_Claim": df["text"],
            "FTC_Label": df["predicted_label"],
        }
    )

    logger.info("Saving minimal predictions with columns: %s", list(minimal_df.columns))
    minimal_df.to_csv(output_path, index=False)
    logger.info("Saved to: %s", output_path)


class Predictor(ModelEvaluator):
    """Handles model prediction without evaluation."""

    @classmethod
    def from_config(cls, config: PredictionConfig) -> "Predictor":
        """Create predictor instance from configuration."""
        # Load config defaults first
        yml_config = load_yaml_config()

        # Ensure bert_encoder_path is set
        if not hasattr(config, "bert_encoder_path"):
            config.bert_encoder_path = Path(yml_config["model_paths"]["bert_encoder"])
            if not config.bert_encoder_path.is_absolute():
                config.bert_encoder_path = config.output_root / config.bert_encoder_path

        return cls(model_path=config.best_model, config=config)

    def predict(
        self, save_predictions: bool = True, output_file: str = "predictions.csv"
    ) -> pd.DataFrame:
        """Generate predictions for input data."""
        with suppress_evaluation_warnings():
            # Load data and get label encoder
            data = load_and_preprocess_data(self.config, validation_mode=True)

            # Create new DataFrame if not loaded
            df = pd.read_csv(self.config.data_file)

            # Create dataloader - self.config now has bert_encoder_path from ModelConfig
            test_dataloader = create_dataloaders(
                texts=data.test_texts,
                labels=data.test_labels,
                config=self.config,  # This config now has bert_encoder_path
                batch_size=self.config.batch_size,
            )

            # Generate predictions
            all_preds = []
            all_probs = []

            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc="Generating predictions"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)

                    all_preds.extend(preds.cpu().tolist())
                    all_probs.extend(probs.cpu().tolist())

            # Add predictions to dataframe
            df["predicted_label_id"] = all_preds
            df["predicted_label"] = data.label_encoder.inverse_transform(all_preds)
            df["confidence"] = [max(probs) for probs in all_probs]

            if save_predictions:
                output_path = self.config.output_dir / output_file
                save_minimal_predictions(df, output_path)

            return df


def main():
    """Command-line interface for prediction."""
    parser = argparse.ArgumentParser(
        description="Generate predictions using trained BERT classifier"
    )

    PredictionConfig.add_argparse_args(parser)
    args = parser.parse_args()

    # Initialize config with defaults from config.yml
    config = PredictionConfig.from_args(args)
    setup_logging(config)

    logger.info("Loading model from: %s", config.best_model)

    logger.info("Using input data from: %s", config.data_file)
    logger.info("Saving predictions to: %s", config.output_dir / config.output_file)

    # Create predictor and generate predictions
    predictor = Predictor.from_config(config)
    predictions_df = predictor.predict(
        save_predictions=True, output_file=config.output_file
    )

    if config.verbosity > 0:
        logger.info("\nPrediction Summary:")
        logger.info("Total predictions: %d", len(predictions_df))
        logger.info("Average confidence: %.4f", predictions_df["confidence"].mean())


if __name__ == "__main__":
    main()
