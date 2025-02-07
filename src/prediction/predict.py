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

from ..config.configuration import CONFIG, PredictionConfig
from ..data_utils import create_dataloaders, load_and_preprocess_data
from ..evaluation.evaluator import ModelEvaluator, suppress_evaluation_warnings
from ..utils.logging_manager import get_logger, setup_logging

logger = get_logger(__name__)

class Predictor(ModelEvaluator):
    """Handles model prediction without evaluation."""
    
    def predict(self, save_predictions: bool = True, output_file: str = "predictions.csv") -> pd.DataFrame:
        """Generate predictions for input data."""
        with suppress_evaluation_warnings():
            # Load data and get label encoder
            data = load_and_preprocess_data(self.config, validation_mode=True)
            
            # Load prediction data
            try:
                df = pd.read_csv(self.config.data_file)
                texts = df["text"].tolist()
            except Exception as e:
                raise RuntimeError(f"Failed to load input data: {str(e)}") from e

            # Create dataloader without labels
            test_dataloader = create_dataloaders(
                texts=texts,
                labels=[0] * len(texts),  # Dummy labels
                config=self.config,
                batch_size=self.config.batch_size
            )

            # Generate predictions
            all_preds = []
            all_probs = []
            
            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc="Generating predictions"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
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
                df.to_csv(output_path, index=False)
                logger.info("Saved predictions to: %s", output_path)

            return df

    @classmethod
    def from_config(cls, config: PredictionConfig) -> "Predictor":
        """Create predictor instance from configuration."""
        # Just use config.best_model which is already a full path
        return cls(model_path=config.best_model, config=config)

def main():
    """Command-line interface for prediction."""
    parser = argparse.ArgumentParser(description="Generate predictions using trained BERT classifier")
    
    PredictionConfig.add_argparse_args(parser)
    args = parser.parse_args()

    # Initialize config
    config = PredictionConfig.from_args(args)
    setup_logging(config)

    # Resolve paths
    if not Path(config.bert_model_name).is_absolute():
        config.bert_model_name = str(config.output_root / "bert_encoder")

    # Resolve best_model path
    config.best_model = config.best_trials_dir / args.best_model

    logger.info("Using BERT encoder from: %s", config.bert_model_name)
    logger.info("Loading model from: %s", config.best_model)
    
    # Create predictor and generate predictions
    predictor = Predictor.from_config(config)
    predictions_df = predictor.predict(save_predictions=True, output_file=args.output_file)
    
    if config.verbosity > 0:
        logger.info("\nPrediction Summary:")
        logger.info("Total predictions: %d", len(predictions_df))
        logger.info("Average confidence: %.4f", predictions_df["confidence"].mean())

if __name__ == "__main__":
    main()
