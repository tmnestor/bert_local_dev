from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from ..config.config import EvaluationConfig, ModelConfig
from ..models.model import BERTClassifier
from ..utils.metrics import compute_metrics
from ..utils.train_utils import load_and_preprocess_data, create_dataloaders
from ..utils.logging_manager import setup_logger
from ..utils.progress_manager import ProgressManager
from ..models.factory import ModelFactory
from .cross_validator import CrossValidator
from ..utils.model_utils import format_model_info, count_parameters

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report
)
import numpy as np

logger = setup_logger(__name__)

class ModelEvaluator:
    """Model evaluation class for BERT classifiers.
    
    This class handles model loading and evaluation, providing detailed metrics
    and result analysis capabilities.
    
    Attributes:
        model_path (Path): Path to the saved model checkpoint.
        config (Union[EvaluationConfig, ModelConfig]): Model configuration.
        device (torch.device): Device to run evaluation on.
        metrics (List[str]): Metrics to compute during evaluation.
        model (BERTClassifier): Loaded model instance.
    """
    
    DEFAULT_METRICS = ["accuracy", "f1", "precision", "recall"]
    
    def __init__(self, model_path: Path, config: Union[EvaluationConfig, ModelConfig]):
        """Initialize the evaluator.
        
        Args:
            model_path (Path): Path to the saved model checkpoint.
            config (Union[EvaluationConfig, ModelConfig]): Model configuration.
        """
        self.model_path = Path(model_path)
        self.config = config
        self.device = torch.device(config.device)
        self.metrics = getattr(config, 'metrics', self.DEFAULT_METRICS)
        self.model = self._load_model()
        self.progress = ProgressManager()
        
    def _load_model(self) -> BERTClassifier:
        """Load the trained model from checkpoint."""
        try:
            # Try full load first since our checkpoints contain configuration
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False  # Use full load for configuration
            )
            
            logger.info("Successfully loaded checkpoint")
            
            # Extract configurations
            config_dict = checkpoint.get('config', {})
            classifier_config = config_dict.get('classifier_config', {})
            num_classes = checkpoint.get('num_classes', self.config.num_classes)
            
            # Create model params with all necessary configuration
            model_params = {
                'bert_model_name': self.config.bert_model_name,
                'num_classes': num_classes,
                'architecture_type': classifier_config.get('architecture_type', 'standard'),
                'cls_pooling': True,
                **classifier_config  # Include all classifier-specific config
            }
            
            # Create model using factory
            model = ModelFactory.create_model(model_params)
            
            # Load weights with weights_only=True
            state_dict = checkpoint['model_state_dict']
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys in state dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state dict: {unexpected_keys}")
            
            model = model.to(self.device)
            model.eval()
            
            # Log model information
            logger.info("\nModel Architecture:")
            logger.info("-" * 50)
            logger.info(f"Base Model: {self.config.bert_model_name}")
            logger.info(f"Architecture Type: {model_params['architecture_type']}")
            logger.info(f"Number of Classes: {num_classes}")
            logger.info("-" * 50)
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}") from e
    
    def evaluate(
        self,
        save_predictions: bool = True,
        output_dir: Optional[Path] = None,
        cross_validate: bool = False,
        n_splits: int = 7,
        stratify: bool = True,
        **kwargs
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Evaluate model on test dataset."""
        output_dir = output_dir or Path("evaluation_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test data
        test_texts, test_labels, label_encoder = load_and_preprocess_data(
            self.config, validation_mode=True
        )
        
        if cross_validate:
            # Convert labels to numpy array if they aren't already
            if isinstance(test_labels, (list, pd.Series)):
                test_labels = np.array(test_labels)
            
            # Initialize cross-validator
            validator = CrossValidator(n_splits=int(n_splits))  # Ensure integer type
            
            # Run cross-validation directly on the data
            cv_results = validator.validate(
                model=self.model,
                X=test_texts,
                y=test_labels
            )
            
            # Save detailed results if requested
            if save_predictions and output_dir:
                self._save_cv_results(cv_results, output_dir)
            
            return cv_results, pd.DataFrame()
        
        # Regular evaluation code remains unchanged
        test_dataloader = create_dataloaders(
            test_texts,
            test_labels,
            self.config,
            self.config.batch_size,
            validation_mode=True
        )
        
        if cross_validate:
            # Initialize cross-validator with correct arguments
            validator = CrossValidator(self.model)  # Pass model as positional argument
            
            # Run cross-validation
            cv_results = validator.run_cv(
                test_dataloader,
                n_splits=n_splits,
                label_encoder=label_encoder
            )
            
            # Save detailed results if requested
            if save_predictions and output_dir:
                self._save_cv_results(cv_results, output_dir)
            
            return cv_results, pd.DataFrame()
        
        # Regular evaluation
        return self._evaluate_single(test_dataloader, save_predictions, output_dir)
    
    def _save_cv_results(self, results: Dict, output_dir: Path) -> None:
        """Save cross-validation results"""
        # Save detailed metrics
        metrics_file = output_dir / 'cv_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save confusion matrices if available
        if 'confusion_matrices' in results:
            matrices_file = output_dir / 'cv_confusion_matrices.npy'
            np.save(matrices_file, results['confusion_matrices'])
        
        logger.info(f"Saved cross-validation results to {output_dir}")
    
    def _evaluate_single(self, 
                        dataloader: DataLoader,
                        save_predictions: bool = True,
                        output_dir: Optional[Path] = None) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Perform single evaluation pass"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        eval_bar = tqdm(total=len(dataloader), desc="Evaluating", leave=False)
        
        try:
            with torch.no_grad():
                for batch in dataloader:
                    outputs = self.model(
                        input_ids=batch['input_ids'].to(self.device),
                        attention_mask=batch['attention_mask'].to(self.device)
                    )
                    _, preds = torch.max(outputs.logits, dim=1)
                    all_predictions.extend(preds.cpu().numpy())
                    all_labels.extend(batch['label'].cpu().numpy())
                    eval_bar.update(1)

            # Convert to numpy arrays
            predictions = np.array(all_predictions)
            labels = np.array(all_labels)

            # Calculate metrics
            metrics = {}
            for metric_name in self.metrics:
                if metric_name == 'accuracy':
                    metrics['accuracy'] = accuracy_score(labels, predictions)
                elif metric_name == 'f1':
                    metrics['f1'] = f1_score(labels, predictions, average='macro')
                elif metric_name == 'precision':
                    metrics['precision'] = precision_score(labels, predictions, average='macro')
                elif metric_name == 'recall':
                    metrics['recall'] = recall_score(labels, predictions, average='macro')

            # Create predictions DataFrame
            predictions_df = pd.DataFrame({
                'true_label': labels,
                'predicted_label': predictions
            })

            # Save results if requested
            if save_predictions and output_dir:
                self._save_evaluation_results(metrics, predictions_df, output_dir)

            return metrics, predictions_df

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
            
        finally:
            eval_bar.close()
    
    def _save_evaluation_results(self,
                               metrics: Dict[str, float],
                               predictions_df: pd.DataFrame,
                               output_dir: Path) -> None:
        """Save evaluation results to files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Save predictions
        predictions_file = output_dir / 'predictions.csv'
        predictions_df.to_csv(predictions_file, index=False)
        
        # Save classification report
        report = classification_report(
            predictions_df['true_label'],
            predictions_df['predicted_label']
        )
        report_file = output_dir / 'classification_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
            
        logger.info(f"\nSaved evaluation results to {output_dir}")
        logger.info("\nMetrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    def get_predictions(self, dataloader: DataLoader) -> np.ndarray:
        """Get model predictions for a dataloader"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device)
                )
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                
        return np.array(predictions)
    
    def run_cross_validation(self, n_splits: int = 7):
        """Run cross-validation evaluation"""
        test_dataloader = self._create_test_dataloader()
        validator = CrossValidator(self, n_splits)
        return validator.run_cross_validation(test_dataloader)

    @classmethod
    def from_config(cls, config: EvaluationConfig) -> 'ModelEvaluator':
        """Create evaluator instance from configuration.
        
        Args:
            config (EvaluationConfig): Evaluation configuration.
            
        Returns:
            ModelEvaluator: Configured evaluator instance.
        """
        return cls(
            model_path=config.best_model,
            config=config
        )

    @classmethod
    def add_model_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add model-specific arguments to argument parser."""
        model_args = parser.add_argument_group('Model Configuration')
        model_args.add_argument(
            '--device', 
            type=str, 
            default='cpu',
            choices=['cpu', 'cuda'],
            help='Device to use for inference'
        )
        model_args.add_argument(
            '--batch_size', 
            type=int, 
            default=32,
            help='Batch size for evaluation'
        )

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate BERT Classifier')
    
    # Add base model configuration arguments
    ModelConfig.add_argparse_args(parser)
    
    # Add evaluation-specific arguments
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument('--best_model', type=Path,
                          help='Path to trained model checkpoint')
    eval_group.add_argument('--output_dir', type=Path, 
                          default=Path('evaluation_results'),
                          help='Directory to save evaluation results')
    eval_group.add_argument('--metrics', nargs='+', type=str,
                          default=['accuracy', 'f1', 'precision', 'recall'],
                          choices=['accuracy', 'f1', 'precision', 'recall'],
                          help='Metrics to compute')
    
    # Add cross-validation arguments
    cv_group = parser.add_argument_group('Cross-validation')
    cv_group.add_argument('--cross_validate', '-cv', action='store_true',
                         help='Perform cross-validation')
    cv_group.add_argument('--n_splits', type=int, default=7,
                         help='Number of folds for cross-validation')
    cv_group.add_argument('--stratify', action='store_true',
                         help='Use stratified sampling for cross-validation')
    
    return parser.parse_args()

def main():
    """Command-line interface entry point for model evaluation."""
    args = parse_args()
    config = ModelConfig.from_args(args)
    
    # Add evaluation-specific settings to config
    config.best_model = args.best_model
    config.output_dir = args.output_dir
    config.metrics = args.metrics
    
    try:
        logger.info(f"Loading model from: {config.best_model}")
        evaluator = ModelEvaluator.from_config(config)
        
        if args.cross_validate:
            logger.info(f"\nPerforming {args.n_splits}-fold cross-validation")
            metrics = evaluator.evaluate(
                cross_validate=True,
                n_splits=args.n_splits,
                stratify=args.stratify,
                save_predictions=True,
                output_dir=config.output_dir
            )
        else:
            # Regular evaluation
            metrics, _ = evaluator.evaluate(
                save_predictions=True,
                output_dir=config.output_dir
            )
            
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
