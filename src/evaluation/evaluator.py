from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from ..config.config import EvaluationConfig, ModelConfig
from ..models.model import BERTClassifier
from ..utils.metrics import calculate_metrics
from ..data_utils import load_and_preprocess_data, create_dataloaders  # Changed import
from ..utils.logging_manager import setup_logger
from ..utils.model_loading import safe_load_checkpoint, verify_state_dict

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
        self.model_path = Path(model_path)  # Store model_path directly in the instance
        self.config = config
        self.device = torch.device(config.device)
        self.metrics = getattr(config, 'metrics', self.DEFAULT_METRICS)
        self.model = self._load_model()
        
    def _load_model(self) -> BERTClassifier:
        """Load model from checkpoint."""
        try:
            # Load checkpoint safely
            checkpoint = safe_load_checkpoint(
                self.model_path,
                self.device,
                weights_only=True
            )
            classifier_config = checkpoint['config']['classifier_config']
            
            # Create model instance
            model = BERTClassifier(
                self.config.bert_model_name,
                checkpoint.get('num_classes', self.config.num_classes),
                classifier_config
            )
            
            # Verify and load state dict
            verify_state_dict(checkpoint['model_state_dict'], model)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            # Log model details
            logger.info(f"\nLoading model from checkpoint: {self.model_path}")
            logger.info("\nModel Selection Details:")
            logger.info("Checkpoint file: %s", self.model_path)
            logger.info("Model file size: %.2f MB", self.model_path.stat().st_size / (1024 * 1024))
            logger.info("Checkpoint keys: %s", list(checkpoint.keys()))
            
            logger.info("\nOptimization History:")
            logger.info("Study Name: %s", checkpoint.get('study_name', 'N/A'))
            logger.info("Trial Number: %s", checkpoint.get('trial_number', 'N/A'))
            logger.info("Optimization Metric: %s", checkpoint.get('metric', 'N/A'))
            logger.info("Best Optimization Score: %.4f", checkpoint.get('metric_value', float('nan')))
            logger.info("Validation Split Size: %s", checkpoint.get('val_size', 'N/A'))
            
            arch_type = classifier_config.get('architecture_type', 'standard')
            logger.info("\nArchitecture Configuration:")
            logger.info("Type: %s", arch_type)
            logger.info("Learning Rate: %.6f", classifier_config.get('learning_rate', 0.0))
            logger.info("Weight Decay: %.6f", classifier_config.get('weight_decay', 0.0))
            
            if arch_type == 'standard':
                bert_hidden_size = self.config.bert_model_name.config.hidden_size if hasattr(self.config.bert_model_name, 'config') else 768
                layer_sizes = self._calculate_layer_sizes(
                    bert_hidden_size,
                    classifier_config.get('hidden_dim'),
                    classifier_config.get('num_layers'),
                    checkpoint.get('num_classes', self.config.num_classes)
                )
                hidden_layers = layer_sizes[1:-1]

                logger.info("\nStandard Classifier Configuration:")
                logger.info("Number of Layers: %d", classifier_config.get('num_layers', 'N/A'))
                logger.info("Hidden Layer Sizes: %s", hidden_layers)
                logger.info("Activation: %s", classifier_config.get('activation', 'N/A'))
                logger.info("Dropout Rate: %.4f", classifier_config.get('dropout_rate', 'N/A'))
                logger.info("Warmup Ratio: %.3f", classifier_config.get('warmup_ratio', 0.0))
                logger.info("Pooling: Mean pooling with L2 normalization")
            else:
                logger.info("\nPlaneResNet Configuration:")
                logger.info("Number of Planes: %d", classifier_config.get('num_planes', 'N/A'))
                logger.info("Plane Width: %d", classifier_config.get('plane_width', 'N/A'))
                logger.info("Regularization: BatchNorm")
                logger.info("Warmup Ratio: %.3f", classifier_config.get('warmup_ratio', 0.0))

            if 'hyperparameters' in checkpoint:
                logger.info("\nTrial Hyperparameters:")
                for key, value in checkpoint['hyperparameters'].items():
                    logger.info("  %s: %s", key, value)
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}") from e

    def _calculate_layer_sizes(self, input_size: int, hidden_dim: int, num_layers: int, num_classes: int) -> List[int]:
        """Calculate the actual sizes of all layers."""
        if num_layers == 1:
            return [input_size, num_classes]
        
        layer_sizes = [input_size]
        current_size = input_size
        
        if num_layers > 2:
            ratio = (hidden_dim / current_size) ** (1.0 / (num_layers - 1))
            for _ in range(num_layers - 1):
                current_size = int(current_size * ratio)
                current_size = max(current_size, hidden_dim)
                layer_sizes.append(current_size)
        else:
            layer_sizes.append(hidden_dim)
        
        layer_sizes.append(num_classes)
        return layer_sizes
    
    def evaluate(self, 
                save_predictions: bool = True,
                output_dir: Optional[Path] = None) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Evaluate model on test dataset.
        
        Args:
            save_predictions (bool, optional): Whether to save predictions to files.
                Defaults to True.
            output_dir (Optional[Path], optional): Directory to save results.
                Defaults to None, which creates 'evaluation_results' directory.
        
        Returns:
            Tuple[Dict[str, float], pd.DataFrame]: Tuple containing:
                - Dictionary of evaluation metrics
                - DataFrame with predictions and ground truth
        """
        test_texts, test_labels, label_encoder = load_and_preprocess_data(
            self.config, validation_mode=True
        )
        
        logger.info("\nEvaluation Data Split:")
        logger.info("Test set size: %d samples", len(test_texts))
        logger.info("Label distribution:")
        unique_labels, counts = np.unique(test_labels, return_counts=True)
        for label, count in zip(label_encoder.inverse_transform(unique_labels), counts):
            logger.info("  %s: %d samples (%.2f%%)", label, count, 100 * count/len(test_labels))
        
        test_dataloader = create_dataloaders(
            test_texts,
            test_labels,
            self.config,
            self.config.batch_size,
            validation_mode=True
        )
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        logger.info("Starting evaluation (with dropout disabled)...")
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label']
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())
                all_probs.extend(probs.cpu().tolist())
        
        metrics = calculate_metrics(all_labels, all_preds, self.metrics)
        
        if hasattr(self, 'optimization_metric_value'):
            logger.info("\nMetric Comparison:")
            logger.info("Optimization Score: %.4f", self.optimization_metric_value)
            logger.info("Evaluation Score: %.4f", metrics.get(self.config.metric, float('nan')))
            if abs(self.optimization_metric_value - metrics.get(self.config.metric, 0)) > 0.05:
                logger.warning("Large discrepancy between optimization and evaluation scores!")
        
        results_df = pd.DataFrame({
            'text': test_texts,
            'true_label': label_encoder.inverse_transform(all_labels),
            'predicted_label': label_encoder.inverse_transform(all_preds),
            'confidence': [max(probs) for probs in all_probs]
        })
        
        if save_predictions:
            # Use config's evaluation_dir instead of default path
            output_dir = output_dir or self.config.evaluation_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_df.to_csv(output_dir / 'test_predictions.csv', index=False)
            
            report = classification_report(
                all_labels, all_preds,
                target_names=label_encoder.classes_,
                output_dict=True
            )
            pd.DataFrame(report).to_csv(output_dir / 'classification_report.csv')
            
            confusion_df = pd.crosstab(
                results_df['true_label'],
                results_df['predicted_label'],
                margins=True
            )
            confusion_df.to_csv(output_dir / 'confusion_matrix.csv')
            
            logger.info("\nEvaluation Results:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            logger.info("\nConfusion Matrix:")
            logger.info("\n" + str(confusion_df))
            
        return metrics, results_df

    @classmethod
    def from_config(cls, config: EvaluationConfig) -> 'ModelEvaluator':
        """Create evaluator instance from configuration."""
        model_files = list(config.best_trials_dir.glob("best_model_*.pt"))
        if model_files:
            logger.info("\nFound model files:")
            for f in model_files:
                try:
                    # Use safe loading for scanning
                    checkpoint = safe_load_checkpoint(f, 'cpu', strict=False)
                    logger.info("  %s (score: %.4f)", f.name, 
                              checkpoint.get('metric_value', float('nan')))
                except Exception as e:
                    logger.warning("  Failed to load %s: %s", f.name, e)

        logger.info("\nSelected model: %s", config.best_model)
        return cls(model_path=config.best_model, config=config)

    @classmethod
    def add_model_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add model-specific arguments to argument parser.
        
        Args:
            parser (argparse.ArgumentParser): Argument parser to extend.
        """
        model_args = parser.add_argument_group('Model Configuration')
        model_args.add_argument('--device', type=str, default='cpu',
                              choices=['cpu', 'cuda'],
                              help='Device to use for inference')
        model_args.add_argument('--batch_size', type=int, default=32,
                              help='Batch size for evaluation')

# ...existing code...

def main():
    """Command-line interface entry point for model evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate trained BERT classifier')
    
    # Use EvaluationConfig instead of ModelConfig to get proper directory handling
    EvaluationConfig.add_argparse_args(parser)
    
    args = parser.parse_args()
    
    # Create EvaluationConfig instead of ModelConfig
    config = EvaluationConfig.from_args(args)
    
    try:
        logger.info(f"Loading model from: {args.best_model}")
        evaluator = ModelEvaluator(
            model_path=args.best_model,
            config=config  # Pass EvaluationConfig instance
        )
        # Don't override output_dir here - let EvaluationConfig handle it
        metrics, _ = evaluator.evaluate(
            save_predictions=True,
            output_dir=config.evaluation_dir  # Use evaluation_dir from config
        )
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
