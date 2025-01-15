from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import torch
import pandas as pd
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from ..config.config import EvaluationConfig, ModelConfig
from ..models.model import BERTClassifier
from ..utils.metrics import calculate_metrics
from ..utils.train_utils import load_and_preprocess_data, create_dataloaders
from ..utils.logging_manager import setup_logger

logger = setup_logger(__name__)

class ModelEvaluator:
    """Dedicated class for model evaluation"""
    
    DEFAULT_METRICS = ["accuracy", "f1", "precision", "recall"]
    
    def __init__(self, model_path: Path, config: Union[EvaluationConfig, ModelConfig]):
        self.model_path = Path(model_path)
        self.config = config
        self.device = torch.device(config.device)
        self.metrics = getattr(config, 'metrics', self.DEFAULT_METRICS)
        self.model = self._load_model()
        
    def _load_model(self) -> BERTClassifier:
        """Load the trained model"""
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model configuration
            model_state = checkpoint.get('model_state_dict')
            model_config = checkpoint.get('config', {}).get('classifier_config', {})
            num_classes = checkpoint.get('num_classes', self.config.num_classes)
            
            if not model_state:
                raise ValueError("No model state found in checkpoint")
            
            # Ensure all required configuration keys exist
            default_config = {
                'architecture_type': 'standard',
                'num_layers': 2,
                'hidden_dim': 256,
                'activation': 'gelu',
                'regularization': 'dropout',
                'dropout_rate': 0.1,
                'cls_pooling': True,
                'learning_rate': self.config.learning_rate,
                'weight_decay': 0.01
            }
            
            # Update with saved config if available
            model_config = {**default_config, **model_config}
            
            # Create and load model
            model = BERTClassifier(
                self.config.bert_model_name,
                num_classes,
                model_config
            )
            model.load_state_dict(model_state)
            model.to(self.device)
            model.eval()
            
            logger.info("Model loaded successfully:")
            logger.info(f"Architecture: {model_config.get('architecture_type', 'standard')}")
            logger.info(f"Number of classes: {num_classes}")
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}") from e
    
    def evaluate(self, 
                save_predictions: bool = True,
                output_dir: Optional[Path] = None) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Evaluate model on test set
        
        Args:
            save_predictions: Whether to save predictions to file
            output_dir: Directory to save results (defaults to validation_results)
            
        Returns:
            metrics: Dictionary of evaluation metrics
            results_df: DataFrame with predictions and ground truth
        """
        # Load test data
        test_texts, test_labels, label_encoder = load_and_preprocess_data(
            self.config, validation_mode=True
        )
        
        # Create test dataloader
        test_dataloader = create_dataloaders(
            test_texts,
            test_labels,
            self.config,
            self.config.batch_size,
            validation_mode=True
        )
        
        # Evaluate
        all_preds = []
        all_labels = []
        all_probs = []
        
        logger.info("Starting evaluation...")
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label']
                
                # Get predictions
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Store results
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())
                all_probs.extend(probs.cpu().tolist())
        
        # Calculate metrics using the class metrics list
        metrics = calculate_metrics(all_labels, all_preds, self.metrics)
        
        # Create detailed results DataFrame
        results_df = pd.DataFrame({
            'text': test_texts,
            'true_label': label_encoder.inverse_transform(all_labels),
            'predicted_label': label_encoder.inverse_transform(all_preds),
            'confidence': [max(probs) for probs in all_probs]
        })
        
        # Save results if requested
        if save_predictions:
            output_dir = output_dir or Path("evaluation_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save predictions
            results_df.to_csv(output_dir / 'test_predictions.csv', index=False)
            
            # Save detailed classification report
            report = classification_report(
                all_labels, all_preds,
                target_names=label_encoder.classes_,
                output_dict=True
            )
            pd.DataFrame(report).to_csv(output_dir / 'classification_report.csv')
            
            # Save confusion matrix
            confusion_df = pd.crosstab(
                results_df['true_label'],
                results_df['predicted_label'],
                margins=True
            )
            confusion_df.to_csv(output_dir / 'confusion_matrix.csv')
            
            # Log results
            logger.info("\nEvaluation Results:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            logger.info("\nConfusion Matrix:")
            logger.info("\n" + str(confusion_df))
            
        return metrics, results_df

    @classmethod
    def from_config(cls, config: EvaluationConfig) -> 'ModelEvaluator':
        """Create evaluator instance from config"""
        return cls(
            model_path=config.best_model,
            config=config
        )

    @classmethod
    def add_model_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add model-specific arguments"""
        model_args = parser.add_argument_group('Model Configuration')
        model_args.add_argument('--device', type=str, default='cpu',
                              choices=['cpu', 'cuda'],
                              help='Device to use for inference')
        model_args.add_argument('--batch_size', type=int, default=32,
                              help='Batch size for evaluation')

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained BERT classifier')
    parser.add_argument('--best_model', type=Path, required=True,
                       help='Path to best model checkpoint')
    parser.add_argument('--output_dir', type=Path, default=Path('evaluation_results'),
                       help='Directory to save evaluation results')
    parser.add_argument('--metrics', nargs='+', type=str,
                       default=ModelEvaluator.DEFAULT_METRICS,
                       choices=ModelEvaluator.DEFAULT_METRICS,
                       help='Metrics to compute')
    ModelEvaluator.add_model_args(parser)
    
    args = parser.parse_args()
    
    config = EvaluationConfig()
    config.best_model = args.best_model
    config.output_dir = args.output_dir
    config.metrics = args.metrics
    config.device = args.device
    config.batch_size = args.batch_size
    
    try:
        logger.info(f"Loading model from: {config.best_model}")
        evaluator = ModelEvaluator.from_config(config)
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
