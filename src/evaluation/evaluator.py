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
        
    def _load_model(self) -> BERTClassifier:
        """Load the trained model from checkpoint.
        
        Returns:
            BERTClassifier: Loaded and configured model instance.
            
        Raises:
            RuntimeError: If model loading fails.
        """
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            logger.info("Loaded checkpoint with keys: %s", list(checkpoint.keys()))
            
            # Extract model state and config consistently
            model_state = checkpoint['model_state_dict']
            config_container = checkpoint.get('config', {})
            classifier_config = config_container.get('classifier_config', {})
            num_classes = checkpoint.get('num_classes', self.config.num_classes)
            
            # Log source and score
            if 'study_name' in checkpoint:
                logger.info("Loading model from optimization trial")
                logger.info("Study: %s, Trial: %s, Score: %.4f", 
                           checkpoint.get('study_name'),
                           checkpoint.get('trial_number'),
                           checkpoint.get('metric_value', float('nan')))
            
            # Create and load model
            model = BERTClassifier(
                self.config.bert_model_name,
                num_classes,
                classifier_config
            )
            model.load_state_dict(model_state)
            model.to(self.device)
            model.eval()
            
            # Enhanced logging of model architecture
            logger.info("\nModel Architecture Details:")
            logger.info("-" * 50)
            logger.info(f"BERT Model: {self.config.bert_model_name}")
            logger.info(f"Architecture Type: {classifier_config['architecture_type']}")
            logger.info(f"Number of Classes: {num_classes}")
            logger.info(f"CLS Pooling: {classifier_config['cls_pooling']}")
            
            if classifier_config['architecture_type'] == 'standard':
                logger.info("\nStandard Classifier Configuration:")
                logger.info(f"Number of Layers: {classifier_config['num_layers']}")
                logger.info(f"Hidden Dimension: {classifier_config['hidden_dim']}")
                logger.info(f"Activation: {classifier_config['activation']}")
                logger.info(f"Regularization: {classifier_config['regularization']}")
                if classifier_config['regularization'] == 'dropout':
                    logger.info(f"Dropout Rate: {classifier_config['dropout_rate']}")
                
                # Log layer sizes
                input_size = model.bert.config.hidden_size
                logger.info("\nLayer Dimensions:")
                logger.info(f"Input (BERT) -> {input_size}")
                
                # Calculate and log progression of layer sizes
                current_size = input_size
                if classifier_config['num_layers'] > 1:
                    ratio = (classifier_config['hidden_dim'] / current_size) ** (1.0 / (classifier_config['num_layers'] - 1))
                    for i in range(classifier_config['num_layers'] - 1):
                        current_size = int(current_size * ratio)
                        current_size = max(current_size, classifier_config['hidden_dim'])
                        logger.info(f"Hidden Layer {i+1} -> {current_size}")
                logger.info(f"Output Layer -> {num_classes}")
                
            else:  # plane_resnet
                logger.info("\nPlaneResNet Configuration:")
                logger.info(f"Number of Planes: {classifier_config['num_planes']}")
                logger.info(f"Plane Width: {classifier_config['plane_width']}")
                logger.info(f"Input Size: {model.bert.config.hidden_size}")
                logger.info(f"Output Size: {num_classes}")
            
            logger.info("-" * 50)
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}") from e
    
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

def main():
    """Command-line interface entry point for model evaluation.
    
    Raises:
        Exception: If evaluation fails.
    """
    parser = argparse.ArgumentParser(description='Evaluate trained BERT classifier')
    
    # Add all ModelConfig arguments first
    ModelConfig.add_argparse_args(parser)
    
    # Add evaluator-specific arguments
    parser.add_argument('--best_model', type=Path, required=True,
                       help='Path to best model checkpoint')
    parser.add_argument('--output_dir', type=Path, default=Path('evaluation_results'),
                       help='Directory to save evaluation results')
    parser.add_argument('--metrics', nargs='+', type=str,
                       default=ModelEvaluator.DEFAULT_METRICS,
                       choices=ModelEvaluator.DEFAULT_METRICS,
                       help='Metrics to compute')
    
    args = parser.parse_args()
    
    # Create full config from all parsed args
    config = ModelConfig.from_args(args)
    
    # Update with evaluation-specific settings
    config.best_model = args.best_model
    config.output_dir = args.output_dir
    config.metrics = args.metrics
    
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
