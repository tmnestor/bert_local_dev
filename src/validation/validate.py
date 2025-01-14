import argparse
from pathlib import Path
from typing import Dict, Optional
import torch
import pandas as pd
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from ..config.config import ValidationConfig
from ..models.model import BERTClassifier
from ..utils.metrics import calculate_metrics
from ..utils.logging_manager import setup_logger
from ..utils.train_utils import load_and_preprocess_data, create_dataloaders

logger = setup_logger(__name__)

def find_test_split(data_dir: Path) -> Optional[Path]:
    """Find test split in data directory"""
    test_split = data_dir / "test_split.csv"
    return test_split if test_split.exists() else None

def load_model(validation_config: ValidationConfig) -> BERTClassifier:
    """Load trained model from path"""
    logger.info(f"Loading model from: {validation_config.model_path}")
    
    try:
        checkpoint = torch.load(validation_config.model_path, map_location=validation_config.device, weights_only=False)
        
        # Handle different save formats
        model_state = checkpoint.get('model_state_dict') or checkpoint.get('model_state')
        model_config = checkpoint.get('config') or checkpoint.get('classifier_config')
        num_classes = checkpoint.get('num_classes', validation_config.num_classes)
        
        if not model_state or not model_config:
            raise ValueError("Invalid model checkpoint format")
            
        # Create and load model
        model = BERTClassifier(
            model_config.get('bert_model_name', validation_config.bert_model_name),
            num_classes,
            model_config
        )
        model.load_state_dict(model_state)
        model.to(validation_config.device)
        model.eval()
        
        logger.info("Model loaded successfully:")
        logger.info(f"Architecture: {model_config.get('architecture_type', 'standard')}")
        logger.info(f"Number of classes: {num_classes}")
        
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}") from e

def validate_model(validation_config: ValidationConfig) -> Dict[str, float]:
    """Run validation on test data and compute metrics"""
    # Load test data
    texts, labels, label_encoder = load_and_preprocess_data(validation_config, validation_mode=True)
    test_dataloader = create_dataloaders(
        texts, labels, validation_config, validation_config.batch_size, validation_mode=True
    )
    
    # Load model
    model = load_model(validation_config)
    
    results = []
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Validating"):
            outputs = model(
                input_ids=batch['input_ids'].to(validation_config.device),
                attention_mask=batch['attention_mask'].to(validation_config.device)
            )
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            results.extend([{
                'pred': label_encoder.inverse_transform([p])[0],
                'true': label_encoder.inverse_transform([l])[0],
                'confidence': torch.softmax(o, dim=0).max().item()
            } for p, l, o in zip(preds.cpu(), batch['label'].cpu(), outputs.cpu())])
    
    # Calculate metrics
    validation_metrics = calculate_metrics(
        [r['true'] for r in results],
        [r['pred'] for r in results],
        validation_config.metrics
    )
    
    # Save results if requested
    if validation_config.save_predictions:
        pd.DataFrame(results).to_csv(validation_config.output_dir / 'predictions.csv', index=False)
        pd.DataFrame([validation_metrics]).to_csv(validation_config.output_dir / 'metrics.csv', index=False)
        
        # Save detailed report
        report = classification_report(
            [r['true'] for r in results],
            [r['pred'] for r in results],
            output_dict=True
        )
        pd.DataFrame(report).to_csv(validation_config.output_dir / 'classification_report.csv')
    
    return validation_metrics

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Validate BERT classifier on test set',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    ValidationConfig.add_argparse_args(parser)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = ValidationConfig.from_args(args)
    
    try:
        logger.info("Starting validation...")
        
        if not (config.data_file.parent / "test_split.csv").exists():
            logger.error(
                "\nTest split not found! Please run training first:\n"
                "python -m src.training.train --data_file data/bbc-text.csv\n"
                "This will generate the train/val/test splits."
            )
            raise FileNotFoundError("Missing test split file")
            
        config.validate()  # This will handle path resolution
        
        logger.info(f"Using model: {config.model_path}")
        logger.info(f"Test file: {config.test_file}")
        logger.info(f"Output directory: {config.output_dir}")
        
        metrics = validate_model(config)
        
        logger.info("\nValidation Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
        if config.save_predictions:
            logger.info(f"\nPredictions saved to: {config.output_dir}")
            
    except Exception as e:
        logger.error("Error during validation: %s", str(e), exc_info=True)
        raise
