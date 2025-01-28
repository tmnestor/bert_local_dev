#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Optional
import torch
from transformers import get_linear_schedule_with_warmup

from ..config.config import ModelConfig
from ..models.model import BERTClassifier
from .trainer import Trainer
from ..utils.train_utils import (
    load_and_preprocess_data,
    create_dataloaders,
    initialize_progress_bars,
    save_model_state
)
from ..utils.logging_manager import setup_logger
from ..config.defaults import CLASSIFIER_DEFAULTS
from ..tuning.optimize import create_optimizer  # Update this import

logger = setup_logger(__name__)

def _log_best_configuration(best_file: Path, best_value: float) -> None:
    """Log information about the best configuration"""
    logger.info("Loaded best configuration from %s", best_file)
    logger.info("Best trial score: %.4f", best_value)

def load_best_configuration(best_trials_dir: Path, study_name: str = None) -> Optional[dict]:
    """Load best model configuration from optimization results"""
    pattern = f"best_trial_{study_name or '*'}.pt"
    trial_files = list(best_trials_dir.glob(pattern))

    if not trial_files:
        logger.warning("No previous optimization results found")
        return None

    # Find the best performing trial
    best_trial = None
    best_value = float('-inf')
    best_file = None

    for file in trial_files:
        try:
            trial_data = torch.load(file, map_location='cpu', weights_only=False)
            if trial_data['value'] > best_value:
                best_value = trial_data['value']
                best_trial = trial_data
                best_file = file
        except Exception as e:
            logger.warning(f"Could not load trial file {file}: {e}")
            continue

    if best_trial:
        _log_best_configuration(best_file, best_value)
        # Return the actual configuration
        if 'params' in best_trial:
            config = best_trial['params']
            # Add default values for missing configuration keys
            arch_type = config.get('architecture_type', 'standard')
            
            if arch_type == 'standard':
                config.update({
                    'num_layers': config.get('std/num_layers', 2),
                    'hidden_dim': config.get('std/hidden_dim', 256),
                    'activation': config.get('std/activation', 'gelu'),
                    'regularization': config.get('std/regularization', 'dropout'),
                    'dropout_rate': config.get('std/dropout_rate', 0.1),
                    'cls_pooling': config.get('cls_pooling', True),
                })
            else:  # plane_resnet
                config.update({
                    'architecture_type': 'plane_resnet',
                    'num_planes': config.get('plane/num_planes', 8),
                    'plane_width': config.get('plane/width', 128),
                    'cls_pooling': config.get('cls_pooling', True),
                })
            return config

    logger.info("\nNo previous optimization found. Using default configuration")
    return None

def train_model(model_config: ModelConfig, clf_config: dict = None):
    """Train a model with fixed or optimized configuration"""
    if clf_config is None:
        # Use default configuration
        clf_config = CLASSIFIER_DEFAULTS['standard'].copy()
        clf_config.update({
            'learning_rate': model_config.learning_rate,
            'batch_size': model_config.batch_size
        })
        
    if clf_config is None:
        logger.info("Using default configuration")
        clf_config = {
            'architecture_type': 'standard',
            'num_layers': 2,
            'hidden_dim': 256,
            'learning_rate': model_config.learning_rate,
            'weight_decay': 0.01,
            'activation': 'gelu',
            'regularization': 'dropout',
            'dropout_rate': 0.1,
            'cls_pooling': True,
            'batch_size': model_config.batch_size
        }
    
    # Load and preprocess data using utility function with train/val split
    train_texts, val_texts, train_labels, val_labels, label_encoder = load_and_preprocess_data(model_config)
    logger.info("Loaded %d training and %d validation samples", 
                len(train_texts), len(val_texts))

    # Create dataloaders using utility function
    train_dataloader, val_dataloader = create_dataloaders(
        [train_texts, val_texts], 
        [train_labels, val_labels],
        model_config, 
        model_config.batch_size
    )
    
    # Initialize model and trainer
    model = BERTClassifier(model_config.bert_model_name, model_config.num_classes, clf_config)
    trainer = Trainer(model, model_config)
    
    # Use optimizer factory if optimizer is specified in config
    optimizer_name = clf_config.get('optimizer', 'adamw')
    optimizer_params = {
        'learning_rate': clf_config.get('learning_rate', model_config.learning_rate),
        'weight_decay': clf_config.get('weight_decay', 0.01),
        'beta1': clf_config.get('beta1', 0.9),
        'beta2': clf_config.get('beta2', 0.999)
    }
    
    optimizer = create_optimizer(
        optimizer_name,
        model.parameters(),
        **optimizer_params
    )
    
    total_steps = len(train_dataloader) * model_config.num_epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize progress bars using utility function
    epoch_pbar, batch_pbar = initialize_progress_bars(model_config.num_epochs, len(train_dataloader))
    
    # Training loop
    best_score = 0.0
    try:
        for epoch in range(model_config.num_epochs):
            epoch_pbar.set_description(f"Epoch {epoch+1}/{model_config.num_epochs}")
            
            loss = trainer.train_epoch(train_dataloader, optimizer, scheduler, batch_pbar)
            score, _ = trainer.evaluate(val_dataloader)
            
            epoch_pbar.set_postfix({
                'loss': f'{loss:.4f}',
                f'val_{model_config.metric}': f'{score:.4f}'
            })
            epoch_pbar.update(1)
            
            if score > best_score:
                best_score = score
                # Save model when new best score is achieved
                save_model_state(
                    model.state_dict(),
                    model_config.model_save_path,  # This is "best_trials/bert_classifier.pth"
                    score,
                    {
                        'epoch': epoch,
                        'optimizer_state': optimizer.state_dict(),
                        'classifier_config': clf_config,
                        'num_classes': model_config.num_classes
                    }
                )
                logger.info("\nSaved new best model with %s=%.4f", model_config.metric, score)
    
    finally:
        epoch_pbar.close()
        batch_pbar.close()
        
    logger.info("\nTraining completed. Best %s: %.4f", model_config.metric, best_score)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train BERT Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add all ModelConfig arguments including num_epochs
    ModelConfig.add_argparse_args(parser)
    
    # Add architecture type argument
    parser.add_argument('--architecture', type=str, default=None,
                       choices=['standard', 'plane_resnet'],
                       help='Classifier architecture to use. If not specified, uses best configuration from optimization.')
    
    args = parser.parse_args()
    
    # Validate num_epochs is positive
    if args.num_epochs < 1:
        parser.error("num_epochs must be positive")
        
    return args

if __name__ == "__main__":
    # Remove logging.basicConfig as it's handled by logging_manager
    args = parse_args()
    config = ModelConfig.from_args(args)
    
    # Only create classifier_config if architecture is explicitly specified
    if args.architecture:
        logger.info("\nUsing explicitly specified architecture configuration")
        if args.architecture == 'plane_resnet':
            classifier_config = {
                'architecture_type': 'plane_resnet',
                'num_planes': 8,
                'plane_width': 128,
                'cls_pooling': True
            }
        else:
            classifier_config = {
                'architecture_type': 'standard',
                'num_layers': 2,
                'hidden_dim': 256,
                'activation': 'gelu',
                'regularization': 'dropout',
                'dropout_rate': 0.1,
                'cls_pooling': True
            }
    else:
        classifier_config = None  # Let train_model try to load best configuration
    
    train_model(config, classifier_config)
