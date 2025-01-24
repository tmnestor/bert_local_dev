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
    save_model_state,
    load_best_trial_info  # Add this import
)
from ..utils.logging_manager import setup_logger

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

from src.models.factory import ModelFactory
from src.models.encoders import BERTEncoder
from src.models.heads import StandardClassifierHead, PlaneResNetHead
from ..utils.progress_manager import ProgressManager
from ..utils.model_utils import format_model_info, count_parameters

def train_model(model_config: ModelConfig, clf_config: dict = None, disable_pbar: bool = False):
    """Train a model with fixed or optimized configuration"""
    if clf_config is None:
        # Try to load best trial info first
        trial_info = load_best_trial_info(model_config.best_trials_dir, model_config.study_name)
        if (trial_info and 'params' in trial_info):
            clf_config = trial_info['params']
            logger.info("Using configuration from best trial")
            # Set architecture type in model_config immediately
            arch_type = clf_config.get('architecture_type', 'standard').lower()
            model_config.architecture = arch_type
            logger.info(f"Setting architecture type to: {arch_type}")
            
            # Load model weights if available
            model_path = trial_info['model_path']
            if Path(model_path).exists():
                logger.info("Found best trial model weights")
                model_config.pretrained_weights = model_path
        else:
            clf_config = load_best_configuration(model_config.best_trials_dir)
        
    if clf_config is None:
        logger.info("Using default configuration")
        clf_config = {
            'architecture_type': 'standard',  # Ensure architecture type is set
            'num_layers': 2,
            'hidden_dim': 256,
            'learning_rate': model_config.learning_rate,
            'weight_decay': model_config.weight_decay,
            'activation': 'gelu',
            'regularization': 'dropout',
            'dropout_rate': 0.1,
            'cls_pooling': True,
            'batch_size': model_config.batch_size
        }
        # Set the architecture in model_config to match clf_config
        model_config.architecture = clf_config['architecture_type']
    
    # Initialize config parameters early with guaranteed defaults
    training_params = {
        'learning_rate': clf_config.get('learning_rate', model_config.learning_rate or 2e-5),
        'weight_decay': 0.01,  # Set default first
        'batch_size': max(2, clf_config.get('batch_size', model_config.batch_size or 16)),  # Ensure minimum batch size
        'gradient_accumulation_steps': clf_config.get('gradient_accumulation_steps', 1),
        'max_grad_norm': clf_config.get('max_grad_norm', 1.0)
    }
    
    # Then override with config values if they exist
    if 'weight_decay' in clf_config and clf_config['weight_decay'] is not None:
        training_params['weight_decay'] = clf_config['weight_decay']
    elif model_config.weight_decay is not None:
        training_params['weight_decay'] = model_config.weight_decay
    
    # Ensure training params are set on model_config
    for key, value in training_params.items():
        setattr(model_config, key, value)
    
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
    
    # Ensure all training parameters are set
    model_config.weight_decay = clf_config.get('weight_decay', model_config.weight_decay)
    
    # Calculate and set warmup steps before creating trainer
    total_steps = len(train_dataloader) * model_config.num_epochs
    model_config.warmup_steps = int(0.1 * total_steps)  # 10% of total steps
    
    # Create model using factory
    model_params = {
        'bert_model_name': model_config.bert_model_name,
        'num_classes': model_config.num_classes,
        'architecture_type': model_config.architecture,
        'cls_pooling': True,
        **clf_config
    }
    
    try:
        model = ModelFactory.create_model(model_params)
        model = model.to(model_config.device)
        
        # Load pretrained weights if available
        if model_config.pretrained_weights and Path(model_config.pretrained_weights).exists():
            logger.info(f"Loading pretrained weights from {model_config.pretrained_weights}")
            checkpoint = torch.load(model_config.pretrained_weights, map_location=model_config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore training mode if saved
            if 'training' in checkpoint and checkpoint['training']:
                model.train()
            else:
                model.eval()
        
        # Log detailed model information
        logger.info("\n%s", format_model_info(model_params, model_params))
        logger.info("%s", count_parameters(model))
        
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise
    
    # Initialize a trainer with progress manager
    trainer = Trainer(model, model_config, disable_pbar=disable_pbar)
    progress = ProgressManager(disable=disable_pbar)
    
    # Setup optimizer with guaranteed parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_params['learning_rate'],
        weight_decay=training_params['weight_decay']
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=model_config.warmup_steps,  # Use the pre-calculated warmup steps
        num_training_steps=total_steps
    )
    
    # Setup training loop with progress tracking
    best_score = 0.0
    try:
        epoch_bar = progress.init_epoch_bar(model_config.num_epochs, mode='train')
        
        for epoch in range(model_config.num_epochs):
            # Train with progress updates
            loss = trainer.train_epoch(train_dataloader, optimizer, scheduler)
            score, _ = trainer.evaluate(val_dataloader)
            
            # Update progress
            progress.update_epoch(
                epoch,
                model_config.num_epochs,
                {'loss': f'{loss:.4f}', 'score': f'{score:.4f}'}
            )
            
            if score > best_score:
                best_score = score
                # Save model when new best score is achieved
                save_model_state(
                    model.state_dict(),
                    model_config.model_save_path,
                    score,
                    {
                        'epoch': epoch,
                        'optimizer_state': optimizer.state_dict(),
                        'classifier_config': clf_config,
                        'num_classes': model_config.num_classes
                    }
                )
                # Clear line and print improvement message
                print('\r', end='')  # Clear current line
                logger.info(f"[Epoch {epoch+1}/{model_config.num_epochs}] New best {model_config.metric}: {score:.4f}")
    
    finally:
        progress.close_all()
        print('\r', end='')  # Clear last progress bar line
        
    # Final summary on clean line
    logger.info(f"\nTraining completed. Best {model_config.metric}: {best_score:.4f}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train BERT Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add all ModelConfig arguments including num_epochs and study_name
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
    args = parse_args()
    config = ModelConfig.from_args(args)
    # study_name is now set from args in from_args()
    
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
