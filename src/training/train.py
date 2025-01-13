#!/usr/bin/env python

import logging
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pandas as pd
import glob

from ..config.config import ModelConfig
from ..models.model import BERTClassifier
from .trainer import Trainer
from .dataset import TextClassificationDataset
from ..optimize.optimize import initialize_progress_bars  # Reuse progress bars

logger = logging.getLogger(__name__)

def load_best_configuration(best_trials_dir: Path, study_name: str = None) -> dict:
    """Load best model configuration from optimization results"""
    pattern = f"best_trial_{study_name if study_name else '*'}.pt"
    trial_files = list(best_trials_dir.glob(pattern))
    
    if not trial_files:
        logger.warning("No previous optimization results found")
        return None
        
    # Find the best performing trial
    best_trial = None
    best_value = float('-inf')
    
    for file in trial_files:
        trial_data = torch.load(file)
        if trial_data['value'] > best_value:
            best_value = trial_data['value']
            best_trial = trial_data
            
    if best_trial:
        logger.info(f"Loaded best configuration from {file}")
        logger.info(f"Best trial score: {best_value:.4f}")
        return best_trial['params']
    return None

def train_model(config: ModelConfig, classifier_config: dict = None):
    """Train a model with fixed or optimized configuration"""
    # Try to load best configuration if none provided
    if classifier_config is None:
        best_config = load_best_configuration(config.best_trials_dir)
        if best_config:
            logger.info("\nUsing best configuration from previous optimization:")
            logger.info(f"Architecture: {best_config.get('architecture_type', 'standard')}")
            logger.info(f"Learning rate: {best_config.get('learning_rate', config.learning_rate)}")
            logger.info(f"Weight decay: {best_config.get('weight_decay', 0.01)}")
            
            classifier_config = {
                'architecture_type': best_config.get('architecture_type', 'standard'),
                'learning_rate': best_config.get('learning_rate', config.learning_rate),
                'weight_decay': best_config.get('weight_decay', 0.01),
                'cls_pooling': best_config.get('cls_pooling', True),
            }
            
            # Add architecture-specific parameters
            if classifier_config['architecture_type'] == 'plane_resnet':
                logger.info(f"Number of planes: {best_config['num_planes']}")
                logger.info(f"Plane width: {best_config['plane_width']}")
                classifier_config.update({
                    'num_planes': best_config['num_planes'],
                    'plane_width': best_config['plane_width']
                })
            else:
                logger.info(f"Number of layers: {best_config.get('num_layers', 2)}")
                logger.info(f"Hidden dimension: {best_config.get('hidden_dim', 256)}")
                logger.info(f"Activation: {best_config.get('activation', 'gelu')}")
                logger.info(f"Regularization: {best_config.get('regularization', 'dropout')}")
                classifier_config.update({
                    'num_layers': best_config.get('num_layers', 2),
                    'hidden_dim': best_config.get('hidden_dim', 256),
                    'activation': best_config.get('activation', 'gelu'),
                    'regularization': best_config.get('regularization', 'dropout'),
                    'dropout_rate': best_config.get('dropout_rate', 0.1)
                })
        else:
            logger.info("\nNo previous optimization found. Using default configuration:")
            logger.info("Architecture: standard")
            logger.info("Number of layers: 2")
            logger.info("Hidden dimension: 256")
            logger.info("Activation: gelu")
            logger.info("Regularization: dropout")
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
        logger.info("\nUsing provided configuration:")
        logger.info(f"Architecture: {classifier_config['architecture_type']}")
        if classifier_config['architecture_type'] == 'plane_resnet':
            logger.info(f"Number of planes: {classifier_config['num_planes']}")
            logger.info(f"Plane width: {classifier_config['plane_width']}")
        else:
            logger.info(f"Number of layers: {classifier_config['num_layers']}")
            logger.info(f"Hidden dimension: {classifier_config['hidden_dim']}")
            logger.info(f"Activation: {classifier_config['activation']}")
            logger.info(f"Regularization: {classifier_config['regularization']}")
    
    # Load and preprocess data
    df = pd.read_csv(config.data_file)
    texts = df['text'].tolist()
    labels = pd.Categorical(df['category']).codes.tolist()
    logger.info(f"Loaded {len(texts)} samples with {len(set(labels))} classes")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer with explicit clean_up_tokenization_spaces
    tokenizer = BertTokenizer.from_pretrained(
        config.bert_model_name,
        clean_up_tokenization_spaces=True  # Explicitly set to avoid warning
    )
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, config.max_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Initialize model and trainer
    model = BERTClassifier(config.bert_model_name, config.num_classes, classifier_config)
    trainer = Trainer(model, config)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=classifier_config.get('learning_rate', config.learning_rate),
        weight_decay=classifier_config.get('weight_decay', 0.01)
    )
    
    # Setup warmup scheduler
    total_steps = len(train_dataloader) * config.num_epochs
    warmup_steps = int(total_steps * 0.1)  # 10% warmup by default
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize progress bars
    epoch_pbar, batch_pbar = initialize_progress_bars(config.num_epochs, len(train_dataloader))
    
    # Training loop
    best_score = 0.0
    try:
        for epoch in range(config.num_epochs):
            epoch_pbar.set_description(f"Epoch {epoch+1}/{config.num_epochs}")
            
            # Train with scheduler
            loss = trainer.train_epoch(train_dataloader, optimizer, scheduler, batch_pbar)
            
            # Evaluate
            score, _ = trainer.evaluate(val_dataloader)
            
            # Update progress bar with metrics
            epoch_pbar.set_postfix({
                'loss': f'{loss:.4f}',
                f'val_{config.metric}': f'{score:.4f}'
            })
            epoch_pbar.update(1)
            
            # Save best model
            if score > best_score:
                best_score = score
                trainer.save_checkpoint(
                    config.model_save_path,
                    epoch,
                    optimizer
                )
                logger.info(f"\nSaved new best model with {config.metric}={score:.4f}")
    
    finally:
        epoch_pbar.close()
        batch_pbar.close()
        
    logger.info(f"\nTraining completed. Best {config.metric}: {best_score:.4f}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train BERT Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    ModelConfig.add_argparse_args(parser)
    
    # Add architecture type argument
    parser.add_argument('--architecture', type=str, default='standard',
                       choices=['standard', 'plane_resnet'],
                       help='Classifier architecture to use')
    
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    args = parse_args()
    config = ModelConfig.from_args(args)
    
    # Define architecture config
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
    
    train_model(config, classifier_config)
