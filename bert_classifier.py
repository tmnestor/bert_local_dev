#!/usr/bin/env python

import os
import logging
import argparse
from pathlib import Path
import torch
from torch import nn  # Add this import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW  # Add this import
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import ModelConfig
from dataset import TextClassificationDataset
from model import BERTClassifier
from trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='BERT Classifier Training Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add all configuration options
    ModelConfig.add_argparse_args(parser)
    
    # Add any script-specific arguments here
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error("CUDA device requested but CUDA is not available")
    
    # Validate paths
    if not args.data_file.exists():
        parser.error(f"Data file not found: {args.data_file}")
    
    # Validate numeric arguments
    if args.num_epochs < 1:
        parser.error("num_epochs must be positive")
    if args.batch_size < 1:
        parser.error("batch_size must be positive")
    if not 0 < args.learning_rate < 1:
        parser.error("learning_rate must be between 0 and 1")
    if not 0 <= args.hidden_dropout <= 1:
        parser.error("hidden_dropout must be between 0 and 1")
        
    return args

def load_data(config: ModelConfig):
    df = pd.read_csv(config.data_file)
    le = LabelEncoder()
    labels = le.fit_transform(df["category"])
    return df['text'].tolist(), labels.tolist(), le

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, config, trainer):
    epoch_pbar = tqdm(total=config.num_epochs, desc='Training', position=0)
    batch_pbar = tqdm(total=len(train_dataloader), desc='Epoch Progress', position=1, leave=False)
    
    try:
        for epoch in range(config.num_epochs):
            epoch_pbar.set_description(f'Epoch {epoch + 1}/{config.num_epochs}')
            batch_pbar.reset()
            
            model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch['input_ids'].to(config.device),
                    attention_mask=batch['attention_mask'].to(config.device)
                )
                loss = nn.CrossEntropyLoss()(outputs, batch['label'].to(config.device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                batch_pbar.update(1)
            
            accuracy, report = trainer.evaluate(val_dataloader)
            epoch_pbar.set_postfix({'accuracy': f'{accuracy:.4f}'})
            epoch_pbar.update(1)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"\nClassification Report:\n{report}")
    finally:
        epoch_pbar.close()
        batch_pbar.close()

def main() -> None:
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create config from args
    config = ModelConfig.from_args(args)
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    texts, labels, label_encoder = load_data(config)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    classifier_config = {
        'num_layers': 2,
        'activation': 'relu',
        'regularization': 'dropout',
        'dropout_rate': config.hidden_dropout,
        'cls_pooling': True
    }
    
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name, clean_up_tokenization_spaces=True)
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, config.max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    model = BERTClassifier(
        bert_model_name=config.bert_model_name,
        num_classes=config.num_classes,
        classifier_config=classifier_config
    )
    trainer = Trainer(model, config)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, config, trainer)
    
    torch.save(model.state_dict(), config.model_save_path)
    logger.info(f"Model saved to {config.model_save_path}")

if __name__ == "__main__":
    main()

