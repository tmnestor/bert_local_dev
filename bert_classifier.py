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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=Path, default='data/bbc-text.csv')
    parser.add_argument('--model_save_path', type=Path, default='bert_classifier.pth')
    parser.add_argument('--device', type=str, default='cpu')  # Change default to 'cpu'
    return parser.parse_args()

def load_data(config: ModelConfig):
    df = pd.read_csv(config.data_file)
    le = LabelEncoder()
    labels = le.fit_transform(df["category"])
    return df['text'].tolist(), labels.tolist(), le

def main():
    args = parse_args()
    config = ModelConfig(
        data_file=args.data_file,
        model_save_path=args.model_save_path,
        device=args.device
    )
    
    texts, labels, label_encoder = load_data(config)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Define default classifier configuration
    classifier_config = {
        'num_layers': 2,
        'activation': 'relu',
        'regularization': 'dropout',
        'dropout_rate': config.hidden_dropout
    }
    
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
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
    
    # Create progress bars
    epoch_pbar = tqdm(total=config.num_epochs, desc='Training', position=0)
    batch_pbar = tqdm(total=len(train_dataloader), desc='Epoch Progress', position=1, leave=False)
    
    try:
        for epoch in range(config.num_epochs):
            # Update epoch progress bar
            epoch_pbar.set_description(f'Epoch {epoch + 1}/{config.num_epochs}')
            
            # Reset batch progress bar
            batch_pbar.reset()
            
            # Training
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
                
                # Update batch progress
                batch_pbar.update(1)
            
            # Evaluation
            accuracy, report = trainer.evaluate(val_dataloader)
            
            # Update progress bar with metrics
            epoch_pbar.set_postfix({
                'accuracy': f'{accuracy:.4f}'
            })
            epoch_pbar.update(1)
            
            # Log detailed report less frequently
            if (epoch + 1) % 5 == 0:  # Log every 5 epochs
                logger.info(f"\nClassification Report:\n{report}")
    
    finally:
        # Clean up progress bars
        epoch_pbar.close()
        batch_pbar.close()
    
    torch.save(model.state_dict(), config.model_save_path)
    logger.info(f"Model saved to {config.model_save_path}")

if __name__ == "__main__":
    main()

