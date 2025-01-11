import optuna
import torch
from torch.utils.data import DataLoader
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # Add this import
import pandas as pd
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from functools import partial
from transformers import BertTokenizer, get_linear_schedule_with_warmup  # Add this import

from config import ModelConfig
from model import BERTClassifier
from dataset import TextClassificationDataset
from trainer import Trainer

logger = logging.getLogger(__name__)

def objective(trial, config: ModelConfig, texts, labels, trial_pbar=None, epoch_pbar=None):
    logger.info(f"\nStarting trial #{trial.number}")
    logger.info("Sampling hyperparameters...")
    
    # Define hyperparameters to optimize
    classifier_config = {
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'gelu', 'selu']),
        'regularization': trial.suggest_categorical('regularization', ['dropout', 'batchnorm']),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5) if trial.suggest_categorical('regularization', ['dropout', 'batchnorm']) == 'dropout' else 0.0
    }
    
    logger.info(f"Trial #{trial.number} hyperparameters:")
    for key, value in classifier_config.items():
        logger.info(f"  {key}: {value}")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets and dataloaders
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, config.max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Create model with trial hyperparameters
    model = BERTClassifier(config.bert_model_name, config.num_classes, classifier_config)
    trainer = Trainer(model, config)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # Training loop
    best_accuracy = 0.0
    if epoch_pbar is not None:
        epoch_pbar.reset()
        epoch_pbar.total = config.num_epochs
        
    if trial_pbar is not None:
        trial_pbar.set_description(f'Trial {trial.number}/{config.n_trials}')  # Use config.n_trials
        
    for epoch in range(config.num_epochs):
        if epoch_pbar is not None:
            epoch_pbar.set_description(f'Trial {trial.number} Epoch {epoch+1}/{config.num_epochs}')
        
        trainer.train_epoch(train_dataloader, optimizer, scheduler)
        accuracy, _ = trainer.evaluate(val_dataloader)
        
        if epoch_pbar is not None:
            epoch_pbar.update(1)
            epoch_pbar.set_postfix({
                'accuracy': f'{accuracy:.4f}',
                'trial': f'{trial.number}/{config.n_trials}'  # Use config.n_trials
            })
            
        trial.report(accuracy, epoch)
        if trial.should_prune():
            logger.info(f"Trial #{trial.number} pruned!")
            raise optuna.TrialPruned()
            
        best_accuracy = max(best_accuracy, accuracy)
    
    logger.info(f"Trial #{trial.number} finished with best accuracy: {best_accuracy:.4f}")
    
    if trial_pbar is not None:
        trial_pbar.update(1)
        
    return best_accuracy

def run_optimization(config: ModelConfig, n_trials: int = 100):
    config.n_trials = n_trials  # Set n_trials in config
    logger.info("\n" + "="*50)
    logger.info("Starting optimization")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Model config:")
    logger.info(f"  BERT model: {config.bert_model_name}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Max length: {config.max_length}")
    logger.info(f"  Num epochs: {config.num_epochs}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Data file: {config.data_file}")
    
    # Load data
    logger.info("\nLoading data...")
    df = pd.read_csv(config.data_file)
    logger.info(f"Loaded {len(df)} samples")
    
    le = LabelEncoder()
    texts = df['text'].tolist()
    labels = le.fit_transform(df["category"]).tolist()
    logger.info(f"Number of classes: {len(set(labels))}")
    logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # Create progress bars
    logger.info("\nInitializing progress bars...")
    trial_pbar = tqdm(total=n_trials, desc='Trials', position=0)
    epoch_pbar = tqdm(total=config.num_epochs, desc='Epochs', position=1, leave=True)
    
    # Create study
    logger.info("\nCreating Optuna study...")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Run optimization with progress bars
    objective_with_progress = partial(objective, 
                                    config=config, 
                                    texts=texts, 
                                    labels=labels,
                                    trial_pbar=trial_pbar,
                                    epoch_pbar=epoch_pbar)
    
    try:
        study.optimize(
            objective_with_progress,
            n_trials=n_trials,
            timeout=None,
            show_progress_bar=True
        )
    finally:
        # Clean up progress bars
        trial_pbar.close()
        epoch_pbar.close()
    
    logger.info("\n" + "="*50)
    logger.info("Optimization finished!")
    logger.info("Best trial:")
    trial = study.best_trial
    
    logger.info(f"  Value: {trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    
    return study.best_params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=100, help='Number of Optuna trials to run')
    parser.add_argument('--data_file', type=Path, default='data/bbc-text.csv')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

if __name__ == "__main__":
    # Set up logging to both file and console with more verbose settings
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,  # Force reconfiguration of the root logger
        handlers=[
            logging.StreamHandler(),  # Console handler first
            logging.FileHandler('optuna_optimization.log')  # File handler second
        ]
    )
    
    # Add console output for immediate feedback
    print("="*50)
    print("Starting Optuna optimization script")
    
    args = parse_args()
    print(f"Parsed arguments: n_trials={args.n_trials}, device={args.device}, data_file={args.data_file}")
    
    print("Initializing ModelConfig...")
    config = ModelConfig(data_file=args.data_file, device=args.device)
    
    print("Starting optimization run...")
    try:
        best_params = run_optimization(config, n_trials=args.n_trials)
        print("Optimization completed successfully")
        print(f"Best parameters found: {best_params}")
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        logger.error("Error during optimization", exc_info=True)
        raise
