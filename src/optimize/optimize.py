#!/usr/bin/env python

import os
from typing import Dict, List, Tuple, Optional, Any
import optuna
from optuna._experimental import ExperimentalWarning  # Import first
import warnings
# Silence specific Optuna warnings
warnings.filterwarnings('ignore', category=ExperimentalWarning)

import torch
import time
from datetime import datetime  # Add this import
from torch.utils.data import DataLoader
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from functools import partial
from transformers import (
    BertTokenizer,
    get_linear_schedule_with_warmup
)
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import SuccessiveHalvingPruner, HyperbandPruner

# Use relative imports for local modules
from ..config.config import ModelConfig
from ..models.model import BERTClassifier
from ..training.dataset import TextClassificationDataset
from ..training.trainer import Trainer

logger = logging.getLogger(__name__)

def load_data(config: ModelConfig) -> Tuple[List[str], List[int], LabelEncoder]:
    """Load and preprocess data from config.data_file
    
    Args:
        config: ModelConfig instance with data_file path
        
    Returns:
        Tuple containing:
        - List[str]: texts
        - List[int]: encoded labels
        - LabelEncoder: fitted label encoder
    """
    df = pd.read_csv(config.data_file)
    le = LabelEncoder()
    texts = df['text'].tolist()
    labels = le.fit_transform(df["category"]).tolist()
    return texts, labels, le

def initialize_progress_bars(n_trials: int, num_epochs: int) -> Tuple[tqdm, tqdm]:
    """Initialize progress bars for trials and epochs
    
    Args:
        n_trials: Total number of trials
        num_epochs: Number of epochs per trial
        
    Returns:
        Tuple containing:
        - trial progress bar
        - epoch progress bar
    """
    trial_pbar = tqdm(total=n_trials, desc='Trials', position=0)
    epoch_pbar = tqdm(total=num_epochs, desc='Epochs', position=1, leave=False)
    return trial_pbar, epoch_pbar

def _create_study(study_name: str, storage: Optional[str] = None, 
                sampler_type: str = 'tpe', seed: Optional[int] = None) -> optuna.Study:
    """Internal function to create an Optuna study."""
    if seed is None:
        seed = int(time.time())
        
    sampler = {
        'tpe': TPESampler(
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True,
            seed=seed
        ),
        'random': optuna.samplers.RandomSampler(seed=seed),
        'cmaes': optuna.samplers.CmaEsSampler(
            n_startup_trials=10,
            seed=seed
        ),
        'qmc': optuna.samplers.QMCSampler(
            qmc_type='sobol',
            seed=seed
        )
    }.get(sampler_type, TPESampler(seed=seed))

    pruner = HyperbandPruner(
        min_resource=1,
        max_resource=10,
        reduction_factor=3
    )

    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

def run_optimization(config: ModelConfig, timeout: Optional[int] = None, 
                    study_name: str = 'bert_optimization',
                    storage: Optional[str] = None,
                    seed: Optional[int] = None,
                    n_trials: Optional[int] = None) -> Dict[str, Any]:
    logger.info("\n" + "="*50)
    logger.info("Starting optimization")
    logger.info(f"Number of trials: {n_trials or config.n_trials}")
    logger.info(f"Model config:")
    logger.info(f"  BERT model: {config.bert_model_name}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Max length: {config.max_length}")
    logger.info(f"  Metric: {config.metric}")
    
    # Load data
    logger.info("\nLoading data...")
    texts, labels, _ = load_data(config)
    logger.info(f"Loaded {len(texts)} samples")
    logger.info(f"Number of classes: {len(set(labels))}")
    
    # Initialize progress bars
    trial_pbar, epoch_pbar = initialize_progress_bars(n_trials or config.n_trials, config.num_epochs)
    
    # Create study (fix the reference)
    study = _create_study(study_name, storage, config.sampler, seed)
    study.set_user_attr("best_value", 0.0)
    
    # Track best model info
    best_model_info = {}
    
    # Run optimization
    objective_with_progress = partial(objective, 
                                    config=config, 
                                    texts=texts, 
                                    labels=labels,
                                    study_name=study_name,
                                    best_model_info=best_model_info,
                                    trial_pbar=trial_pbar,
                                    epoch_pbar=epoch_pbar)
    
    try:
        study.optimize(
            objective_with_progress,
            n_trials=n_trials or config.n_trials,
            timeout=timeout,
            callbacks=[save_trial_callback(study_name, config)],
            gc_after_trial=True
        )
    finally:
        trial_pbar.close()
        epoch_pbar.close()
        
        if best_model_info:
            save_best_trial(best_model_info, study_name, config)
            
    return study.best_trial.params

def save_best_trial(best_model_info: Dict[str, Any], study_name: str, config: ModelConfig) -> None:
    config.best_trials_dir.mkdir(exist_ok=True, parents=True)
    metric_key = f'{config.metric}_score'
    final_model_path = config.best_trials_dir / f'best_model_{study_name}.pt'
    
    save_dict = {
        'model_state_dict': best_model_info['model_state'],
        'config': best_model_info['config'],
        'trial_number': best_model_info['trial_number'],
        metric_key: best_model_info[metric_key],
        'study_name': study_name,
        'hyperparameters': best_model_info['params'],
        'timestamp': datetime.now().isoformat()
    }
    torch.save(save_dict, final_model_path)
    logger.info(f"Saved best model to {final_model_path}")
    logger.info(f"Best {config.metric}: {best_model_info[metric_key]:.4f}")

def objective(trial: optuna.Trial, config: ModelConfig, texts: List[str], labels: List[int],
             study_name: str, best_model_info: Dict[str, Any], 
             trial_pbar: Optional[tqdm] = None, epoch_pbar: Optional[tqdm] = None) -> float:
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Define hyperparameters
    arch_type = trial.suggest_categorical('architecture_type', ['standard', 'plane_resnet'])
    classifier_config = {
        'architecture_type': arch_type,
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'cls_pooling': trial.suggest_categorical('cls_pooling', [True, False])
    }
    
    # Add architecture-specific parameters
    if arch_type == 'plane_resnet':
        classifier_config.update({
            'num_planes': trial.suggest_int('num_planes', 4, 16),
            'plane_width': trial.suggest_categorical('plane_width', [32, 64, 128, 256]),
            'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.2)
        })
    else:
        classifier_config.update({
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256, 512, 1024]),
            'activation': trial.suggest_categorical('activation', ['gelu', 'relu']),
            'regularization': trial.suggest_categorical('regularization', ['dropout', 'batchnorm']),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.2)
        })
    
    # Create dataloaders with explicit clean_up_tokenization_spaces
    tokenizer = BertTokenizer.from_pretrained(
        config.bert_model_name,
        clean_up_tokenization_spaces=True  # Explicitly set parameter
    )
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, config.max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=classifier_config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=classifier_config['batch_size'])
    
    # Create model and trainer
    model = BERTClassifier(config.bert_model_name, config.num_classes, classifier_config)
    trainer = Trainer(model, config)
    
    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=classifier_config['learning_rate'],
        weight_decay=classifier_config['weight_decay']
    )
    
    total_steps = len(train_dataloader) * config.num_epochs
    warmup_steps = int(total_steps * classifier_config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Training loop
    best_score = 0.0
    patience = max(5, trial.number // 5)
    no_improve_count = 0
    
    for epoch in range(config.num_epochs):
        if epoch_pbar:
            epoch_pbar.set_description(f"Trial {trial.number} Epoch {epoch+1}")
            
        trainer.train_epoch(train_dataloader, optimizer, scheduler)
        score, _ = trainer.evaluate(val_dataloader)
        
        if epoch_pbar:
            epoch_pbar.update(1)
            
        trial.report(score, epoch)
        
        if score > best_score:
            best_score = score
            no_improve_count = 0
            best_model_info.update({
                'model_state': model.state_dict(),
                'config': classifier_config.copy(),
                f'{config.metric}_score': score,
                'trial_number': trial.number,
                'params': trial.params
            })
        else:
            no_improve_count += 1
            
        if epoch >= min(5, config.num_epochs // 2):
            if trial.should_prune() or no_improve_count >= patience:
                raise optuna.TrialPruned()
    
    if trial_pbar:
        trial_pbar.update(1)
        
    return best_score

def save_trial_callback(study_name: str, config: ModelConfig):
    def callback(study: optuna.Study, trial: optuna.Trial):
        if trial.value and trial.value > study.user_attrs.get("best_value", 0.0):
            study.set_user_attr("best_value", trial.value)
            config.best_trials_dir.mkdir(exist_ok=True, parents=True)
            torch.save({
                'trial_number': trial.number,
                'params': trial.params,
                'value': trial.value,
                'study_name': study_name
            }, config.best_trials_dir / f'best_trial_{study_name}.pt')
    return callback

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='BERT Classifier Hyperparameter Optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    ModelConfig.add_argparse_args(parser)
    
    optim = parser.add_argument_group('Optimization')
    optim.add_argument('--timeout', type=int, default=None,
                      help='Optimization timeout in seconds')
    optim.add_argument('--study-name', type=str, default='bert_optimization',
                      help='Name for the Optuna study')
    optim.add_argument('--storage', type=str, default=None,
                      help='Database URL for Optuna storage')
    optim.add_argument('--seed', type=int, default=None,
                      help='Random seed for sampler')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    args = parse_args()
    config = ModelConfig.from_args(args)
    
    try:
        trials_per_exp = args.trials_per_experiment or args.n_trials
        for experiment_id in range(args.n_experiments):
            study_name = f"{args.study_name}_exp{experiment_id}"
            seed = args.seed + experiment_id if args.seed is not None else None
            
            best_params = run_optimization(
                config,
                timeout=args.timeout,
                study_name=study_name,
                storage=args.storage,
                seed=seed,
                n_trials=trials_per_exp
            )
            logger.info(f"Experiment {experiment_id + 1} completed")
            logger.info(f"Best parameters: {best_params}")
            
        logger.info("\nAll experiments completed successfully")
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}", exc_info=True)
        raise