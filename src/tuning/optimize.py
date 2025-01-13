#!/usr/bin/env python

from typing import Dict, List, Optional, Any
import optuna
from optuna._experimental import ExperimentalWarning
import warnings
# Silence specific Optuna warnings
warnings.filterwarnings('ignore', category=ExperimentalWarning)

import torch
import time
from datetime import datetime
import logging
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from functools import partial
from transformers import get_linear_schedule_with_warmup
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from ..config.config import ModelConfig
from ..models.model import BERTClassifier
from ..training.trainer import Trainer
from ..utils.train_utils import (
    load_and_preprocess_data,
    create_dataloaders,
    initialize_progress_bars,
    log_separator  # Add this import
)
from ..utils.logging_manager import setup_logger

logger = setup_logger(__name__)

def _create_study(name: str, storage: Optional[str] = None, 
                sampler_type: str = 'tpe', random_seed: Optional[int] = None) -> optuna.Study:
    """Internal function to create an Optuna study."""
    if random_seed is None:
        random_seed = int(time.time())
        
    sampler = {
        'tpe': TPESampler(
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True,
            seed=random_seed
        ),
        'random': optuna.samplers.RandomSampler(seed=random_seed),
        'cmaes': optuna.samplers.CmaEsSampler(
            n_startup_trials=10,
            seed=random_seed
        ),
        'qmc': optuna.samplers.QMCSampler(
            qmc_type='sobol',
            seed=random_seed
        )
    }.get(sampler_type, TPESampler(seed=random_seed))

    pruner = HyperbandPruner(
        min_resource=1,
        max_resource=10,
        reduction_factor=3
    )

    return optuna.create_study(
        study_name=name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

def run_optimization(model_config: ModelConfig, timeout: Optional[int] = None, 
                    experiment_name: str = 'bert_optimization',
                    storage: Optional[str] = None,
                    random_seed: Optional[int] = None,
                    n_trials: Optional[int] = None) -> Dict[str, Any]:
    log_separator(logger)  # Replace logger.info("\n" + "="*50)
    logger.info("Starting optimization")
    logger.info("Number of trials: %s", n_trials or model_config.n_trials)
    
    # Load data using utility function
    logger.info("\nLoading data...")
    texts, labels, _ = load_and_preprocess_data(model_config)
    logger.info("Loaded %d samples with %d classes", len(texts), model_config.num_classes)
    
    logger.info("Model config:")
    logger.info("  BERT model: %s", model_config.bert_model_name)
    logger.info("  Device: %s", model_config.device)
    logger.info("  Max length: %s", model_config.max_length)
    logger.info("  Metric: %s", model_config.metric)
    
    # Initialize progress bars using utility function
    trial_pbar, epoch_pbar = initialize_progress_bars(n_trials or model_config.n_trials, model_config.num_epochs)
    
    # Create study (fix the reference)
    study = _create_study(experiment_name, storage, model_config.sampler, random_seed)
    study.set_user_attr("best_value", 0.0)
    
    # Track best model info
    best_model_info = {}
    
    # Run optimization
    objective_with_progress = partial(objective, 
                                    model_config=model_config,  # Changed from config to model_config
                                    texts=texts, 
                                    labels=labels,
                                    best_model_info=best_model_info,
                                    trial_pbar=trial_pbar,
                                    epoch_pbar=epoch_pbar)
    
    try:
        study.optimize(
            objective_with_progress,
            n_trials=n_trials or model_config.n_trials,
            timeout=timeout,
            callbacks=[save_trial_callback(experiment_name, model_config)],
            gc_after_trial=True
        )
    finally:
        trial_pbar.close()
        epoch_pbar.close()
        
        if best_model_info:
            save_best_trial(best_model_info, experiment_name, model_config)
            
    return study.best_trial.params

def save_best_trial(best_model_info: Dict[str, Any], trial_study_name: str, model_config: ModelConfig) -> None:
    model_config.best_trials_dir.mkdir(exist_ok=True, parents=True)
    metric_key = f'{model_config.metric}_score'
    final_model_path = model_config.best_trials_dir / f'best_model_{trial_study_name}.pt'
    
    save_dict = {
        'model_state_dict': best_model_info['model_state'],
        'config': best_model_info['config'],
        'trial_number': best_model_info['trial_number'],
        metric_key: best_model_info[metric_key],
        'study_name': trial_study_name,
        'hyperparameters': best_model_info['params'],
        'timestamp': datetime.now().isoformat(),
        'num_classes': model_config.num_classes  # Add num_classes to saved info
    }
    torch.save(save_dict, final_model_path)
    logger.info("Saved best model to %s", final_model_path)
    logger.info("Best %s: %.4f", model_config.metric, best_model_info[metric_key])

def objective(trial: optuna.Trial, model_config: ModelConfig, texts: List[str], labels: List[int],
             best_model_info: Dict[str, Any], 
             trial_pbar: Optional[tqdm] = None, epoch_pbar: Optional[tqdm] = None) -> float:
    train_dataloader, val_dataloader = create_dataloaders(
        texts, 
        labels, 
        model_config,  # Changed from config to model_config
        trial.suggest_categorical('batch_size', [16, 32, 64])
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
    
    # Create model and trainer
    model = BERTClassifier(model_config.bert_model_name, model_config.num_classes, classifier_config)
    trainer = Trainer(model, model_config)
    
    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=classifier_config['learning_rate'],
        weight_decay=classifier_config['weight_decay']
    )
    
    total_steps = len(train_dataloader) * model_config.num_epochs
    warmup_steps = int(total_steps * classifier_config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Training loop
    best_score = 0.0
    patience = max(5, trial.number // 5)
    no_improve_count = 0
    
    for epoch in range(model_config.num_epochs):
        if epoch_pbar:
            epoch_pbar.set_description(f"Trial {trial.number} Epoch {epoch+1}")
            
        trainer.train_epoch(train_dataloader, optimizer, scheduler)
        score, _ = trainer.evaluate(val_dataloader)
        
        if epoch_pbar:
            epoch_pbar.update(1)
            
        trial.report(score, epoch)
        
        if score > best_score:
            no_improve_count = 0
            best_score = score
            best_model_info.update({
                'model_state': model.state_dict(),
                'config': classifier_config.copy(),
                f'{model_config.metric}_score': best_score,
                'trial_number': trial.number,
                'params': trial.params
            })
        else:
            no_improve_count += 1
            
        if epoch >= min(5, model_config.num_epochs // 2) and (trial.should_prune() or no_improve_count >= patience):
            raise optuna.TrialPruned()
    
    if trial_pbar:
        trial_pbar.update(1)
        
    return best_score

def save_trial_callback(trial_study_name: str, model_config: ModelConfig):
    def callback(study: optuna.Study, trial: optuna.Trial):
        if trial.value and trial.value > study.user_attrs.get("best_value", 0.0):
            study.set_user_attr("best_value", trial.value)
            model_config.best_trials_dir.mkdir(exist_ok=True, parents=True)
            torch.save({
                'trial_number': trial.number,
                'params': trial.params,
                'value': trial.value,
                'study_name': trial_study_name
            }, model_config.best_trials_dir / f'best_trial_{trial_study_name}.pt')
    return callback

def load_best_configuration(best_trials_dir: Path, exp_name: str = None) -> dict:
    """Load best model configuration from optimization results"""
    pattern = f"best_trial_{exp_name or '*'}.pt"
    trial_files = list(best_trials_dir.glob(pattern))
    
    if not trial_files:
        logger.warning("No previous optimization results found")
        return None
        
    # Find the best performing trial
    best_trial = None
    best_value = float('-inf')
    
    for file in trial_files:
        trial_data = torch.load(file, weights_only=True)
        if trial_data['value'] > best_value:
            best_value = trial_data['value']
            best_trial = trial_data

    return best_trial

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='BERT Classifier Hyperparameter Optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    ModelConfig.add_argparse_args(parser)
    
    optim = parser.add_argument_group('Optimization')
    optim.add_argument('--timeout', type=int, default=None,
                      help='Optimization timeout in seconds')
    optim.add_argument('--study_name', type=str, default='bert_optimization',
                      help='Base name for the Optuna study')
    optim.add_argument('--storage', type=str, default=None,
                      help='Database URL for Optuna storage')
    optim.add_argument('--seed', type=int, default=None,
                      help='Random seed for sampler')
    
    return parser.parse_args()

if __name__ == "__main__":
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
                        experiment_name=study_name,
                        storage=args.storage,
                        random_seed=seed,
                        n_trials=trials_per_exp
                    )
            logger.info("Experiment %d completed", experiment_id + 1)
            logger.info("Best parameters: %s", best_params)
            
        logger.info("\nAll experiments completed successfully")
    except Exception as e:
        logger.error("Error during optimization: %s", str(e), exc_info=True)
        raise