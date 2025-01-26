#!/usr/bin/env python
import argparse
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Any

import optuna
import torch
from optuna._experimental import ExperimentalWarning
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from ..config.config import ModelConfig
from ..models.model import BERTClassifier
from ..training.trainer import Trainer
from ..utils.train_utils import (
    load_and_preprocess_data,
    create_dataloaders,
    initialize_progress_bars,
    log_separator
)
from ..utils.logging_manager import setup_logger
from ..config.defaults import OPTIM_SEARCH_SPACE

# Silence specific Optuna warnings
warnings.filterwarnings('ignore', category=ExperimentalWarning)

logger = setup_logger(__name__)

def _create_study(name: str, storage: Optional[str] = None, 
                sampler_type: str = 'tpe', random_seed: Optional[int] = None) -> optuna.Study:
    """Create an Optuna study for hyperparameter optimization.

    Args:
        name: Study name for identification.
        storage: Optional database URL for study storage.
        sampler_type: Type of optimization sampler ('tpe', 'random', 'cmaes', 'qmc').
        random_seed: Optional seed for reproducibility.

    Returns:
        optuna.Study: Configured study instance.
    """
    if random_seed is None:
        random_seed = int(time.time())
        
    sampler = {
        'tpe': TPESampler(
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True,
            warn_independent_sampling=False,  # Suppress warnings
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
    """Run hyperparameter optimization for the BERT classifier.

    Performs systematic hyperparameter search using Optuna, with support for
    multiple trials and early stopping.

    Args:
        model_config: Model configuration instance.
        timeout: Optional timeout in seconds.
        experiment_name: Name for the optimization experiment.
        storage: Optional database URL for persisting results.
        random_seed: Optional seed for reproducibility.
        n_trials: Number of optimization trials to run.

    Returns:
        Dict containing the best parameters found.

    Raises:
        RuntimeError: If optimization fails.
    """
    log_separator(logger)
    logger.info("Starting optimization")
    logger.info("Number of trials: %s", n_trials or model_config.n_trials)
    
    # Load data using utility function - updated to handle train/val split correctly
    logger.info("\nLoading data...")
    train_texts, val_texts, train_labels, val_labels, label_encoder = load_and_preprocess_data(model_config)
    
    # Combine train and validation sets for optimization trials
    texts = train_texts + val_texts
    labels = train_labels + val_labels
    
    logger.info("Loaded %d samples with %d classes", len(texts), model_config.num_classes)
    
    logger.info("Model config:")
    logger.info("  BERT model: %s", model_config.bert_model_name)
    logger.info("  Device: %s", model_config.device)
    logger.info("  Max sequence length: %s", model_config.max_seq_len)  # Updated from max_length
    logger.info("  Metric: %s", model_config.metric)
    
    # Initialize progress bars using utility function
    trial_pbar, epoch_pbar = initialize_progress_bars(n_trials or model_config.n_trials, model_config.num_epochs)
    
    # Create study (fix the reference)
    study = _create_study(experiment_name, storage, model_config.sampler, random_seed)
    study.set_user_attr("best_value", 0.0)
    
    # Track best performance across all trials
    global_best_info = {
        'score': float('-inf'),
        'model_info': None
    }
    
    # Run optimization
    objective_with_progress = partial(objective, 
                                    model_config=model_config,
                                    texts=texts, 
                                    labels=labels,
                                    best_model_info=global_best_info,  # Pass global tracking
                                    trial_pbar=trial_pbar,
                                    epoch_pbar=epoch_pbar)
    
    try:
        study.optimize(
            objective_with_progress,
            n_trials=n_trials or model_config.n_trials,
            timeout=timeout,
            callbacks=[save_trial_callback(experiment_name, model_config, global_best_info)],  # Pass global tracking
            gc_after_trial=True
        )
    finally:
        trial_pbar.close()
        epoch_pbar.close()
        
        if global_best_info['model_info']:
            save_best_trial(global_best_info['model_info'], experiment_name, model_config)
            
    return study.best_trial.params

def save_best_trial(best_model_info: Dict[str, Any], trial_study_name: str, model_config: ModelConfig) -> None:
    """Save the best trial model and configuration.

    Args:
        best_model_info: Dictionary containing model state and metadata.
        trial_study_name: Name of the optimization study.
        model_config: Model configuration instance.

    Raises:
        IOError: If saving fails.
    """
    model_config.best_trials_dir.mkdir(exist_ok=True, parents=True)
    final_model_path = model_config.best_trials_dir / f'best_model_{trial_study_name}.pt'
    
    # Debug logging
    logger.info("Saving trial with performance:")
    logger.info("Trial Score in best_model_info: %s", best_model_info.get(f'{model_config.metric}_score'))
    logger.info("Model State Dict Size: %d", len(best_model_info['model_state']))
    
    metric_key = f'{model_config.metric}_score'
    save_dict = {
        'model_state_dict': best_model_info['model_state'],
        'config': {
            'classifier_config': best_model_info['config'],
            'epoch': -1,
        },
        'metric_value': best_model_info[metric_key],  # Verify this value
        'study_name': trial_study_name,
        'trial_number': best_model_info['trial_number'],
        'num_classes': model_config.num_classes,
        'hyperparameters': best_model_info['params'],
        'val_size': 0.2,  # Add validation split size
        'metric': model_config.metric  # Add which metric was optimized
    }
    torch.save(save_dict, final_model_path)
    logger.info("Best trial metric (%s): %s", metric_key, best_model_info[metric_key])
    logger.info("Saved best model to %s", final_model_path)
    logger.info("Best %s: %.4f", model_config.metric, best_model_info[metric_key])

# First save location: During optimization in the objective function
def objective(trial: optuna.Trial, model_config: ModelConfig, texts: List[str], labels: List[int],
             best_model_info: Dict[str, Any], trial_pbar: Optional[tqdm] = None, 
             epoch_pbar: Optional[tqdm] = None) -> float:
    """Optimization objective function for a single trial."""
    batch_size = trial.suggest_categorical('batch_size', OPTIM_SEARCH_SPACE['batch_size'])
    
    # Create new train/val split for each trial
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=trial.number, stratify=labels
    )
    
    # ... existing split logging code ...
    
    # Single classifier config definition with all parameters (removed cls_pooling)
    classifier_config = {
        'num_layers': trial.suggest_int('num_layers', *OPTIM_SEARCH_SPACE['num_layers']),
        'hidden_dim': trial.suggest_categorical('hidden_dim', OPTIM_SEARCH_SPACE['hidden_dim']),
        'activation': trial.suggest_categorical('activation', OPTIM_SEARCH_SPACE['activation']),
        'dropout_rate': trial.suggest_float('dropout_rate', *OPTIM_SEARCH_SPACE['dropout_rate']),
        'learning_rate': trial.suggest_float('learning_rate', *OPTIM_SEARCH_SPACE['learning_rate'], log=True),
        'weight_decay': trial.suggest_float('weight_decay', *OPTIM_SEARCH_SPACE['weight_decay'], log=True),
        'warmup_ratio': trial.suggest_float('warmup_ratio', *OPTIM_SEARCH_SPACE['warmup_ratio'])
    }

    train_dataloader, val_dataloader = create_dataloaders(
        [train_texts, val_texts],
        [train_labels, val_labels],
        model_config,
        batch_size
    )
    
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
    trial_best_score = 0.0
    trial_best_state = None
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
        
        if score > trial_best_score:
            no_improve_count = 0
            trial_best_score = score
            trial_best_state = {
                'model_state': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'config': classifier_config.copy(),
                f'{model_config.metric}_score': trial_best_score,
                'trial_number': trial.number,
                'params': trial.params.copy(),
                'epoch': epoch  # Add epoch information
            }
            # Update global best if better
            if trial_best_score > best_model_info['score']:
                best_model_info['score'] = trial_best_score
                best_model_info['model_info'] = trial_best_state
                logger.info("New best model found in trial %d with score: %.4f", trial.number, trial_best_score)
        else:
            no_improve_count += 1
            
        if epoch >= min(5, model_config.num_epochs // 2) and (trial.should_prune() or no_improve_count >= patience):
            raise optuna.TrialPruned()
    
    if trial_pbar:
        trial_pbar.update(1)
        
    return trial_best_score

def save_trial_callback(trial_study_name: str, model_config: ModelConfig, best_model_info: Dict[str, Any]):
    """Create a callback for saving trial information.

    Args:
        trial_study_name: Name of the optimization study.
        model_config: Model configuration instance.
        best_model_info: Dictionary tracking best model state.

    Returns:
        Callable: Callback function for Optuna.
    """
    def callback(study: optuna.Study, trial: optuna.Trial):
        if trial.value and trial.value > study.user_attrs.get("best_value", float('-inf')):
            study.set_user_attr("best_value", trial.value)
            model_config.best_trials_dir.mkdir(exist_ok=True, parents=True)
            best_trial_info = {
                'trial_number': trial.number,
                'params': trial.params,
                'value': trial.value,
                'study_name': trial_study_name,
                'best_model_score': best_model_info['score'] if best_model_info['model_info'] else None
            }
            torch.save(best_trial_info, 
                      model_config.best_trials_dir / f'best_trial_{trial_study_name}.pt')
    return callback

def load_best_configuration(best_trials_dir: Path, exp_name: str = None) -> dict:
    """Load best model configuration from optimization results.

    Args:
        best_trials_dir: Directory containing trial results.
        exp_name: Optional experiment name filter.

    Returns:
        dict: Best configuration or None if not found.
    """
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
    """Parse command line arguments for optimization.

    Returns:
        Namespace containing parsed arguments.
    """
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
        raise