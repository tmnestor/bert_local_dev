#!/usr/bin/env python
import argparse
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple

import optuna
import torch
from torch import optim
from optuna._experimental import ExperimentalWarning
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

from ..config.config import ModelConfig
from ..models.model import BERTClassifier
from ..training.trainer import Trainer
from ..data_utils import (
    load_and_preprocess_data,
    create_dataloaders
)
from ..utils.train_utils import (
    initialize_progress_bars,
    log_separator
)
from ..utils.logging_manager import get_logger, setup_logging  # Change from setup_logger
from ..config.defaults import MODEL_DEFAULTS  # Add this import

# Silence specific Optuna warnings
warnings.filterwarnings('ignore', category=ExperimentalWarning)

logger = get_logger(__name__)  # Change to get_logger

def create_optimizer(
    optimizer_name: str,
    model_params: Iterator[torch.nn.Parameter],
    **kwargs
) -> torch.optim.Optimizer:
    """Create optimizer instance with proper parameter mapping.
    
    Args:
        optimizer_name: Name of optimizer to create
        model_params: Model parameters to optimize
        **kwargs: Optimizer configuration parameters
        
    Returns:
        Configured optimizer instance
        
    Raises:
        ValueError: If optimizer_name is invalid
    """
    # Map common parameter names to optimizer-specific names
    param_mapping = {
        'learning_rate': 'lr',
        'weight_decay': 'weight_decay',
        'momentum': 'momentum',
        'beta1': 'betas[0]',
        'beta2': 'betas[1]'
    }
    
    # Convert parameters using mapping
    optimizer_kwargs = {}
    for key, value in kwargs.items():
        if key in param_mapping:
            mapped_key = param_mapping[key]
            if '[' in mapped_key:  # Handle nested params like betas
                base_key, idx = mapped_key.split('[')
                idx = int(idx.rstrip(']'))
                if base_key not in optimizer_kwargs:
                    optimizer_kwargs[base_key] = [0.9, 0.999]  # Default AdamW betas
                optimizer_kwargs[base_key][idx] = value
            else:
                optimizer_kwargs[mapped_key] = value
        else:
            optimizer_kwargs[key] = value

    optimizers = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop
    }
    
    if optimizer_name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. "
                       f"Available optimizers: {list(optimizers.keys())}")
    
    return optimizers[optimizer_name.lower()](model_params, **optimizer_kwargs)

def _create_study(name: str, storage: Optional[str] = None, 
                sampler_type: str = 'tpe', random_seed: Optional[int] = None) -> optuna.Study:
    """Create an Optuna study for hyperparameter optimization."""
    # Silence optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
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

    # Adjust pruner to be less aggressive
    pruner = HyperbandPruner(
        min_resource=3,  # Increased from 1
        max_resource=15,  # Increased from 10
        reduction_factor=2  # Decreased from 3
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
    """Run hyperparameter optimization for the BERT classifier."""
    # Only show detailed info if verbosity > 0
    if model_config.verbosity > 0:
        log_separator(logger)
        logger.info("Starting optimization")
        logger.info("Number of trials: %s", n_trials or model_config.n_trials)
        logger.info("\nLoading data...")
    
    # Load data silently
    train_texts, val_texts, train_labels, val_labels, label_encoder = load_and_preprocess_data(model_config)
    
    if model_config.num_classes is None:
        model_config.num_classes = len(label_encoder.classes_)
    
    texts = train_texts + val_texts
    labels = train_labels + val_labels
    
    # Initialize progress bar
    trial_pbar = tqdm(
        total=n_trials or model_config.n_trials,
        desc='Trials',
        position=0,
        leave=True,
        ncols=80,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
    )
    
    # Create study and run optimization
    study = _create_study(experiment_name, storage, model_config.sampler, random_seed)
    study.set_user_attr("best_value", 0.0)
    global_best_info = {'score': float('-inf'), 'model_info': None}
    
    try:
        study.optimize(
            partial(objective, 
                   model_config=model_config,
                   texts=texts, 
                   labels=labels,
                   best_model_info=global_best_info,
                   trial_pbar=trial_pbar),
            n_trials=n_trials or model_config.n_trials,
            timeout=timeout,
            callbacks=[save_trial_callback(experiment_name, model_config, global_best_info)],
            gc_after_trial=True
        )
    finally:
        trial_pbar.close()
        
        # Always show best trial summary and save best model
        if global_best_info['model_info']:
            print("\nBest Trial Configuration:")
            print("=" * 50)
            print(f"Trial Number: {global_best_info['model_info']['trial_number']}")
            print(f"Score: {global_best_info['score']:.4f}")
            print("\nHyperparameters:")
            for key, value in global_best_info['model_info']['params'].items():
                print(f"  {key}: {value}")
            print("=" * 50)
            
            # Always save best model regardless of verbosity
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
    
    try:
        # Debug logging
        logger.info("Saving trial with performance:")
        logger.info("Trial Score in best_model_info: %s", best_model_info.get(f'{model_config.metric}_score'))
        logger.info("Model State Dict Size: %d", len(best_model_info['model_state']))
        
        metric_key = f'{model_config.metric}_score'
        save_dict = {
            'model_state_dict': best_model_info['model_state'],
            'config': {
                'classifier_config': best_model_info['config'],
                'model_config': {
                    'bert_hidden_size': MODEL_DEFAULTS['bert_hidden_size'],
                    'num_classes': model_config.num_classes
                }
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
    except Exception as e:
        logger.error(f"Failed to save best trial: {str(e)}")
        raise IOError(f"Failed to save best trial: {str(e)}")

# First save location: During optimization in the objective function
def setup_training_components(model_config, classifier_config, optimizer_name, optimizer_config, train_dataloader):
    """Set up model, trainer, optimizer and scheduler."""
    model = BERTClassifier(model_config.bert_model_name, model_config.num_classes, classifier_config)
    trainer = Trainer(model, model_config)
    
    optimizer = create_optimizer(
        optimizer_name,
        model.parameters(),
        **optimizer_config
    )
    
    total_steps = len(train_dataloader) * model_config.num_epochs
    warmup_steps = int(total_steps * classifier_config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    return model, trainer, optimizer, scheduler

def calculate_hidden_sizes(bert_hidden_size: int, num_classes: int, num_layers: int) -> List[int]:
    """Calculate hidden layer sizes that decrease geometrically from BERT size to near num_classes.
    
    Args:
        bert_hidden_size: Size of BERT hidden layer
        num_classes: Number of output classes
        num_layers: Number of hidden layers to generate
        
    Returns:
        List of hidden layer sizes
    """
    if num_layers == 0:
        return []
    
    # Start with double BERT size and decrease geometrically
    start_size = 2 * bert_hidden_size
    min_size = max(num_classes * 2, 64)  # Don't go smaller than this
    
    # Calculate ratio for geometric decrease
    ratio = (min_size / start_size) ** (1.0 / (num_layers + 1))
    
    sizes = []
    current_size = start_size
    for _ in range(num_layers):
        current_size = int(current_size * ratio)
        # Round to nearest even number and ensure minimum size
        current_size = max(min_size, 2 * (current_size // 2))
        sizes.append(current_size)
    
    return sizes

def get_trial_config(trial: optuna.Trial, model_config: ModelConfig) -> Tuple[int, Dict, str, Dict]:
    """Get trial configuration parameters."""
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    
    # Get number of hidden layers
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 4)
    
    # Calculate hidden dimensions based on BERT size and number of classes
    hidden_dims = calculate_hidden_sizes(
        bert_hidden_size=MODEL_DEFAULTS['bert_hidden_size'],
        num_classes=model_config.num_classes,  # Use from model_config instead
        num_layers=num_hidden_layers
    )
    
    # Add activation function as a trial parameter
    activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'silu', 'tanh'])
    
    classifier_config = {
        'hidden_dim': hidden_dims,
        'activation': activation,  # Now using suggested activation
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.6),
        'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True),
        'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.3)
    }
    
    optimizer_name = trial.suggest_categorical('optimizer', ['adamw', 'sgd', 'rmsprop'])
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 5e-3, log=True)
    
    optimizer_config = get_optimizer_config(trial, optimizer_name, learning_rate)
    classifier_config.update({
        'learning_rate': learning_rate,
        'optimizer': optimizer_name,
        'optimizer_config': optimizer_config.copy()
    })
    
    return batch_size, classifier_config, optimizer_name, optimizer_config

def get_optimizer_config(trial, optimizer_name, learning_rate):
    """Get optimizer-specific configuration."""
    optimizer_config = {
        'lr': learning_rate,
        'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
    }
    
    if optimizer_name == 'sgd':
        optimizer_config.update({
            'momentum': trial.suggest_float('momentum', 0.0, 0.99),
            'nesterov': trial.suggest_categorical('nesterov', [True, False])
        })
    elif optimizer_name == 'adamw':
        optimizer_config.update({
            'betas': (
                trial.suggest_float('beta1', 0.5, 0.9999),
                trial.suggest_float('beta2', .9, 0.9999)
            ),
            'eps': trial.suggest_float('eps', 1e-8, 1e-6, log=True)
        })
    elif optimizer_name == 'rmsprop':
        optimizer_config.update({
            'momentum': trial.suggest_float('momentum', 0.0, 0.99),
            'alpha': trial.suggest_float('alpha', 0.8, 0.99)
        })
    
    return optimizer_config

def log_current_best(best_info: Dict[str, Any]) -> None:
    """Log the current best trial configuration in a standardized format."""
    if not best_info['model_info']:
        return
        
    info = best_info['model_info']
    params = info['params']
    clf_config = info['config']
    
    logger.info("\nCurrent Best Trial Configuration:")
    logger.info("=" * 50)
    logger.info("Trial Number: %d", info['trial_number'])
    logger.info("Score: %.4f", best_info['score'])
    logger.info("\nHyperparameters:")
    logger.info("  batch_size: %d", params['batch_size'])
    logger.info("  hidden_layers: %s", clf_config['hidden_dim'])
    logger.info("  dropout_rate: %.4f", clf_config['dropout_rate'])
    logger.info("  weight_decay: %.6f", clf_config['weight_decay'])
    logger.info("  warmup_ratio: %.2f", clf_config['warmup_ratio'])
    logger.info("  optimizer: %s", clf_config['optimizer'])
    logger.info("  learning_rate: %.6f", clf_config['learning_rate'])
    logger.info("=" * 50)

def log_trial_config(trial_num: int, classifier_config: Dict[str, Any], total_trials: int, score: Optional[float] = None) -> None:
    """Log trial configuration in a clean, structured format."""
    hidden_dims = classifier_config['hidden_dim']
    opt_config = classifier_config['optimizer_config']
    activation = classifier_config.get('activation', 'gelu')
    
    logger.info("\nTrial %d of %d:", trial_num + 1, total_trials)
    logger.info("=" * 50)
    logger.info("Architecture:")
    logger.info("  Hidden layers: %s", hidden_dims)
    logger.info("  Activation: %s", activation)
    logger.info("  Dropout rate: %.3f", classifier_config['dropout_rate'])
    logger.info("  Weight decay: %.2e", classifier_config['weight_decay'])
    logger.info("  Warmup ratio: %.2f", classifier_config['warmup_ratio'])
    
    if classifier_config['optimizer'] == 'rmsprop':
        logger.info("  Momentum: %.3f", opt_config['momentum'])
        logger.info("  Alpha: %.3f", opt_config['alpha'])
    elif classifier_config['optimizer'] == 'sgd':
        logger.info("  Momentum: %.3f", opt_config['momentum'])
        logger.info("  Nesterov: %s", opt_config.get('nesterov', False))
        
    if score is not None:
        logger.info("\nScore:")
        logger.info("  f1: %.4f", score)
    logger.info("=" * 50)

def log_trial_summary(trial_num: int, config: Dict[str, Any], score: float, total_trials: int) -> None:
    """Log a clean summary of trial configuration and performance."""
    # Only log if we have a score (i.e., at end of trial)
    if score is None:
        return
        
    tqdm.write("\n" + "=" * 50)
    tqdm.write(f"Trial {trial_num + 1} of {total_trials}:")
    tqdm.write("=" * 50)
    tqdm.write("Architecture:")
    tqdm.write(f"  Hidden layers: {config['hidden_dim']}")
    tqdm.write(f"  Activation: {config['activation']}")
    tqdm.write(f"  Dropout rate: {config['dropout_rate']:.3f}")
    tqdm.write("")
    tqdm.write(f"Optimizer: {config['optimizer']}")
    tqdm.write(f"  Learning rate: {config['learning_rate']:.2e}")
    tqdm.write(f"  Weight decay: {config['weight_decay']:.2e}")
    tqdm.write(f"  Warmup ratio: {config['warmup_ratio']:.2f}")

    opt_config = config['optimizer_config']
    if config['optimizer'] == 'rmsprop':
        tqdm.write(f"  Momentum: {opt_config['momentum']:.3f}")
        tqdm.write(f"  Alpha: {opt_config['alpha']:.3f}")
    elif config['optimizer'] == 'sgd':
        tqdm.write(f"  Momentum: {opt_config['momentum']:.3f}")
        tqdm.write(f"  Nesterov: {opt_config.get('nesterov', False)}")
    
    tqdm.write("\nScore:")
    tqdm.write(f"  f1: {score:.4f}")
    tqdm.write("=" * 50 + "\n")

def objective(trial: optuna.Trial, model_config: ModelConfig, texts: List[str], labels: List[int],
             best_model_info: Dict[str, Any], trial_pbar: Optional[tqdm] = None) -> float:
    """Optimization objective function for a single trial."""
    trial_best_score = 0.0
    classifier_config = None
    
    try:
        batch_size, classifier_config, optimizer_name, optimizer_config = get_trial_config(
            trial, model_config
        )
        
        # Only show trial progress if verbosity > 0
        if model_config.verbosity > 0:
            log_trial_summary(trial.number, classifier_config, None, model_config.n_trials)
        
        # Create train/val split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=trial.number, stratify=labels
        )
        
        # Create dataloaders
        train_dataloader, val_dataloader = create_dataloaders(
            [train_texts, val_texts],
            [train_labels, val_labels],
            model_config,
            batch_size
        )
        
        # Setup training components
        model, trainer, optimizer, scheduler = setup_training_components(
            model_config, classifier_config, optimizer_name, optimizer_config, train_dataloader
        )
        
        # Training loop
        trial_best_state = None
        patience = max(3, min(8, trial.number // 2))
        min_epochs = max(5, model_config.num_epochs // 3)
        no_improve_count = 0
        last_score = float('-inf')
        
        # Main training loop
        for epoch in range(model_config.num_epochs):
            try:
                # Train for one epoch
                train_loss = trainer.train_epoch(train_dataloader, optimizer, scheduler)
                
                # Evaluate
                score, metrics = trainer.evaluate(val_dataloader)
                
                # Report to Optuna
                trial.report(score, epoch)
                
                # Check for improvement
                if score > trial_best_score:
                    no_improve_count = 0
                    trial_best_score = score
                    last_score = score
                    trial_best_state = {
                        'model_state': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                        'config': classifier_config.copy(),
                        f'{model_config.metric}_score': trial_best_score,
                        'trial_number': trial.number,
                        'params': trial.params.copy(),
                        'epoch': epoch,
                        'metrics': metrics
                    }
                else:
                    no_improve_count += 1
                
                # Early stopping checks
                should_stop = False
                if epoch >= min_epochs:
                    if score < last_score * 0.8:  # Significant regression
                        should_stop = True
                    elif no_improve_count >= patience:  # No improvement
                        should_stop = True
                    elif trial.should_prune():  # Optuna wants to prune
                        should_stop = True
                
                if should_stop:
                    break
                    
            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {str(e)}")
                raise
        
        # Update progress and log final result
        if trial_pbar:
            trial_pbar.update(1)
            # Only show per-trial results if verbosity > 0
            if model_config.verbosity > 0:
                log_trial_summary(trial.number, classifier_config, trial_best_score, model_config.n_trials)
        
        # Save best state if this is the best trial so far
        if trial_best_state and trial_best_score > best_model_info['score']:
            best_model_info['score'] = trial_best_score
            best_model_info['model_info'] = trial_best_state
            
        return trial_best_score
        
    except Exception as e:
        if trial_pbar:
            trial_pbar.update(1)
        if model_config.verbosity > 0:  # Only log error details if not in minimal mode
            logger.error(f"Trial {trial.number} failed: {str(e)}", exc_info=True)
        raise

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
        trial_data = torch.load(file, map_location='cpu', weights_only=False)
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

def log_best_configuration(study: optuna.Study, best_info: Dict[str, Any]) -> None:
    """Log details about the best trial configuration."""
    logger.info("\nBest Trial Configuration:")
    logger.info("=" * 50)
    logger.info("Trial Number: %d", best_info['model_info']['trial_number'])
    logger.info("Score: %.4f", best_info['score'])
    logger.info("\nHyperparameters:")
    for key, value in best_info['model_info']['params'].items():
        logger.info("  %s: %s", key, value)
    logger.info("=" * 50)

if __name__ == "__main__":
    args = parse_args()
    config = ModelConfig.from_args(args)
    setup_logging(config)  # Initialize logging configuration first

    
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