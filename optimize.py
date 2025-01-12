import os
from typing import Dict, List, Tuple, Optional, Any
import optuna
import torch
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
    BertModel,  # Add this import
    get_linear_schedule_with_warmup
)
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import SuccessiveHalvingPruner, HyperbandPruner

from config import ModelConfig
from model import BERTClassifier
from dataset import TextClassificationDataset
from trainer import Trainer
import yaml  # Add this import at the top
from datetime import datetime

logger = logging.getLogger(__name__)

def create_study(study_name: str, storage: Optional[str] = None, 
                sampler_type: str = 'tpe') -> optuna.Study:
    """Create an Optuna study with enhanced sampling and pruning
    
    Available samplers:
    - 'tpe': Tree-structured Parzen Estimators (default)
    - 'random': Random sampling
    - 'cmaes': Covariance Matrix Adaptation Evolution Strategy
    - 'qmc': Quasi Monte Carlo
    - 'grid': Grid sampling
    """
    sampler = {
        'tpe': TPESampler(
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True,
            seed=42
        ),
        'random': optuna.samplers.RandomSampler(seed=42),
        'cmaes': optuna.samplers.CmaEsSampler(
            n_startup_trials=10,
            seed=42
        ),
        'qmc': optuna.samplers.QMCSampler(
            qmc_type='sobol',
            seed=42
        ),
        'grid': optuna.samplers.GridSampler({
            'learning_rate': [1e-5, 5e-5, 1e-4],
            'batch_size': [16, 32, 64],
            'num_layers': [1, 2, 3]
        })
    }.get(sampler_type, TPESampler(seed=42))

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

def save_model_checkpoint(model: BERTClassifier, trial_number: int, accuracy: float, study_name: str, config: ModelConfig) -> None:
    """Save model checkpoint with trial information."""
    config.best_trials_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = config.best_trials_dir / f'model_checkpoint_{study_name}_trial_{trial_number}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'trial_number': trial_number,
        'accuracy': accuracy,
        'config': model.classifier_config
    }, checkpoint_path)
    logger.info(f"Saved model checkpoint: {checkpoint_path}")

def objective(
    trial: optuna.Trial,
    config: ModelConfig,
    texts: List[str],
    labels: List[int],
    study_name: str,
    best_model_info: Dict[str, Any],  # Add dictionary to track best model
    trial_pbar: Optional[tqdm] = None,
    epoch_pbar: Optional[tqdm] = None
) -> float:
    # First, split the data and create dataloaders with the base batch size
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # First suggest architecture type
    arch_type = trial.suggest_categorical('architecture_type', ['standard', 'plane_resnet'])
    
    # Base configuration
    classifier_config = {
        'architecture_type': arch_type,
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
    }

    if arch_type == 'plane_resnet':
        # Plane ResNet specific parameters
        classifier_config.update({
            'num_planes': trial.suggest_int('num_planes', 4, 16),
            'plane_width': trial.suggest_categorical('plane_width', [32, 64, 128, 256]),
            'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.2),
            'cls_pooling': trial.suggest_categorical('cls_pooling', [True, False])  # Add cls_pooling parameter
        })
    else:
        # Standard architecture parameters
        classifier_config.update({
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256, 512, 1024]),
            'activation': trial.suggest_categorical('activation', ['gelu', 'relu']),
            'regularization': trial.suggest_categorical('regularization', ['dropout', 'batchnorm']),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'cls_pooling': trial.suggest_categorical('cls_pooling', [True, False]),
            'init_scale': trial.suggest_float('init_scale', 0.01, 0.1, log=True),
            'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.2)
        })

        # Existing parameter constraints
        if classifier_config['num_layers'] == 1:
            classifier_config['hidden_dim'] = max(classifier_config['hidden_dim'], 256)
        elif classifier_config['hidden_dim'] < 256:
            classifier_config['learning_rate'] = max(classifier_config['learning_rate'], 1e-4)
    
    # Update config with trial-specific parameters
    config.learning_rate = classifier_config['learning_rate']
    config.batch_size = classifier_config['batch_size']
    
    # Create datasets and dataloaders with the trial's batch size
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name, clean_up_tokenization_spaces=True)
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, config.max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=classifier_config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=classifier_config['batch_size'])
    
    # Calculate total steps and warmup steps after dataloader creation
    total_steps = len(train_dataloader) * config.num_epochs
    classifier_config['warmup_steps'] = int(total_steps * classifier_config['warmup_ratio'])
    
    # Log trial configuration only once
    logger.info(f"\nTrial #{trial.number} configuration:")
    for key, value in classifier_config.items():
        logger.info(f"  {key}: {value}")
    
    # Create model and setup training (model will log its own architecture)
    model = BERTClassifier(config.bert_model_name, config.num_classes, classifier_config)
    trainer = Trainer(model, config)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=classifier_config['learning_rate'],
        weight_decay=classifier_config['weight_decay']
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=classifier_config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Training loop
    best_accuracy = 0.0
    no_improve_count = 0
    # Increase base patience and scale with trial number for better exploration
    patience = max(5, trial.number // 5)  # Modified patience calculation
    
    if epoch_pbar is not None:
        epoch_pbar.reset()
        epoch_pbar.total = config.num_epochs
    
    if trial_pbar is not None:
        trial_pbar.set_description(f'Trial {trial.number}/{config.n_trials}')
    
    for epoch in range(config.num_epochs):
        if epoch_pbar is not None:
            epoch_pbar.set_description(f'Trial {trial.number} Epoch {epoch+1}/{config.num_epochs}')
        
        trainer.train_epoch(train_dataloader, optimizer, scheduler)
        accuracy, _ = trainer.evaluate(val_dataloader)
        
        if epoch_pbar is not None:
            epoch_pbar.update(1)
            epoch_pbar.set_postfix({
                'accuracy': f'{accuracy:.4f}',
                'trial': f'{trial.number}/{config.n_trials}'
            })
        
        trial.report(accuracy, epoch)
        
        # Save model if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improve_count = 0
            # Update best model info atomically
            best_model_info.update({
                'model_state': model.state_dict(),
                'config': classifier_config.copy(),
                'accuracy': accuracy,
                'trial_number': trial.number,
                'params': trial.params
            })
        else:
            no_improve_count += 1
        
        # Pruning check - only stop if we've seen enough epochs to be confident
        if epoch >= min(5, config.num_epochs // 2):  # Add minimum epochs check
            if trial.should_prune() or no_improve_count >= patience:
                logger.info(f"Trial #{trial.number} pruned after {epoch + 1} epochs!")
                logger.info(f"Best accuracy achieved: {best_accuracy:.4f}")
                raise optuna.TrialPruned()
    
    logger.info(f"Trial #{trial.number} finished with best accuracy: {best_accuracy:.4f}")
    
    if trial_pbar is not None:
        trial_pbar.update(1)
    
    return best_accuracy

def load_data(config: ModelConfig) -> Tuple[List[str], List[int], LabelEncoder]:
    df = pd.read_csv(config.data_file)
    le = LabelEncoder()
    texts = df['text'].tolist()
    labels = le.fit_transform(df["category"]).tolist()
    return texts, labels, le

def initialize_progress_bars(n_trials: int, num_epochs: int) -> Tuple[tqdm, tqdm]:
    trial_pbar = tqdm(total=n_trials, desc='Trials', position=0)
    epoch_pbar = tqdm(total=num_epochs, desc='Epochs', position=1, leave=True)
    return trial_pbar, epoch_pbar

def load_previous_best_trial(study_name: str, config: ModelConfig) -> Optional[Dict[str, Any]]:
    """Load the previous best trial if it exists."""
    best_model_path = config.best_trials_dir / f'best_model_{study_name}.pt'
    
    # Ensure directory exists
    config.best_trials_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(best_model_path):
        logger.info(f"No previous best trial found at: {best_model_path}")
        return None
        
    try:
        checkpoint = torch.load(best_model_path, map_location='cpu')
        return {
            'accuracy': checkpoint['accuracy'],
            'trial_number': checkpoint['trial_number'],
            'model_state': checkpoint['model_state_dict'],
            'config': checkpoint['config'],
            'params': checkpoint['hyperparameters']
        }
    except Exception as e:
        logger.warning(f"Error loading previous best trial from {best_model_path}: {str(e)}")
        if os.path.exists(best_model_path):
            corrupt_path = best_model_path.with_suffix('.corrupt')
            os.rename(best_model_path, corrupt_path)
            logger.warning(f"Renamed corrupt model file to: {corrupt_path}")
        return None

def save_best_trial(best_model_info: Dict[str, Any], study_name: str, config: ModelConfig) -> None:
    """Save the current best trial, comparing with previous best if it exists."""
    config.best_trials_dir.mkdir(exist_ok=True, parents=True)
    final_model_path = config.best_trials_dir / f'best_model_{study_name}.pt'
    previous_best = load_previous_best_trial(study_name, config)
    
    if previous_best is not None:
        if previous_best['accuracy'] >= best_model_info['accuracy']:
            logger.info("Previous best trial has better or equal performance. Keeping previous model.")
            logger.info(f"Previous best accuracy: {previous_best['accuracy']:.4f}")
            logger.info(f"Current best accuracy: {best_model_info['accuracy']:.4f}")
            return

    # Save new best model
    torch.save({
        'model_state_dict': best_model_info['model_state'],
        'config': best_model_info['config'],
        'trial_number': best_model_info['trial_number'],
        'accuracy': best_model_info['accuracy'],
        'study_name': study_name,
        'hyperparameters': best_model_info['params'],
        'timestamp': datetime.now().isoformat()
    }, final_model_path)
    
    logger.info(f"Saved new best model to {final_model_path}")
    logger.info(f"New best trial number: {best_model_info['trial_number']}")
    logger.info(f"New best accuracy: {best_model_info['accuracy']:.4f}")

def get_best_trial_ever(study_name: str, config: ModelConfig) -> Optional[Dict[str, Any]]:
    """Get the best trial from all previous experiments."""
    best_model_path = config.best_trials_dir / f'best_model_{study_name}.pt'
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path)
            return {
                'accuracy': checkpoint['accuracy'],
                'trial_number': checkpoint['trial_number'],
                'params': checkpoint['hyperparameters'],
                'timestamp': checkpoint.get('timestamp', 'unknown')
            }
        except Exception as e:
            logger.warning(f"Error loading previous best trial: {e}")
    return None

def run_optimization(config: ModelConfig, timeout: Optional[int] = None, 
                    study_name: str = 'bert_optimization',
                    storage: Optional[str] = None) -> Dict[str, Any]:
    logger.info("\n" + "="*50)
    logger.info("Starting optimization")
    logger.info(f"Number of trials: {config.n_trials}")
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
    texts, labels, _ = load_data(config)
    logger.info(f"Loaded {len(texts)} samples")
    logger.info(f"Number of classes: {len(set(labels))}")
    logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # Create progress bars
    logger.info("\nInitializing progress bars...")
    trial_pbar, epoch_pbar = initialize_progress_bars(config.n_trials, config.num_epochs)
    
    # Create study with sampler from config
    logger.info(f"Creating study with {config.sampler} sampler...")
    study = create_study(study_name, storage, config.sampler)
    
    # Add callbacks for better analysis
    study.set_user_attr("best_value", 0.0)
    
    # Fix the callback to use the study_name from the closure
    def callback(study: optuna.Study, trial: optuna.Trial) -> None:
        if trial.value is not None and trial.value > study.user_attrs["best_value"]:
            study.set_user_attr("best_value", trial.value)
            config.best_trials_dir.mkdir(exist_ok=True, parents=True)
            # Save best trial info
            torch.save({
                'trial_number': trial.number,
                'params': trial.params,
                'value': trial.value,
                'study_name': study_name  # Include study_name in the saved data
            }, config.best_trials_dir / f'best_trial_{study_name}.pt')
    
    # Dictionary to track best model info
    best_model_info = {}
    
    # Run optimization with progress bars
    objective_with_progress = partial(objective, 
                                    config=config, 
                                    texts=texts, 
                                    labels=labels,
                                    study_name=study_name,
                                    best_model_info=best_model_info,  # Pass best_model_info
                                    trial_pbar=trial_pbar,
                                    epoch_pbar=epoch_pbar)
    
    try:
        study.optimize(
            objective_with_progress,
            n_trials=config.n_trials,
            timeout=timeout,
            callbacks=[callback],  # Use the inner callback function
            gc_after_trial=True  # Add garbage collection
        )
    finally:
        # Clean up progress bars
        trial_pbar.close()
        epoch_pbar.close()
        
        # Save the best model if we found one
        if best_model_info:
            save_best_trial(best_model_info, study_name, config)
        
        # Plot optimization results
        try:
            from optuna.visualization import plot_optimization_history, plot_parallel_coordinate
            import plotly
            
            n_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            
            if n_completed_trials > 0:
                # Create visualizations based on number of completed trials
                figs = {}
                
                # Always create optimization history plot if we have at least one trial
                figs['optimization_history'] = plot_optimization_history(study)
                
                # Only create these plots if we have more than one completed trial
                if n_completed_trials > 1:
                    figs['parallel_coordinate'] = plot_parallel_coordinate(study)
                    figs['param_importances'] = optuna.visualization.plot_param_importances(study)
                    figs['slice_plot'] = optuna.visualization.plot_slice(study)
                
                config.best_trials_dir.mkdir(exist_ok=True, parents=True)
                for name, fig in figs.items():
                    fig.write_html(config.best_trials_dir / f"{name}_{study_name}.html")
            else:
                logger.warning("No trials completed. Skipping visualizations.")
                
        except ImportError:
            logger.warning("Plotly not installed. Skipping visualization.")
        except Exception as e:
            logger.warning(f"Error creating visualizations: {str(e)}")
        
        # Save study statistics in YAML
        try:
            if n_completed_trials > 0:
                study_stats = {
                    'best_trial': study.best_trial.number,
                    'best_value': float(study.best_trial.value),
                    'best_params': dict(study.best_trial.params),
                    'n_trials': len(study.trials),
                    'n_complete': n_completed_trials,
                    'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                }
            else:
                study_stats = {
                    'n_trials': len(study.trials),
                    'n_complete': 0,
                    'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                    'status': 'No trials completed'
                }
            
            # Save as YAML
            config.best_trials_dir.mkdir(exist_ok=True, parents=True)
            with open(config.best_trials_dir / f'study_stats_{study_name}.yaml', 'w') as f:
                yaml.safe_dump(study_stats, f, default_flow_style=False, sort_keys=False)
                
        except Exception as e:
            logger.error(f"Error saving study statistics: {str(e)}")
    
    logger.info("\n" + "="*50)
    logger.info("Optimization finished!")
    
    if n_completed_trials > 0:
        logger.info("Best trial:")
        trial = study.best_trial
        logger.info(f"  Value: {trial.value:.4f}")
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")
        return study.best_trial.params
    else:
        logger.warning("No trials completed in current experiment.")
        best_ever = get_best_trial_ever(study_name, config)
        if best_ever:
            logger.info("Returning best trial from previous experiments:")
            logger.info(f"  Accuracy: {best_ever['accuracy']:.4f}")
            logger.info(f"  Trial number: {best_ever['trial_number']}")
            logger.info(f"  From experiment at: {best_ever['timestamp']}")
            logger.info("  Parameters:")
            for key, value in best_ever['params'].items():
                logger.info(f"    {key}: {value}")
            return best_ever['params']
        else:
            logger.warning("No successful trials found in any experiment.")
            return {}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='BERT Classifier Hyperparameter Optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add all configuration options (including sampler)
    ModelConfig.add_argparse_args(parser)
    
    # Add optimization-specific arguments
    optim = parser.add_argument_group('Optimization')
    optim.add_argument('--timeout', type=int, default=None,
                      help='Optimization timeout in seconds')
    optim.add_argument('--study-name', type=str, default='bert_optimization',
                      help='Name for the Optuna study')
    optim.add_argument('--storage', type=str, default=None,
                      help='Database URL for Optuna storage')
    # Remove duplicate sampler argument since it's now in ModelConfig
    
    args = parser.parse_args()
    
    # Validate arguments (similar to bert_classifier.py)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error("CUDA device requested but CUDA is not available")
    # ...other validations...
    
    
    return args

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
    config = ModelConfig.from_args(args)
    
    print("Starting optimization run...")
    try:
        best_params = run_optimization(
            config,
            timeout=args.timeout,
            study_name=args.study_name,
            storage=args.storage
        )
        print("Optimization completed successfully")
        print(f"Best parameters found: {best_params}")
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        logger.error("Error during optimization", exc_info=True)
        raise
