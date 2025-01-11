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

def create_study(study_name: str, storage: Optional[str] = None) -> optuna.Study:
    """Create an Optuna study with enhanced sampling and pruning"""
    sampler = TPESampler(
        n_startup_trials=10,  # More startup trials for better exploration
        n_ei_candidates=24,   # More candidates for expected improvement
        multivariate=True,    # Consider parameter relationships
        seed=42
    )

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

def save_model_checkpoint(model: BERTClassifier, trial_number: int, accuracy: float, study_name: str) -> None:
    """Save model checkpoint with trial information."""
    checkpoint_path = f'model_checkpoint_{study_name}_trial_{trial_number}.pt'
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
    logger.info(f"\nStarting trial #{trial.number}")
    logger.info("Sampling hyperparameters...")
    
    # First, split the data and create dataloaders with the base batch size
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Define hyperparameters to optimize
    classifier_config = {
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'hidden_dim': trial.suggest_int('hidden_dim', 64, 1024, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128]),
        'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.2),
        'activation': trial.suggest_categorical('activation', ['relu', 'gelu']),
        'regularization': trial.suggest_categorical('regularization', ['dropout', 'batchnorm']),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'cls_pooling': trial.suggest_categorical('cls_pooling', [True, False]),
        'init_scale': trial.suggest_float('init_scale', 0.01, 0.1, log=True)
    }

    # Parameter constraints
    if classifier_config['hidden_dim'] < 256:
        # Smaller networks might need higher learning rates
        classifier_config['learning_rate'] = max(classifier_config['learning_rate'], 1e-4)
    
    classifier_config.update({
        'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128]),
        'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.2),
        'activation': trial.suggest_categorical('activation', ['relu', 'gelu']),
        'regularization': trial.suggest_categorical('regularization', ['dropout', 'batchnorm']),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'cls_pooling': trial.suggest_categorical('cls_pooling', [True, False])
    })

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
    
    logger.info(f"Trial #{trial.number} hyperparameters:")
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
    patience = max(3, trial.number // 10)  # Increase patience as trials progress
    
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
        
        # Handle early stopping
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
        
        # Pruning check
        if trial.should_prune() or no_improve_count >= patience:
            logger.info(f"Trial #{trial.number} pruned!")
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

def load_previous_best_trial(study_name: str) -> Optional[Dict[str, Any]]:
    """Load the previous best trial if it exists."""
    best_model_path = f'best_model_{study_name}.pt'
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path)
            return {
                'accuracy': checkpoint['accuracy'],
                'trial_number': checkpoint['trial_number'],
                'model_state': checkpoint['model_state_dict'],
                'config': checkpoint['config'],
                'params': checkpoint['hyperparameters']
            }
        except Exception as e:
            logger.warning(f"Error loading previous best trial: {e}")
    return None

def save_best_trial(best_model_info: Dict[str, Any], study_name: str) -> None:
    """Save the current best trial, comparing with previous best if it exists."""
    final_model_path = f'best_model_{study_name}.pt'
    previous_best = load_previous_best_trial(study_name)
    
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
    
    # Create study
    logger.info("\nCreating Optuna study...")
    study = create_study(study_name, storage)
    
    # Add callbacks for better analysis
    study.set_user_attr("best_value", 0.0)
    
    def callback(study: optuna.Study, trial: optuna.Trial):
        if trial.value > study.user_attrs["best_value"]:
            study.set_user_attr("best_value", trial.value)
            # Save best trial info
            torch.save({
                'trial_number': trial.number,
                'params': trial.params,
                'value': trial.value
            }, f'best_trial_{study_name}.pt')
    
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
            callbacks=[callback],
            gc_after_trial=True  # Add garbage collection
        )
    finally:
        # Clean up progress bars
        trial_pbar.close()
        epoch_pbar.close()
        
        # Save the best model if we found one
        if best_model_info:
            save_best_trial(best_model_info, study_name)
        
        # Plot optimization results
        try:
            from optuna.visualization import plot_optimization_history, plot_parallel_coordinate
            import plotly
            
            # Create visualizations based on number of completed trials
            figs = {}
            n_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            
            # Always create optimization history plot
            figs['optimization_history'] = plot_optimization_history(study)
            
            # Only create these plots if we have more than one completed trial
            if n_completed_trials > 1:
                figs['parallel_coordinate'] = plot_parallel_coordinate(study)
                figs['param_importances'] = optuna.visualization.plot_param_importances(study)
                figs['slice_plot'] = optuna.visualization.plot_slice(study)
            else:
                logger.warning("Some visualizations skipped: need more than one completed trial")
            
            for name, fig in figs.items():
                fig.write_html(f"{name}_{study_name}.html")
                
        except ImportError:
            logger.warning("Plotly not installed. Skipping visualization.")
        except Exception as e:
            logger.warning(f"Error creating visualizations: {str(e)}")
            
        # Save study statistics in YAML instead of JSON
        study_stats = {
            'best_trial': study.best_trial.number,
            'best_value': float(study.best_trial.value),  # Convert numpy float to Python float
            'best_params': dict(study.best_trial.params),  # Convert to regular dict
            'n_trials': len(study.trials),
            'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        }
        
        # Save as YAML instead of JSON
        with open(f'study_stats_{study_name}.yaml', 'w') as f:
            yaml.safe_dump(study_stats, f, default_flow_style=False, sort_keys=False)
    
    logger.info("\n" + "="*50)
    logger.info("Optimization finished!")
    logger.info("Best trial:")
    trial = study.best_trial
    
    logger.info(f"  Value: {trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    
    return study.best_params

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='BERT Classifier Hyperparameter Optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add all configuration options
    ModelConfig.add_argparse_args(parser)
    
    # Add optimization-specific arguments
    optim = parser.add_argument_group('Optimization')
    optim.add_argument('--timeout', type=int, default=None,
                      help='Optimization timeout in seconds')
    optim.add_argument('--study-name', type=str, default='bert_optimization',
                      help='Name for the Optuna study')
    optim.add_argument('--storage', type=str, default=None,
                      help='Database URL for Optuna storage')
    
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

        raise
