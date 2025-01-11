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

def objective(
    trial: optuna.Trial,
    config: ModelConfig,
    texts: List[str],
    labels: List[int],
    trial_pbar: Optional[tqdm] = None,
    epoch_pbar: Optional[tqdm] = None
) -> float:
    logger.info(f"\nStarting trial #{trial.number}")
    logger.info("Sampling hyperparameters...")
    
    # Calculate layer sizes
    initial_size = BertModel.from_pretrained(config.bert_model_name).config.hidden_size
    current_size = initial_size
    layer_sizes = [initial_size]
    
    num_layers = trial.suggest_int('num_layers', 1, 4)
    for _ in range(num_layers):
        current_size = current_size // 2
        layer_sizes.append(current_size)
    
    # Define hyperparameters to optimize
    classifier_config = {
        'num_layers': trial.suggest_int('num_layers', 1, 6),
        'hidden_dim': trial.suggest_int('hidden_dim', 64, 1024, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128]),
        'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.2),
        'activation': trial.suggest_categorical('activation', ['relu', 'gelu']),
        'regularization': trial.suggest_categorical('regularization', ['dropout', 'batchnorm']),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'cls_pooling': trial.suggest_categorical('cls_pooling', [True, False]),
    }

    # Dynamic warmup steps based on ratio
    total_steps = len(train_dataloader) * config.num_epochs
    classifier_config['warmup_steps'] = int(total_steps * classifier_config['warmup_ratio'])

    # Update config with trial-specific parameters
    config.learning_rate = classifier_config['learning_rate']
    config.batch_size = classifier_config['batch_size']
    
    logger.info(f"Trial #{trial.number} hyperparameters:")
    for key, value in classifier_config.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("Layer architecture:")
    for i, size in enumerate(layer_sizes[:-1]):
        logger.info(f"  Layer {i}: {size} -> {layer_sizes[i+1]}")
    logger.info(f"  Output: {layer_sizes[-1]} -> {config.num_classes}")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets and dataloaders
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name, clean_up_tokenization_spaces=True)
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, config.max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Create model with trial hyperparameters
    model = BERTClassifier(config.bert_model_name, config.num_classes, classifier_config)
    trainer = Trainer(model, config)
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=classifier_config['learning_rate'],
        weight_decay=classifier_config['weight_decay']
    )
    total_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=classifier_config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Early stopping implementation
    patience = max(3, trial.number // 10)  # Increase patience as trials progress
    best_accuracy = 0
    no_improve_count = 0
    
    # Training loop
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
        
        # Handle early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Pruning check
        if trial.should_prune() or no_improve_count >= patience:
            logger.info(f"Trial #{trial.number} pruned!")
            raise optuna.TrialPruned()
            
        best_accuracy = max(best_accuracy, accuracy)
    
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
            n_trials=config.n_trials,  # Use config.n_trials instead of undefined n_trials
            timeout=timeout,
            callbacks=[callback],
            gc_after_trial=True  # Add garbage collection
        )
    finally:
        # Clean up progress bars
        trial_pbar.close()
        epoch_pbar.close()
        
        # Plot optimization results
        try:
            from optuna.visualization import plot_optimization_history, plot_parallel_coordinate
            import plotly
            
            figs = {
                'optimization_history': plot_optimization_history(study),
                'parallel_coordinate': plot_parallel_coordinate(study),
                'param_importances': optuna.visualization.plot_param_importances(study),
                'slice_plot': optuna.visualization.plot_slice(study)
            }
            
            for name, fig in figs.items():
                fig.write_html(f"{name}_{study_name}.html")
                
        except ImportError:
            logger.warning("Plotly not installed. Skipping visualization.")
            
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
