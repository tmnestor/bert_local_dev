#!/usr/bin/env python
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import optuna
import torch
from optuna._experimental import ExperimentalWarning
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config.config import ModelConfig
from ..models.factory import ModelFactory
from ..training.trainer import Trainer
from ..utils.train_utils import load_and_preprocess_data, create_dataloaders
from ..utils.logging_manager import setup_logger
from ..utils.progress_manager import ProgressManager

# Silence Optuna warnings
warnings.filterwarnings('ignore', category=ExperimentalWarning)

# Add import for disabling Optuna logging
import logging
optuna.logging.set_verbosity(optuna.logging.CRITICAL)  # Completely disable Optuna logging
logging.getLogger('optuna').setLevel(logging.WARNING)

logger = setup_logger(__name__)

class OptimizationError(Exception):
    """Custom exception for optimization errors"""
    pass

class ModelOptimizer:
    """Handles model optimization process"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.best_score = float('-inf')
        self.best_model_info = None
        self.current_model = None  # Track current model
        
        # Load and split data - store as tuples
        logger.info("Loading data for optimization...")
        self.train_data = (
            self.train_texts, 
            self.train_labels
        ) = load_and_preprocess_data(config)[0:2]  # Get only train texts and labels
        
        self.val_data = (
            self.val_texts,
            self.val_labels
        ) = load_and_preprocess_data(config)[2:4]  # Get only val texts and labels
        
        logger.info(f"Train size: {len(self.train_texts)}, Val size: {len(self.val_texts)}")
        self.trial_pbar = None
        self.epoch_pbar = None
        self.batch_pbar = None  # Add batch progress bar
        self.progress = ProgressManager()
        
    def initialize_progress_bars(self):
        """Initialize progress bars for trials and epochs"""
        self._cleanup_progress_bars()
            
        # Create trial progress bar
        self.trial_pbar = tqdm(
            total=self.config.n_trials,
            desc="Trial",
            position=0,
            leave=True,
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
        )
        
        # Create epoch progress bar with compact format
        self.epoch_pbar = tqdm(
            total=self.config.num_epochs,
            desc=f'[Epoch: 1/{self.config.num_epochs}]',  # Initial description
            position=1,
            leave=False,
            ncols=80,
            bar_format='{desc} {postfix}'  # Simplified format
        )

    def _cleanup_progress_bars(self):
        """Clean up all progress bars"""
        for pbar in [self.trial_pbar, self.epoch_pbar, self.batch_pbar]:
            if pbar is not None:
                pbar.close()
        self.trial_pbar = None
        self.epoch_pbar = None
        self.batch_pbar = None

    def objective(self, trial: optuna.Trial) -> float:
        """Optimization objective function"""
        try:
            params = self._suggest_parameters(trial)
            score = self._evaluate_trial(trial, params)
            return score
        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            raise OptimizationError(f"Trial failed: {str(e)}") from e

    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for trial"""
        # Base parameters that apply to both architectures
        arch_type = trial.suggest_categorical('architecture_type', ['standard', 'plane_resnet'])
        
        params = {
            'bert_model_name': self.config.bert_model_name,
            'num_classes': self.config.num_classes,
            'architecture_type': arch_type,  # Ensure this is always set
            'batch_size': trial.suggest_int('batch_size', 32, 64),
            'cls_pooling': True,
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True)
        }
        
        # Architecture-specific parameters
        if arch_type == 'standard':
            params.update({
                'num_layers': trial.suggest_int('std/num_layers', 1, 3),
                'hidden_dim': trial.suggest_int('std/hidden_dim', 128, 512),
                'activation': trial.suggest_categorical('std/activation', ['gelu', 'relu']),
                'dropout_rate': trial.suggest_float('std/dropout_rate', 0.1, 0.5)
            })
        else:  # plane_resnet
            params.update({
                'num_planes': trial.suggest_int('plane/num_planes', 4, 8),
                'plane_width': trial.suggest_int('plane/width', 256, 512, step=32)
            })
        
        return params

    def _create_dataloaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """Create dataloaders with drop_last=True to avoid small batches"""
        train_loader, val_loader = create_dataloaders(
            self.train_data,
            self.val_data,
            self.config,
            batch_size,
            drop_last=True  # Drop incomplete batches
        )
        return train_loader, val_loader

    def _evaluate_trial(self, trial: optuna.Trial, params: Dict[str, Any]) -> float:
        """Evaluate a single trial"""
        try:
            # Clear any existing model and force garbage collection
            if self.current_model is not None:
                del self.current_model
                torch.cuda.empty_cache()
            
            # Set configuration parameters BEFORE creating model
            arch_type = params['architecture_type']
            self.config.architecture = arch_type  # Set architecture first
            self.config.learning_rate = params['learning_rate']
            self.config.weight_decay = params['weight_decay']
            self.config.batch_size = params['batch_size']
            
            # Create dataloaders with proper batch size
            train_loader, val_loader = self._create_dataloaders(params['batch_size'])
            
            # Create model with updated configuration
            model = self._create_model(params)
            self.current_model = model  # Assign to instance variable
            
            # Create trainer after model and config are properly set up
            trainer = Trainer(self.current_model, self.config, disable_pbar=True)
            
            # Reset epoch progress bar
            if self.epoch_pbar:
                self.epoch_pbar.reset()
            
            # Initialize progress bars for this trial
            epoch_bar = self.progress.init_epoch_bar(
                self.config.num_epochs,
                mode='optimize'
            )
            
            # Train and evaluate
            score = self._train_and_evaluate(trainer, train_loader, val_loader)
            
            # Handle best model saving
            if score > self.best_score:
                self.current_model.cpu()
                self._save_best_model(self.current_model, params, score, trial.number)
                self.current_model.to(self.config.device)
            
            # Training finished, log results
            logger.info("\nTrial %d finished with score: %.4f", 
                       trial.number, score)
            
            # Update trial progress
            self.progress.update_trial(trial.number, score)
            
            return score
            
        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            raise OptimizationError(f"Trial failed: {str(e)}") from e
        finally:
            if self.current_model is not None:
                self.current_model.cpu()
            self.progress.close_all()

    def _create_model(self, params: Dict[str, Any]) -> torch.nn.Module:
        """Create model using factory"""
        try:
            model_params = {
                'bert_model_name': self.config.bert_model_name,
                'num_classes': self.config.num_classes,
                'architecture_type': self.config.architecture,  # Use from config
                'cls_pooling': params.get('cls_pooling', True)
            }
            
            # Add architecture-specific parameters
            if self.config.architecture == 'standard':
                model_params['config'] = {
                    'num_layers': params.get('num_layers', 2),
                    'hidden_dim': params.get('hidden_dim', 256),
                    'dropout_rate': params.get('dropout_rate', 0.1),
                    'activation': params.get('activation', 'gelu')
                }
            else:
                model_params.update({
                    'num_planes': params.get('num_planes', 8),
                    'plane_width': params.get('plane_width', 256)
                })
            
            model = ModelFactory.create_model(model_params)
            return model.to(self.config.device)
            
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise OptimizationError(f"Model creation failed: {str(e)}") from e

    def _save_best_model(self, model: torch.nn.Module, params: Dict[str, Any], 
                        score: float, trial_number: int) -> None:
        """Save best model with proper metadata"""
        try:
            if model is None:
                raise OptimizationError("Cannot save None model")
                
            # Create deep copy of model state and ensure training/eval mode is preserved
            training = model.training
            model.eval()  # Switch to eval mode to ensure correct statistics
            model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if training:
                model.train()  # Restore original mode
            
            self.best_score = score
            self.best_model_info = {
                'model_state_dict': model_state,
                'config': {
                    'classifier_config': params.copy(),
                    'architecture_type': params['architecture_type']
                },
                'score': score,
                'trial': trial_number,
                'num_classes': self.config.num_classes,
                'training': training  # Save training mode
            }
            self._save_trial(trial_number)
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise OptimizationError(f"Model saving failed: {str(e)}")

    def _train_and_evaluate(self, trainer: Trainer, train_loader: DataLoader,
                          val_loader: DataLoader) -> float:
        """Train model and return best validation score"""
        optimizer = torch.optim.AdamW(
            trainer.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=len(train_loader) * self.config.num_epochs
        )
        
        best_score = 0.0
        self.epoch_pbar.reset()
        
        epoch_bar = self.progress.init_epoch_bar(self.config.num_epochs, mode='optimize')
        
        try:
            for epoch in range(self.config.num_epochs):
                if self.epoch_pbar:
                    self.epoch_pbar.set_description(
                        f'[Epoch: {epoch+1}/{self.config.num_epochs}]'
                    )
                
                # Train and evaluate first
                loss = trainer.train_epoch(train_loader, optimizer, scheduler)
                score, _ = trainer.evaluate(val_loader)
                best_score = max(best_score, score)
                
                if self.epoch_pbar:
                    self.epoch_pbar.set_postfix_str(f"loss={loss:.4f} score={score:.4f}")
                    self.epoch_pbar.update(1)
                    self.epoch_pbar.refresh()
                
                # Update progress
                self.progress.update_epoch(
                    epoch, 
                    self.config.num_epochs,
                    {'loss': f'{loss:.4f}', 'score': f'{score:.4f}'}
                )
            
            return best_score
            
        finally:
            if self.epoch_pbar:
                self.epoch_pbar.refresh()
                print("", flush=True)

    def __del__(self):
        """Ensure progress bars are cleaned up"""
        self._cleanup_progress_bars()
        if self.current_model is not None:
            del self.current_model
            torch.cuda.empty_cache()

    def _save_trial(self, trial_number: int):
        """Save trial information"""
        if not self.best_model_info:
            return
            
        model_path = self.config.best_trials_dir / f"trial_{trial_number}_model.pt"
        trial_path = self.config.best_trials_dir / f"best_trial_{self.config.study_name}.pt"
        self.config.best_trials_dir.mkdir(exist_ok=True, parents=True)
        
        # Save model weights and config
        torch.save(self.best_model_info, model_path)
        
        # Save complete trial information
        trial_info = {
            'params': self.best_model_info['config']['classifier_config'],
            'value': self.best_score,
            'number': trial_number,
            'model_path': str(model_path)
        }
        torch.save(trial_info, trial_path)
        
        print("", flush=True)  # Add newline before logging
        logger.info("Saved best model (score: %.4f) to %s", self.best_score, str(model_path))
        logger.info("Saved best trial info to %s", str(trial_path))
        logger.info("Trial %d finished with score: %.4f", trial_number, self.best_score)

    def get_best_trial_path(self) -> Path:
        """Get path to best trial file"""
        return self.config.best_trials_dir / f"best_trial_{self.config.study_name}.pt"

def create_study(name: str, random_seed: Optional[int] = None) -> optuna.Study:
    """Create Optuna study"""
    # Generate random seed if none provided
    if (random_seed is None):
        import time
        random_seed = int(time.time())
    
    sampler = TPESampler(
        n_startup_trials=10,
        n_ei_candidates=24,
        multivariate=True,
        seed=random_seed
    )
    
    pruner = HyperbandPruner(
        min_resource=1,
        max_resource=10,
        reduction_factor=3
    )
    
    return optuna.create_study(
        study_name=name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

def run_optimization(config: ModelConfig) -> None:
    """Run hyperparameter optimization"""
    logger.info("Starting optimization")
    logger.info(f"Number of trials: {config.n_trials}")
    
    # Create optimizer and initialize progress bars
    optimizer = ModelOptimizer(config)
    optimizer.initialize_progress_bars()
    
    def progress_callback(study: optuna.Study, trial: optuna.Trial) -> None:
        """Single source of truth for progress updates"""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            # Update trial progress
            if optimizer.trial_pbar:
                best_value = study.best_value if study.best_value is not None else float('-inf')
                optimizer.trial_pbar.update(1)
                optimizer.trial_pbar.set_postfix({'best_score': f'{best_value:.4f}'})
                optimizer.trial_pbar.refresh()
            
            # Reset epoch bar for next trial
            if optimizer.epoch_pbar:
                optimizer.epoch_pbar.reset()
                optimizer.epoch_pbar.refresh()
    
    try:
        study = create_study(config.study_name, config.random_seed)
        
        # Run optimization with progress callback
        study.optimize(
            optimizer.objective,
            n_trials=config.n_trials,
            callbacks=[progress_callback],
            show_progress_bar=False,
            catch=(Exception,)  # Catch all exceptions to ensure proper cleanup
        )
        
        logger.info(f"Best trial score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        raise
    finally:
        # Clean up progress bars
        optimizer._cleanup_progress_bars()
        # Move cursor to bottom
        print("\n", flush=True)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BERT Classifier Optimization')
    
    # Add model configuration arguments
    ModelConfig.add_argparse_args(parser)
    
    # Remove study_name from study_group since it's already in ModelConfig
    """
    # Add study configuration arguments
    study_group = parser.add_argument_group('Study Configuration')
    study_group.add_argument('--study_name', type=str, default='bert_optimization',
                           help='Name for the optimization study')
    """
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Add study_name to config
    config = ModelConfig.from_args(args)
    config.study_name = args.study_name
    run_optimization(config)