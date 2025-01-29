"""Default configuration values."""
import torch
import os
import yaml
from pathlib import Path

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Try project root first, then environment variable
        config_paths = [
            Path.cwd() / 'config.yml',
            Path.cwd() / 'directories.yml',  # Backward compatibility
            Path(os.environ.get('BERT_CONFIG', 'config/config.yml'))
        ]
        config_path = next((p for p in config_paths if p.exists()), None)
        if not config_path:
            raise FileNotFoundError("No configuration file found")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Convert paths to Path objects
    config['output_root'] = Path(config['output_root'])
    
    # Convert directory paths
    if 'dirs' in config:
        config['dirs'] = {k: str(v) for k, v in config['dirs'].items()}
    
    # Convert model paths
    if 'model_paths' in config:
        config['model_paths'] = {k: str(v) for k, v in config['model_paths'].items()}
    
    return config

# Load configuration
CONFIG = load_config()

# Export configuration sections
DIR_DEFAULTS = {
    'output_root': CONFIG['output_root'],
    'dirs': CONFIG['dirs']
}

DATA_DEFAULTS = CONFIG.get('data', {'default_file': 'bbc-text.csv'})  # Add this line

MODEL_PATHS = CONFIG['model_paths']

MODEL_DEFAULTS = CONFIG['model']

CLASSIFIER_DEFAULTS = CONFIG['classifier']

# Optimization search space remains hardcoded for now
OPTIM_SEARCH_SPACE = {
    'batch_size': [16, 32, 64],
    'num_layers': (1, 4),
    'hidden_dim': [32, 64, 128, 256, 512, 1024],
    'activation': ['relu', 'gelu', 'elu', 'leaky_relu', 'selu', 
                  'mish', 'swish', 'hardswish', 'tanh', 'prelu'], 
    'dropout_rate': (0.1, 0.5),
    'learning_rate': (1e-5, 1e-3),  # log scale
    'weight_decay': (1e-8, 1e-3),   # log scale
    'warmup_ratio': (0.0, 0.2)
}
