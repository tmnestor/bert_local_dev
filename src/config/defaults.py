"""Default configuration values."""
import os
from pathlib import Path

import yaml


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Try project root first, then environment variable
        config_paths = [
            Path.cwd() / 'config.yml',
            # Path(os.environ.get('BERT_CONFIG', 'config/config.yml'))
        ]
        config_path = next((p for p in config_paths if p.exists()), None)
        if not config_path:
            raise FileNotFoundError("No configuration file found")
    
    with open(config_path, encoding='utf-8') as f:
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

# Update CLASSIFIER_DEFAULTS to remove 'standard' nesting
CLASSIFIER_DEFAULTS = {
    'hidden_dim': [256, 218],
    'dropout_rate': 0.1
}

# Update OPTIM_SEARCH_SPACE to use dynamic hidden dimensions
OPTIM_SEARCH_SPACE = {
    'batch_size': [16, 32, 64],
    'num_hidden_layers': (1, 4),  # Now controlling number of layers directly
    'dropout_rate': (0.1, 0.5),
    'learning_rate': (1e-5, 1e-3),  # log scale
    'weight_decay': (1e-8, 1e-3),   # log scale
    'warmup_ratio': (0.0, 0.2)
}

# Remove hidden_dims from search space since it's now dynamically generated
