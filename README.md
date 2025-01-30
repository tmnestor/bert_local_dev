# BERT Text Classification Framework

A robust framework for BERT-based text classification with automated optimization, comprehensive evaluation, and flexible configuration management.

## Key Features

- **Robust Configuration System**: Multi-level configuration with validation and inheritance
- **Advanced Optimization**: Optuna-based tuning with multiple sampling strategies
- **Flexible Architecture**: Configurable classifier head with dynamic layer sizing
- **Comprehensive Evaluation**: Multi-metric assessment and result analysis
- **Production-Ready**: Error handling, logging, and progress tracking

## Project Structure

```
src
├── __init__.py
├── config
│   ├── __init__.py
│   ├── base_config.py
│   ├── config.py
│   └── defaults.py
├── data_utils
│   ├── __init__.py
│   ├── dataset.py
│   ├── loaders.py
│   ├── splitter.py
│   └── validation.py
├── evaluation
│   └── evaluator.py
├── models
│   ├── __init__.py
│   └── model.py
├── training
│   ├── __init__.py
│   ├── train.py
│   └── trainer.py
├── tuning
│   ├── __init__.py
│   └── optimize.py
└── utils
    ├── data_splitter_old.py
    ├── logging_manager.py
    ├── metrics.py
    ├── model_loading.py
    └── train_utils.py
```

Key Components:
- **config/**: Manages configuration settings with validation and inheritance.
  - `base_config.py`: Base configuration class.
  - `config.py`: Handles model and training configurations.
  - `defaults.py`: Contains default configuration values.
- **data_utils/**: Handles data processing and management.
  - `dataset.py`: PyTorch dataset implementation for text classification.
  - `loaders.py`: Utilities for loading data.
  - `splitter.py`: Manages data splitting into training, validation, and test sets.
  - `validation.py`: Tools for data validation.
- **models/**: Contains model architectures.
  - `model.py`: Implements the BERT-based classifier.
- **training/**: Manages the training pipeline.
  - `train.py`: Script to initiate training.
  - `trainer.py`: Implements the training loop and evaluation during training.
- **tuning/**: Handles hyperparameter optimization using Optuna.
  - `optimize.py`: Script to perform hyperparameter tuning.
- **evaluation/**: Tools for evaluating model performance.
  - `evaluator.py`: Implements model evaluation metrics and analysis.
- **utils/**: Common utilities used across the project.
  - `logging_manager.py`: Configures logging.
  - `metrics.py`: Defines evaluation metrics.
  - `model_loading.py`: Utilities for loading models.
  - `train_utils.py`: Helper functions for training.

## Quick Usage

### 1. Setup
```bash
# Create environment
conda env create -f nlp_env.yml

# Prepare BERT model
python scripts/download_BERT.py
```

### 2. Train
```bash
# Basic usage (data must be in /Users/tod/BERT_TRAINING/data/bbc-text.csv)
python -m src.training.train \
    --data_file "bbc-text.csv" \
    --output_root "/Users/tod/BERT_TRAINING" \
    --num_epochs 10 \
    --batch_size 32

```

### 3. Optimize
```bash
python -m src.tuning.optimize \
    --data_file "bbc-text.csv" \
    --output_root "/Users/tod/BERT_TRAINING" \
    --n_trials 5 \
    --study_name "bert_opt" \
    --batch_size 32 \
    --device cpu

```

### 4. Evaluate
```bash
# Evaluate model (paths are relative to output_root)
python -m src.evaluation.evaluator \
    --data_file "bbc-text.csv" \
    --output_root "/Users/tod/BERT_TRAINING" \
    --best_model "best_trials/bert_classifier.pth" \
    --device cpu
```

## Data Format

Required CSV structure:
```csv
text,category
"Sample text 1","class_a"
"Sample text 2","class_b"
```

Requirements:
- UTF-8 encoding
- No missing values
- Headers: "text", "category"

## Configuration

### Configuration File
Place `config.yml` in the project root:

```yaml
# Directory structure
output_root: /path/to/outputs
dirs:
  best_trials: best_trials
  checkpoints: checkpoints
  evaluation: evaluation_results
  logs: logs
  data: data
  models: models

# Model paths and configuration
model_paths:
  bert_encoder: /path/to/bert/encoder

model:
  max_seq_len: 64
  batch_size: 32
  num_epochs: 10
  learning_rate: 2e-5
```

Configuration precedence:
1. Command line arguments (highest priority)
2. Project root `config.yml`
3. Environment variable `BERT_CONFIG`
4. Default values (lowest priority)

### Model Settings
```python
config = {
    'architecture_type': 'standard',  # or 'plane_resnet'
    'num_layers': 2,
    'hidden_dim': 256,
    'dropout_rate': 0.1
}
```

### Optimization Space
```python
search_space = {
    'learning_rate': (1e-5, 1e-3),
    'batch_size': [16, 32, 64],
    'num_layers': (1, 4),
    'hidden_dim': [128, 256, 512]
}
```