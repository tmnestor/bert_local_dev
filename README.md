# BERT Text Classification Framework

Advanced BERT fine-tuning framework featuring Population Based Training (PBT) and comprehensive hyperparameter optimization.

## Key Features

### Population Based Training (PBT)
- Dynamic hyperparameter adaptation during training
- Population of 4 parallel training runs 
- Adaptive strategies:
  - Exploit: Copy parameters from better performers (50% chance)
  - Explore: Perturb parameters by random factors (0.8-1.2)
- Parameters optimized:
  ```python
  OPTIMIZED_PARAMS = [
      "lr",                # Learning rate
      "weight_decay",     # L2 regularization
      "dropout_rate",     # Dropout probability
      "warmup_ratio"      # Learning rate warmup
  ]
  ```

### Hyperparameter Search Space
```python
SEARCH_SPACE = {
    'batch_size': [8, 16, 32, 64],          # Memory-aware
    'num_hidden_layers': (1, 4),            # Architecture depth
    'dropout_rate': (0.1, 0.6),            # Regularization
    'lr': (1e-6, 5e-3),                    # Log scale
    'weight_decay': (1e-8, 1e-2),          # Log scale
    'warmup_ratio': (0.0, 0.3),            # LR schedule
    'activation': ['relu', 'gelu', 'silu', 'tanh']
}
```

### Early Stopping Intelligence
- Multiple stopping criteria:
  - No improvement patience (adaptive)
  - Performance regression detection (20% drop)
  - Negative trend analysis
  - Moving average smoothing
  - Minimum epochs enforcement

## Usage

### Basic Training
```bash
python -m src.training.train \
    --data_file "datafile_name.csv" \
    --output_root "/path/to/output" \
    --num_epochs 10 \
    --batch_size 32 \
    --verbosity 2
```

### Hyperparameter Optimization
```bash
python -m src.tuning.optimize \
    --data_file "datafile_name.csv" \
    --output_root "/path/to/output" \
    --study_name "bert_opt" \
    --n_trials 50 \
    --device cpu \
    --verbosity 2
```
### Model Evaluation
```bash
python -m src.evaluation.evaluator \
    --data_file "datafile_name.csv" \
    --output_root "/path/to/output" \
    --best_model "best_trials/bert_classifier.pth"
```
### Configuration (config.yml)
```yaml
model:
  max_seq_len: 64
  batch_size: 32
  num_epochs: 10
  device: cpu
  metric: f1

classifier:
  bert_hidden_size: 384
  hidden_dims: [412, 220, 118]
  dropout_rate: 0.35
  activation: gelu

optimizer:
  optimizer_choice: adamw
  lr: 0.003
  weight_decay: 0.005
  warmup_ratio: 0.165
  betas: (0.656, 0.999)
```

## Advanced Features

### Population Based Training
The PBT implementation maintains a population of 4 models training in parallel:
1. Models in bottom 20% of population trigger adaptation
2. 50% chance to either:
   - Exploit: Copy params from better model
   - Explore: Perturb current params
3. Adaptation occurs during training, not between trials
4. Maintains diverse population of good solutions

### Memory Management
```python
class MemoryManager:
    """Dynamic batch size adjustment based on memory usage"""
    memory_threshold = 0.85  # 85% memory usage threshold
    batch_size_limits = {
        "min": 8,
        "max": 64,
        "current": 32
    }
```

### Early Stopping Configuration
```python
early_stopping = {
    'patience': max(3, min(8, trial.number // 2)),
    'min_epochs': max(5, num_epochs // 4),
    'improvement_threshold': 0.001,
    'smoothing_window': 3,
    'trend_window': 5
}
```

### Logging Levels
- 0: Minimal (warnings only)
- 1: Normal (info + progress)
- 2: Debug (detailed paths, sizes, configs)

### Data Format
Required CSV structure:
```csv
text,category
"Sample text 1","class_a"
"Sample text 2","class_b"
```

## Project Structure
```
src/
├── __init__.py
├── config                # Configuration management
│   ├── __init__.py
│   ├── base_config.py
│   ├── config.py
│   └── defaults.py
├── data_utils            # Data handling and splits
│   ├── __init__.py
│   ├── dataset.py
│   ├── loaders.py
│   ├── splitter.py
│   └── validation.py
├── evaluation            #model evaluation
│   └── evaluator.py
├── models                # BERT classifier architecture
│   ├── __init__.py
│   └── model.py
├── training              # Training implementation
│   ├── __init__.py
│   ├── train.py
│   └── trainer.py
├── tuning                # PBT and optimization
│   ├── __init__.py
│   └── optimize.py
└── utils                 # Common utilities
    ├── logging_manager.py
    ├── metrics.py
    ├── model_loading.py
    └── train_utils.py
```
