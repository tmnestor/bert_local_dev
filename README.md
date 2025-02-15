# BERT Text Classification Framework

A production-ready BERT fine-tuning framework for text classification tasks, featuring comprehensive configuration management, evaluation metrics, and optimization capabilities.

## Core Features

### 1. Configuration Management
- YAML-based configuration with anchor support
- Environment-aware path resolution
- Type-safe configuration classes
- Command-line argument integration
- Configuration validation

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
    'activation': ["relu", "gelu", "silu", "elu", "tanh", "leaky_relu", "prelu"]
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
    --output_root "/Users/tod/BERT_TRAINING" \
    --num_epochs 5 \
    --batch_size 32 \
    --verbosity 1 \
    --max_seq_len 128
```

### Hyperparameter Optimization
```bash
python -m src.tuning.optimize \
  --output_root /Users/tod/BERT_TRAINING \
  --study_name bert_opt \
  --n_trials 10 \
  --max_seq_len 64 \
  --verbosity 2
```
### Model Evaluation
```bash
python -m src.evaluation.evaluator \
    --output_root /Users/tod/BERT_TRAINING \
    --best_model best_model.pt
```

### Model Prediction
```bash
python -m src.prediction.predict \
    --output_root /Users/tod/BERT_TRAINING \
    --best_model best_model.pt \
    --data_file test.csv \
    --output_file predictions.csv
```

#### Model Inspection
```bash
python -m src.utils.model_info \
    --output_root "/Users/tod/BERT_TRAINING" \
    --model best_model_bert_opt.pt
```

### Configuration Override Precedence
Values are resolved in this order (highest to lowest priority):

1. Command-line arguments
2. config.yml values
3. Default values in dataclasses


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

## Advanced Hyperparameter Tuning Features

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
├── config
│   ├── __init__.py
│   ├── base_config.py
│   └── configuration.py
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
├── prediction
│   ├── __init__.py
│   └── predict.py
├── training
│   ├── __init__.py
│   ├── train.py
│   └── trainer.py
├── tuning
│   ├── __init__.py
│   └── optimize.py
└── utils
    ├── logging_manager.py
    ├── metrics.py
    ├── model_loading.py
    └── train_utils.py


output_root/
├── bert_encoder
│   ├── 1_Pooling
│   │   └── config.json
│   ├── README.md
│   ├── config.json
│   ├── config_sentence_transformers.json
│   ├── model.safetensors
│   ├── modules.json
│   ├── sentence_bert_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── best_trials
│   ├── best_model.pt
│   └── best_model_bert_opt_exp0.pt
├── data
│   ├── Untitled-1.ipynb
│   ├── bbc-text-10.csv
│   ├── bbc-text.csv
│   ├── label_encoder.joblib
│   ├── test.csv
│   ├── train.csv
│   └── val.csv
├── evaluation_results
│   ├── all_predictions.csv
│   ├── classification_report_heatmap.png
│   ├── confidence_analysis.csv
│   ├── confusion_matrix.csv
│   ├── confusion_matrix.png
│   ├── cv_metrics.csv
│   ├── error_analysis.csv
│   ├── error_confidence_dist.png
│   ├── error_confusion_patterns.png
│   ├── error_length_dist.png
│   ├── error_metrics.csv
│   ├── error_wordclouds.png
│   ├── fold_1
│   │   ├── confusion_matrix.png
│   │   ├── error_analysis.csv
│   │   └── predictions.csv
│   ├── fold_2
│   │   ├── confusion_matrix.png
│   │   ├── error_analysis.csv
│   │   └── predictions.csv
│   ├── fold_3
│   │   ├── confusion_matrix.png
│   │   ├── error_analysis.csv
│   │   └── predictions.csv
│   ├── fold_4
│   │   ├── confusion_matrix.png
│   │   ├── error_analysis.csv
│   │   └── predictions.csv
│   ├── fold_5
│   │   ├── confusion_matrix.png
│   │   ├── error_analysis.csv
│   │   └── predictions.csv
│   ├── fold_6
│   │   ├── confusion_matrix.png
│   │   ├── error_analysis.csv
│   │   └── predictions.csv
│   ├── fold_7
│   │   ├── confusion_matrix.png
│   │   ├── error_analysis.csv
│   │   └── predictions.csv
│   ├── predictions.csv
│   └── sklearn_classification_report.csv
├── logs
│   ├── evaluation_20250209_101556.log
│   ├── evaluation_20250209_102302.log
│   ├── evaluation_20250210_095748.log
│   ├── prediction_20250209_101639.log
│   ├── prediction_20250209_102235.log
│   ├── training_20250209_100650.log
│   └── training_20250209_101804.log
└── predictions
    └── predictions.csv
```
