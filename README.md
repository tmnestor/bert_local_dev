# BERT Text Classification Framework

A production-grade framework for fine-tuning BERT models on text classification tasks, featuring automated optimization and comprehensive evaluation.

## Purpose

This framework addresses three key challenges in BERT-based text classification:

1. **Optimization Complexity** - Automated hyperparameter tuning using Optuna
2. **Evaluation Rigor** - Comprehensive metrics and analysis tools
3. **Architecture Flexibility** - Support for both standard and innovative classifier architectures

## Structure

```
src/
├── config/          # Configuration management 
├── models/          # Model architectures
├── training/        # Training pipeline
├── tuning/         # Hyperparameter optimization 
├── evaluation/     # Evaluation tools
├── data_utils/     # Data handling utilities
└── utils/          # Common utilities
```

Key Components:
- **data_utils/**: Dataset management and validation
- **models/**: BERT classifier implementations
- **tuning/**: Optuna-based optimization
- **evaluation/**: Metrics and analysis tools

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
# Basic usage with direct path
python -m src.training.train \
    --data_file "/Users/tod/BERT_TRAINING/data/bbc-text.csv" \
    --num_epochs 10 \
    --batch_size 32 \
    --bert_model_name "/Users/tod/BERT_TRAINING/bert_encoder" \
    --output_root "/Users/tod/BERT_TRAINING"

# python -m src.training.train \
#     --data_file "/Users/tod/BERT_TRAINING/data/bbc-text.csv" \
#     --bert_model_name "/Users/tod/BERT_TRAINING/bert_encoder" \
#     --num_epochs 10 \
#     --batch_size 32

# # Usage with directories.yml configuration
# python -m src.training.train \
#     --data_file "/Users/tod/BERT_TRAINING/data/bbc-text.csv" \
#     --output_root "/Users/tod/BERT_TRAINING" \
#     --num_epochs 10 \
#     --batch_size 32 \
#     --device cpu
```

### 3. Optimize
```bash
python -m src.tuning.optimize \
    --data_file "/Users/tod/BERT_TRAINING/data/bbc-text.csv" \
    --n_trials 5 \
    --study_name "bert_opt" \
    --batch_size 32 \
    --device cpu

# python -m src.tuning.optimize \
#     --data_file "data/dataset.csv" \
#     --n_trials 100 \
#     --n_experiments 3 \
#     --study_name "bert_opt" \
#     --sampler tpe \
#     --metric f1 \
#     --device cuda \
#     --timeout 36000

# python -m src.tuning.optimize \
#     --data_file "data/dataset.csv" \
#     --output_root /custom/output/path \
#     --bert_encoder_path /path/to/bert/encoder \
#     --n_trials 50 \
#     --study_name "bert_opt" \
#     --sampler tpe \
#     --metric f1 \
#     --device cuda \
#     --batch_size 32 \
#     --max_seq_len 128 \
#     --seed 42
```

### 4. Evaluate
```bash
python -m src.evaluation.evaluator \
    --data_file "/Users/tod/BERT_TRAINING/data/bbc-text.csv" \
    --best_model "/Users/tod/BERT_TRAINING/best_trials/bert_classifier.pth" \
    --output_root "/Users/tod/BERT_TRAINING" \
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

### Directory Structure
Place `directories.yml` in the project root:

```yaml
output_root: /path/to/outputs
dirs:
  best_trials: best_trials
  checkpoints: checkpoints
  evaluation: evaluation_results
  logs: logs
  data: data
  models: models

model_paths:
  bert_encoder: /absolute/path/to/bert/encoder  # or relative: models/bert_encoder
```

Configuration precedence:
1. Command line arguments (highest priority)
2. Project root `directories.yml`
3. Environment variable `BERT_DIR_CONFIG`
4. Default values (lowest priority)

Override using command line:
```bash
python -m src.training.train \
    --output_root /custom/output/path \
    --bert_encoder_path /path/to/bert/encoder \
    --dir_config /path/to/custom/directories.yml
```

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

## Best Practices

1. Data Preparation
   - Clean and validate inputs
   - Use stratified splits
   - Check class balance

2. Training
   - Start with standard architecture
   - Enable early stopping
   - Monitor validation metrics

3. Optimization
   - Set reasonable trial budget
   - Define informed parameter ranges
   - Use TPE sampler

4. Evaluation
   - Check multiple metrics
   - Analyze error patterns
   - Save predictions for analysis

## Troubleshooting

1. Memory Issues:
   ```python
   config.batch_size = 16  # Reduce batch size
   config.max_seq_len = 128  # Limit sequence length
   ```

2. Performance Issues:
   ```python
   config.learning_rate = 2e-5  # Adjust learning rate
   config.num_epochs = 10  # Increase epochs
   ```

## Contributing

1. Fork repository
2. Create feature branch
3. Add tests
4. Submit pull request

## License

MIT License - See LICENSE file