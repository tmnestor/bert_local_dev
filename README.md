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
python -m src.training.train \
    --data_file "data/your_data.csv" \
    --bert_model_name "./bert_encoder" \
    --batch_size 32
```

### 3. Optimize
```bash
python -m src.tuning.optimize \
    --data_file "data/your_data.csv" \
    --n_trials 50 \
    --study_name "bert_opt"
```

### 4. Evaluate
```bash
python -m src.evaluation.evaluator \
    --best_model "best_trials/best_model.pt" \
    --output_dir "evaluation_results"
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