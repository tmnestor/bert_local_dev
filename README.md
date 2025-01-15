# BERT Text Classification Framework

A production-ready framework for BERT-based text classification with automated optimization, comprehensive evaluation, and flexible architectures.

## Features

### Core Capabilities
- 🚀 Two optimized classifier architectures (Standard and PlaneResNet)
- 📊 Automated hyperparameter optimization with Optuna
- 📈 Comprehensive evaluation metrics and reporting
- 🔄 Robust data management and validation
- 📝 Structured logging and error handling

### Architecture Support
- **Standard Classifier**: Configurable deep neural network
- **PlaneResNet**: Innovative parallel residual architecture
- **BERT Integration**: Efficient handling of BERT embeddings
- **Flexible Pooling**: CLS token or mean pooling strategies

### Data Management
- Automated train/val/test splitting (60/20/20)
- Data leakage prevention
- Persistent storage of splits
- Label encoding and validation

## Installation

```bash
# Create new environment
conda env create -f nlp_env.yml

# Update existing environment
conda env update -f nlp_env.yml --prune
```

## Project Structure

```
bert_local_dev/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── base_config.py      # Base configuration class
│   │   └── config.py           # Model and evaluation configs
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py            # BERT classifier architectures
│   │   └── bert_classifier.py  # High-level classifier interface
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py           # Training script
│   │   ├── trainer.py         # Training logic
│   │   ├── dataset.py         # Dataset classes
│   │   └── validate.py        # Validation script
│   ├── tuning/
│   │   ├── __init__.py
│   │   └── optimize.py        # Hyperparameter optimization
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py       # Model evaluation
│   └── utils/
│       ├── __init__.py
│       ├── data_splitter.py   # Data management
│       ├── logging_manager.py # Logging setup
│       ├── metrics.py         # Evaluation metrics
│       └── train_utils.py     # Training utilities
├── tests/                     # Unit tests
├── logs/                      # Training logs
├── models/                    # Saved models
├── data/                      # Dataset storage
├── evaluation_results/        # Evaluation outputs
├── nlp_env.yml               # Environment specification
├── README.md                 # Project documentation
└── LICENSE                   # License file
```

### Key Components

- **src/config/**: Configuration management and validation
- **src/models/**: Model architectures and implementations
- **src/training/**: Training pipeline and utilities
- **src/tuning/**: Hyperparameter optimization framework
- **src/evaluation/**: Evaluation tools and metrics
- **src/utils/**: Common utilities and helpers

## Quick Start

### 1. Basic Training
```bash
python -m src.training.train \
    --data_file "data/dataset.csv" \
    --bert_model_name "bert-base-uncased" \
    --architecture "standard"
```

### 2. Hyperparameter Optimization
```bash
python -m src.tuning.optimize \
    --data_file "data/dataset.csv" \
    --n_trials 50 \
    --study_name "bert_opt" \
    --metric f1
```

### 3. Model Evaluation
```bash
python -m src.evaluation.evaluator \
    --best_model "models/best_model.pt" \
    --output_dir "evaluation_results"
```

## Architecture Details

### Standard Classifier
- Configurable number of layers
- Progressive dimension reduction
- Multiple activation functions
- Flexible regularization options

```python
config = {
    'architecture_type': 'standard',
    'num_layers': 2,
    'hidden_dim': 256,
    'activation': 'gelu',
    'regularization': 'dropout',
    'dropout_rate': 0.1
}
```

### PlaneResNet
- Parallel residual blocks
- Efficient feature processing
- Skip connections
- Batch normalization

```python
config = {
    'architecture_type': 'plane_resnet',
    'num_planes': 8,
    'plane_width': 128,
    'cls_pooling': True
}
```

## Data Format

Required CSV format:
```csv
text,category
"Sample text 1","class_a"
"Sample text 2","class_b"
```

Requirements:
- UTF-8 encoding
- Headers: "text" and "category" (exact names)
- No missing values
- Proper CSV escaping for quotes/commas

## Configuration

### Model Configuration
```python
config = ModelConfig(
    bert_model_name="bert-base-uncased",
    num_classes=3,
    batch_size=32,
    learning_rate=2e-5,
    num_epochs=10
)
```

### Evaluation Configuration
```python
config = EvaluationConfig(
    metrics=["accuracy", "f1", "precision", "recall"],
    output_dir="evaluation_results",
    batch_size=64
)
```

## Best Practices

### Training
1. Start with standard architecture for baseline
2. Use optimization for complex datasets
3. Monitor validation metrics
4. Enable early stopping

### Optimization
1. Set appropriate trial budget
2. Use TPE sampler for efficiency
3. Define reasonable parameter ranges
4. Save best configurations

### Evaluation
1. Use held-out test set
2. Consider multiple metrics
3. Analyze confusion matrix
4. Check confidence scores

## Troubleshooting

Common issues and solutions:

1. **Memory Issues**
   - Reduce batch size
   - Use gradient accumulation
   - Switch to CPU if needed

2. **Poor Performance**
   - Check learning rate
   - Increase training epochs
   - Verify data quality
   - Try different architecture

3. **Data Issues**
   - Verify CSV format
   - Check encoding
   - Validate label consistency

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

## License

MIT License - See LICENSE file for details

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{bert_classification_framework,
    title = {BERT Text Classification Framework},
    year = {2023},
    author = {Framework Authors},
    url = {https://github.com/username/repo}
}
```