# BERT Local Development Environment

A robust framework for training, fine-tuning, and optimizing BERT-based models with advanced architecture support and hyperparameter optimization.

## Project Structure

```
bert_local_dev/
├── src/
│   ├── models/              # Model architectures
│   │   ├── model.py        # Base BERT classifier with PlaneResNet
│   │   └── bert_classifier.py
│   ├── training/           # Training components
│   │   ├── trainer.py      # Training loop and evaluation
│   │   └── dataset.py      # Dataset handling
│   ├── optimize/           # Hyperparameter optimization
│   │   └── optimize.py     # Optuna-based optimization
│   └── config/            # Configuration management
├── all-MiniLM-L6-v2/      # Model files
│   ├── data_config.json   # Dataset configuration
│   └── vocab.txt         # Model vocabulary
└── nlp_env.yml           # Conda environment
```

## Key Features

- **Advanced Architectures**:
  - Standard BERT classifier with configurable layers
  - PlaneResNet architecture for improved performance
  - Flexible pooling strategies (CLS token or mean pooling)

- **Hyperparameter Optimization**:
  - Optuna-based optimization with multiple samplers
  - Support for TPE, CMA-ES, and random sampling
  - Automated trial pruning and early stopping
  - Parallel optimization support

- **Training Features**:
  - Configurable learning schedules
  - Multiple regularization options
  - Progress tracking and checkpointing
  - Comprehensive metrics and evaluation

## Environment Setup

```bash
# Create and activate environment
conda env create --file nlp_env.yml
conda activate nlp_env

# Optional: Recreate from scratch
conda env remove --name nlp_env -y && conda env create --file nlp_env.yml
```

## Model Configuration

The system supports two main classifier architectures:

### Standard Architecture
```python
classifier_config = {
    'architecture_type': 'standard',
    'num_layers': 2,
    'hidden_dim': 256,
    'activation': 'gelu',
    'regularization': 'dropout',
    'dropout_rate': 0.1,
    'cls_pooling': True
}
```

### PlaneResNet Architecture
```python
classifier_config = {
    'architecture_type': 'plane_resnet',
    'num_planes': 8,
    'plane_width': 128,
    'cls_pooling': True
}
```

## Running Optimization

```bash
python -m src.optimize.optimize \
    --model-name "all-MiniLM-L6-v2" \
    --data-file "path/to/data.csv" \
    --n-trials 100 \
    --study-name "bert_optimization" \
    --metric f1
```

## Dataset Configuration

The `data_config.json` file specifies training datasets with weights and line counts. Example format:

```json
{
    "name": "dataset_name",
    "lines": 10000,
    "weight": 1
}
```

## Environment Variables

Configured in nlp_env.yml:
- `TOKENIZERS_PARALLELISM`: Controls tokenizer parallelism
- `PYTORCH_ENABLE_MPS_FALLBACK`: Optional MPS fallback for Apple Silicon
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO`: Optional memory management for MPS

## Development Best Practices

1. Use the provided model configurations for consistent results
2. Monitor optimization trials with progress bars
3. Leverage early stopping for efficient training
4. Use checkpointing for long-running optimizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.