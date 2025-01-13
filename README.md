# BERT Local Development Environment

A development framework for BERT model optimization featuring standard and PlaneResNet architectures with Optuna-based hyperparameter tuning.

## Project Structure

```
bert_local_dev/
├── src/
│   ├── models/
│   │   ├── model.py          # BERTClassifier with PlaneResNet implementation
│   │   └── bert_classifier.py # High-level classifier interface
│   ├── training/
│   │   ├── trainer.py        # Training loop and evaluation logic
│   │   └── dataset.py        # TextClassificationDataset implementation
│   ├── optimize/
│   │   └── optimize.py       # Optuna optimization framework
│   └── config/              # Configuration management
├── all-MiniLM-L6-v2/       # Model files
└── nlp_env.yml             # Conda environment specification
```

## Usage

Example optimization command:
```bash
python -m src.optimize.optimize \
    --bert_model_name "all-MiniLM-L6-v2" \
    --data_file "data/data.csv" \
    --n_trials 3 \
    --study_name "bert_optimization" \
    --metric f1 \
    --num_classes 2
```

## Data Format

The model expects input data in CSV format with specific column names:

```csv
text,category
"The economy grew by 2.5% last quarter","economics"
"Scientists discover new species in Amazon","science"
"Team wins championship in overtime","sports"
```

Required columns:
- `text`: Contains the input text to be classified
- `category`: Contains the target class/label

Notes:
- Column names must be exactly "text" and "category" (case-sensitive)
- Categories are automatically encoded using scikit-learn's LabelEncoder
- Number of unique categories must match `--num_classes` parameter
- CSV should use UTF-8 encoding
- Text can contain quotes and commas (proper CSV escaping required)

## Architectures

This framework provides two specialized classifier architectures for BERT embeddings:

### Standard Classifier Architecture

A configurable deep neural network that processes BERT embeddings through sequential layers:

```python
{
    'architecture_type': 'standard',
    'num_layers': 1-4,         # Number of hidden layers
    'hidden_dim': [32-1024],   # Dimension of hidden layers
    'activation': ['gelu', 'relu'],  # Activation function
    'regularization': ['dropout', 'batchnorm'],  # Regularization type
    'dropout_rate': 0.1-0.5,   # Dropout probability
    'cls_pooling': True/False  # Use CLS token vs mean pooling
}
```

#### Key Features
- **Progressive Dimensionality**: Smooth dimension transitions between layers
- **Flexible Depth**: Configurable number of hidden layers (1-4)
- **Advanced Activations**: Support for modern activation functions
- **Dual Regularization**: Choice of dropout or batch normalization
- **Adaptive Pooling**: CLS token or mean pooling strategies

#### Architecture Details
1. Input layer processes BERT embeddings (768 dimensions)
2. Configurable hidden layers with:
   - Linear transformation
   - Activation function (GELU/ReLU)
   - Regularization (Dropout/BatchNorm)
   - Progressive dimension reduction
3. Final classification layer

#### Advantages
- Simple yet effective architecture
- Well-suited for traditional text classification
- Strong regularization options
- Memory efficient
- Easy to interpret

#### Recommended Settings
- Single layer for simple tasks
- 2-3 layers for complex tasks
- Higher dropout (0.3-0.5) for large datasets
- BatchNorm for deeper configurations
- Start with smaller hidden dimensions (256-512)

### PlaneResNet Architecture

The PlaneResNet architecture is an innovative classifier design that processes BERT embeddings through parallel residual planes:

```python
{
    'architecture_type': 'plane_resnet',
    'num_planes': 4-16,        # Number of parallel residual blocks
    'plane_width': [32, 64, 128, 256],  # Width of each plane
    'cls_pooling': True/False  # Use CLS token vs mean pooling
}
```

#### Key Features
- **Parallel Processing**: Multiple ResNet planes process features simultaneously
- **Residual Connections**: Each plane uses skip connections for better gradient flow
- **Plane Width**: Controls the dimensionality of feature processing
- **Adaptive Pooling**: Supports both CLS token and mean pooling strategies

#### Architecture Details
1. Input projection layer maps BERT embeddings to plane width
2. N parallel ResNet planes process features independently
3. Each plane contains:
   - Two linear transformations
   - Batch normalization
   - ReLU activation
   - Skip connection
4. Final output layer combines plane outputs for classification

#### Advantages
- Better feature extraction through parallel processing
- Reduced vanishing gradient problem via skip connections
- Flexible capacity scaling through num_planes parameter
- Efficient parameter usage compared to deep sequential networks

#### Recommended Settings
- Start with 4-8 planes for small datasets
- Increase planes (8-16) for complex tasks
- Use larger plane widths (128, 256) for rich feature spaces
- Enable cls_pooling for sentence classification tasks

## Optimization Features

- Automated hyperparameter search using Optuna
- Supports multiple sampling strategies:
  - TPE (default)
  - Random
  - CMA-ES
  - QMC (Sobol)
- Early stopping with HyperbandPruner
- Progress tracking with tqdm
- Best model state saving
- Multi-experiment support with seeds

## Environment Setup

```bash
# Create environment
conda env create -f nlp_env.yml

# Optional: Clean reinstall
conda env remove -n nlp_env -y && conda env create -f nlp_env.yml
```

## Training Configuration

Key parameters:
- `learning_rate`: 1e-5 to 1e-3 (log scale)
- `weight_decay`: 1e-8 to 1e-3 (log scale)
- `batch_size`: [16, 32, 64]
- `warmup_ratio`: 0.0 to 0.2
- `num_epochs`: Configurable, default 10

## Metrics

Supported evaluation metrics:
- F1 score (macro)
- Accuracy

Progress is tracked using both metrics, with one selected as primary for optimization.

## Best Practices

1. Start with a small number of trials (3-5) for testing
2. Use TPE sampler for best results
3. Enable checkpointing for long runs
4. Monitor early stopping behavior
5. Consider architecture-specific parameter ranges

## Dependencies

See nlp_env.yml for complete list. Key requirements:
- PyTorch
- Transformers
- Optuna
- scikit-learn
- pandas