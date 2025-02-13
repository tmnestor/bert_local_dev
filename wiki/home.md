# BERT Text Classification Framework

## Overview
A production-ready framework for training and deploying BERT-based text classifiers, featuring Population Based Training (PBT) and comprehensive evaluation.

## Key Features
- **Intelligent Training**: Population Based Training with dynamic adaptation
- **Memory Management**: Dynamic batch sizing based on available memory
- **Advanced Evaluation**: Detailed error analysis and visualizations
- **Production Ready**: Robust logging, checkpointing, and inference

## Quick Navigation
- [[Installation Guide]]
- [[Configuration Guide]]
- [[Training Guide]]
- [[Hyperparameter Optimization]]
- [[Evaluation and Analysis]]
- [[API Reference]]

## Common Usage

### Basic Training
```bash
# Train a model
python -m src.train --data_path data/train.csv --output_dir outputs/my_classifier

# Optimize hyperparameters
python -m src.optimize --data_path data/train.csv --n_trials 50

# Evaluate model
python -m src.evaluate --best_model best_model_v1.pt --test_data data/test.csv

# Run predictions
python -m src.predict --model_path best_model.pt --input_file data/predict.csv
```