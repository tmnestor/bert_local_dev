# The Free Text Categoriser Framework Wiki

## Purpose
A framework for training, tuning and deploying a BERT-based text classifier with:
- Automated hyperparameter optimization
- Cross-validation evaluation
- Detailed performance analysis
- Streamlined model deployment
- Batch prediction capabilities

## Core Features
- **Automated Optimization**: Bayesian and Population Based optimization for hyperparameter tuning
- **Comprehensive Evaluation**: Cross-validation, confusion matrices, error analysis

## Quick Navigation
- [Configuration Guide](configuration.md)
- [CLI Reference] ()
- [Training Guide](training.md)
- [Hyperparameter Optimization](optimization.md)
- [Model Evaluation](evaluation.md)

## Quick Start
```bash
# Train a model
python -m src.train --data_path data/train.csv --output_dir outputs/my_classifier

# Optimize hyperparameters
python -m src.optimize --data_path data/train.csv --n_trials 50 --study_name my_study

# Evaluate model
python -m src.evaluate --best_model best_model_v1.pt --test_data data/test.csv

# Run predictions
python -m src.predict --model_path models/best_model.pt --input_file data/predict.csv
```