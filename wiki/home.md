# BERT Text Classification Framework

## Purpose
A production-ready framework for training and deploying BERT-based text classifiers with:
- Automated hyperparameter optimization
- Cross-validation evaluation
- Detailed performance analysis
- Streamlined model deployment
- Batch prediction capabilities

## Core Features
- **Robust Training**: Multi-GPU support, gradient accumulation, mixed precision
- **Automated Optimization**: Bayesian optimization for hyperparameter tuning
- **Comprehensive Evaluation**: Cross-validation, confusion matrices, error analysis
- **Production Ready**: Model versioning, checkpoint management, inference API

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

See [[CLI Reference]] for complete usage details.