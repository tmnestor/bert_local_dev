# Command Line Interface Reference

## Training
Train a new BERT classifier model:
```bash
python -m src.train \
    --data_path data/train.csv \
    --output_dir outputs/experiment1 \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 2e-5 \
    --warmup_steps 500 \
    --max_length 128 \
    [--fp16] \
    [--gradient_accumulation 2] \
    [--class_weights balanced]
```

Key arguments:
- `--data_path`: Training dataset path (CSV)
- `--output_dir`: Directory for checkpoints and logs
- `--fp16`: Enable mixed precision training
- `--class_weights`: Handle class imbalance

## Hyperparameter Optimization
Optimize model hyperparameters using Bayesian optimization:
```bash
python -m src.optimize \
    --data_path data/train.csv \
    --n_trials 50 \
    --study_name bert_opt \
    --metric f1 \
    [--n_jobs 4] \
    [--pruner median] \
    [--timeout 72000]
```

Key arguments:
- `--n_trials`: Number of optimization trials
- `--metric`: Optimization metric (accuracy/f1/precision/recall)
- `--pruner`: Trial pruning strategy
- `--timeout`: Maximum optimization time in seconds

## Evaluation
Evaluate model performance with detailed analysis:
```bash
python -m src.evaluate \
    --best_model models/best_model_v1.pt \
    --test_data data/test.csv \
    --output_dir eval_results \
    [--n_folds 5] \
    [--confidence_threshold 0.8] \
    [--error_analysis]
```

Key arguments:
- `--best_model`: Path to model checkpoint
- `--n_folds`: Number of cross-validation folds
- `--confidence_threshold`: Filter predictions by confidence
- `--error_analysis`: Generate detailed error analysis

## Prediction
Run batch predictions:
```bash
python -m src.predict \
    --model_path models/best_model.pt \
    --input_file data/predict.csv \
    --output_file predictions.csv \
    [--batch_size 64] \
    [--confidence_threshold 0.5] \
    [--include_probabilities]
```

Key arguments:
- `--model_path`: Path to trained model
- `--confidence_threshold`: Minimum confidence for predictions
- `--include_probabilities`: Include prediction probabilities
- `--batch_size`: Batch size for inference