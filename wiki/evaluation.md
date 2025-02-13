# Model Evaluation

## Evaluation Features

### Cross-Validation
- K-fold cross-validation (default k=7)
- Stratified sampling
- Comprehensive metrics per fold

### Metrics & Visualizations
1. **Performance Metrics**
   - Accuracy, F1-score, Precision, Recall
   - Per-class and macro-averaged metrics

2. **Visualizations**
   - Confusion matrix
   - Error analysis wordclouds
   - Confidence distribution plots
   - Text length analysis

### Running Evaluation
```bash
python -m src.evaluation.evaluator \
    --output_root "/path/to/output" \
    --best_model "best_model.pt" \
    --n_folds 7
```

### Output Structure
```
evaluation_results/
├── all_predictions.csv
├── classification_report.csv
├── confusion_matrix.png
├── cv_metrics.csv
├── error_analysis/
│   ├── error_wordclouds.png
│   ├── error_confidence_dist.png
│   └── error_length_dist.png
└── fold_results/
    ├── fold_1/
    ├── fold_2/
    └── ...
```
