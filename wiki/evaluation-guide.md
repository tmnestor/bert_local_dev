# Evaluation and Analysis

## Metrics Overview
- Accuracy
- F1 Score
- Precision/Recall
- Confusion Matrix

## Running Evaluation
```bash
python -m src.evaluate \
  --model_path outputs/best_model.pt \
  --test_data data/test.csv \
  --output_report evaluation_report.json
```

## Error Analysis
The framework provides detailed error analysis tools:
- Confusion matrix visualization
- Per-class performance metrics
- Error samples examination
- Attention pattern visualization
