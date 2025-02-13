# Configuration Guide

## Config File Structure
The framework uses YAML configuration files for experiment settings.

```yaml
model:
  name: bert-base-uncased
  max_length: 512
  batch_size: 16

training:
  epochs: 10
  learning_rate: 2e-5
  warmup_steps: 500
  
evaluation:
  validation_split: 0.1
  metrics: ['accuracy', 'f1', 'precision', 'recall']
```

## Environment Variables
- `BERT_CACHE_DIR`: Directory for model cache
- `BERT_OUTPUT_DIR`: Directory for outputs
- `BERT_LOG_LEVEL`: Logging level (INFO/DEBUG/WARNING)
