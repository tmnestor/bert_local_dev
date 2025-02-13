# Configuration Guide

## Configuration File (config.yml)
The framework uses a YAML-based configuration system with support for anchors and path resolution.

### Base Configuration Structure
```yaml
output_root: /path/to/output
dirs:
  best_trials: best_trials
  evaluation: evaluation_results
  logs: logs
  data: data

model:
  max_seq_len: 64
  batch_size: 32
  num_epochs: 10
  device: cpu
  metric: f1

classifier:
  bert_hidden_size: 384
  hidden_dims: [412, 220, 118]
  dropout_rate: 0.35
  activation: gelu
```

### Configuration Classes
1. **ModelConfig**
   - Base configuration class for model parameters
   - Handles path resolution and validation

2. **EvaluationConfig**
   - Extends ModelConfig for evaluation settings
   - Adds k-fold cross-validation parameters

3. **PredictionConfig**
   - Extends ModelConfig for prediction tasks
   - Manages output directories for predictions
