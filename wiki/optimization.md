# Hyperparameter Optimization

## Population Based Training (PBT)

### Overview
The framework implements Population Based Training with:
- Population size: 4 parallel models
- Adaptation threshold: Bottom 20%
- Exploration strategy: Random perturbation
- Exploitation strategy: Parameter copying

### Search Space
```python
SEARCH_SPACE = {
    'batch_size': [8, 16, 32, 64],
    'num_hidden_layers': (1, 4),
    'dropout_rate': (0.1, 0.6),
    'lr': (1e-6, 5e-3),
    'weight_decay': (1e-8, 1e-2),
    'warmup_ratio': (0.0, 0.3),
    'activation': ["relu", "gelu", "silu", "elu", "tanh"]
}
```

### Running Optimization
```bash
python -m src.tuning.optimize \
    --output_root "/path/to/output" \
    --study_name "bert_opt" \
    --n_trials 100
```

### Memory Management
- Dynamic batch size adjustment
- Memory usage threshold: 85%
- Automatic scaling based on available resources
