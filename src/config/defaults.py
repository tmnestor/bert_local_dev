"""Default configuration values for model architectures."""

CLASSIFIER_DEFAULTS = {
    'standard': {
        'num_layers': 2,
        'hidden_dim': 256,
        'activation': 'gelu',
        'dropout_rate': 0.1,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'batch_size': 32,
        'warmup_ratio': 0.1,
    },
}

OPTIM_SEARCH_SPACE = {
    'batch_size': [16, 32, 64],
    'num_layers': (1, 4),
    'hidden_dim': [32, 64, 128, 256, 512, 1024],
    'activation': [
        'relu', 'gelu', 'elu', 'leaky_relu', 'selu',  # Basic activations
        'mish', 'swish', 'hardswish', 'tanh', 'prelu'  # Advanced activations
    ],
    'dropout_rate': (0.1, 0.5),
    'learning_rate': (1e-5, 1e-3),  # log scale
    'weight_decay': (1e-8, 1e-3),   # log scale
    'warmup_ratio': (0.0, 0.2)
}

MODEL_DEFAULTS = {
    'max_seq_len': 64,
    'metric': 'f1',
    'metrics': ["accuracy", "f1", "precision", "recall"],
    'device': 'cpu',
    'n_trials': 100,
    'n_experiments': 1
}
