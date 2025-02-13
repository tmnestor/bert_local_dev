# Training Guide

## Basic Training

### Command Line Usage
```bash
python -m src.training.train \
    --output_root "/path/to/output" \
    --num_epochs 5 \
    --batch_size 32 \
    --verbosity 1
```

### Available Parameters
| Parameter | Description | Default |
|-----------|-------------|----------|
| output_root | Root directory for outputs | /Users/tod/BERT_TRAINING |
| num_epochs | Number of training epochs | 10 |
| batch_size | Training batch size | 32 |
| max_seq_len | Maximum sequence length | 64 |
| learning_rate | Initial learning rate | 0.003 |
| device | Training device (cpu/cuda) | cpu |

### Training Flow
1. Data loading and preprocessing
2. Model initialization
3. Training loop with validation
4. Model checkpointing
5. Progress tracking and logging

### Early Stopping
The framework implements intelligent early stopping with:
- Patience-based stopping
- Performance regression detection
- Trend analysis
- Moving average smoothing
