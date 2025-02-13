# Prediction Module (predict.py)

The prediction module provides a command-line interface for generating predictions using a trained BERT classifier. It handles model loading, batch inference, and saving predictions in a standardized format.

## Key Features

- Load trained models with error handling
- Batch prediction generation
- Confidence score calculation
- Standardized output format
- Memory-efficient processing
- Progress tracking

## Command Line Usage

### Basic Usage
```bash
python -m src.prediction.predict \
    --output_root "/path/to/project" \
    --best_model model_name.pt \
    --data_file test.csv \
    --output_file predictions.csv
```

### Required Arguments

- `--best_model`: Name of the trained model file (must be in best_trials directory)
- `--data_file`: Path to input data file (CSV format)

### Optional Arguments

- `--output_root`: Root directory for project (default: current directory)
- `--output_file`: Name for output predictions file (default: predictions.csv)
- `--device`: Computing device ['cpu', 'cuda'] (default: cpu)
- `--batch_size`: Batch size for predictions (default: 32)
- `--bert_model_name`: Path to BERT encoder (default: bert_encoder in output_root)

### Expected Directory Structure

```
output_root/
├── bert_encoder/           # BERT model files
├── best_trials/           # Trained model checkpoints
│   └── best_model.pt     # Your trained model
├── data/                  # Data files
│   └── test.csv          # Input data for predictions
└── predictions/          # Generated predictions
    └── predictions.csv   # Output file
```

### Input Data Format

The input CSV file must contain at least:
- A 'text' column with the text to classify
- A 'Hash_Id' column with unique identifiers

Example:
```csv
Hash_Id,text
001,"This is a sample text"
002,"Another example text"
```

### Output Format

The generated predictions CSV will contain:
```csv
Hash_Id,Cleaned_Claim,FTC_Label
001,"This is a sample text","class_a"
002,"Another example text","class_b"
```

## Example Usage

1. Basic prediction:
```bash
python -m src.prediction.predict \
    --output_root "/Users/tod/BERT_TRAINING" \
    --best_model best_model.pt \
    --data_file test.csv
```

2. With custom device and batch size:
```bash
python -m src.prediction.predict \
    --output_root "/Users/tod/BERT_TRAINING" \
    --best_model best_model.pt \
    --data_file test.csv \
    --device cuda \
    --batch_size 64
```

3. Custom output location:
```bash
python -m src.prediction.predict \
    --output_root "/Users/tod/BERT_TRAINING" \
    --best_model best_model.pt \
    --data_file test.csv \
    --output_file custom_predictions.csv
```

## Error Handling

The module includes comprehensive error handling for:
- Missing model files
- Invalid data formats
- Memory issues
- Device compatibility
- Model loading errors

Error messages are logged with detailed information to help diagnose issues.

## Performance Tips

1. Adjust batch size based on available memory:
   - Lower for CPU (8-32)
   - Higher for GPU (32-128)

2. Use appropriate device:
   - `--device cuda` for GPU acceleration
   - `--device cpu` for CPU-only systems

3. Ensure input data is properly formatted:
   - Clean text data
   - Remove invalid characters
   - Handle missing values

## Logging

The module provides three verbosity levels:
- 0: Minimal (errors only)
- 1: Normal (progress + summary)
- 2: Debug (detailed information)

Control verbosity with:
```bash
--verbosity 1  # Choose 0, 1, or 2
```
