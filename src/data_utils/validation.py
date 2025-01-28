"""Data validation utilities."""

from pathlib import Path
import pandas as pd
from typing import Optional, Set

class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass

def validate_dataset(
    data_file: Path,
    required_columns: Optional[Set[str]] = None
) -> None:
    """Validate a dataset file.
    
    Args:
        data_file: Path to CSV file
        required_columns: Set of required column names
        
    Raises:
        DataValidationError: If validation fails
        FileNotFoundError: If file doesn't exist
    """
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
        
    if required_columns is None:
        required_columns = {'text', 'category'}
    
    try:
        # Try reading with different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(data_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise DataValidationError("Could not read file with any encoding")
            
        # Check required columns
        missing = required_columns - set(df.columns)
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")
            
        # Check for missing values
        na_cols = df.columns[df.isna().any()].tolist()
        if na_cols:
            raise DataValidationError(f"Found missing values in columns: {na_cols}")
            
        # Check text column
        if not df['text'].dtype == object:
            raise DataValidationError("'text' column must be string type")
        if df['text'].str.len().min() == 0:
            raise DataValidationError("Found empty text samples")
            
        # Check category column
        if df['category'].nunique() < 2:
            raise DataValidationError("Need at least 2 unique categories")
            
    except Exception as e:
        if not isinstance(e, DataValidationError):
            raise DataValidationError(f"Validation failed: {str(e)}") from e
        raise
