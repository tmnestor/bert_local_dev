"""Data utilities module."""
from .dataset import TextClassificationDataset
from .splitter import DataSplit, DataSplitter
from .loaders import load_and_preprocess_data, create_dataloaders
from .validation import validate_dataset, DataValidationError
from ..utils.logging_manager import get_logger  # Changed from setup_logger

logger = get_logger(__name__)  # Changed to get_logger


__all__ = [
    'TextClassificationDataset',
    'DataSplit',
    'DataSplitter',
    'load_and_preprocess_data',
    'create_dataloaders',
    'validate_dataset',
    'DataValidationError'
]
