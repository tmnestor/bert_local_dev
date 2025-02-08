import logging
import logging.config
from typing import TYPE_CHECKING
import time
from pathlib import Path

if TYPE_CHECKING:
    from ..config.config import ModelConfig


def setup_logging(config: "ModelConfig") -> None:
    """Initialize logging configuration."""
    # Map verbosity levels to logging levels
    log_levels = {
        0: logging.WARNING,  # Minimal output
        1: logging.INFO,  # Normal operation
        2: logging.DEBUG,  # Debug output
    }

    log_level = log_levels.get(config.verbosity, logging.INFO)
    
    # Ensure logs directory exists
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a more descriptive log filename by parsing the module name differently
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Get operation name from config class name
    config_class_name = config.__class__.__name__.lower()
    operation_map = {
        'modelconfig': 'training',
        'evaluationconfig': 'evaluation',
        'predictionconfig': 'prediction',
    }
    operation = operation_map.get(config_class_name, 'unknown')
        
    log_file = config.logs_dir / f"{operation}_{timestamp}.log"

    # Configure logging with both console and file handlers
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console handler
            logging.FileHandler(filename=log_file)  # File handler
        ],
        force=True  # Override any existing configuration
    )

    # Set level for specific loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    # Only show TQDM output for verbosity > 0
    if config.verbosity == 0:
        logging.getLogger("tqdm").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)
