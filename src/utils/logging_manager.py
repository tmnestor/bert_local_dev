import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.config import ModelConfig

def setup_logger(name: str, config: Optional["ModelConfig"] = None) -> logging.Logger:
    """Setup logger using configuration directory paths.

    Args:
        name: Logger name (typically __name__)
        config: ModelConfig instance. If provided, uses its logs_dir for file output.
               If None, only console output is setup.

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Console formatter - Fix typo in levelname
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'  # Changed from levellevel to levelname
    )
    
    # Always add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Only add file handler if config with logs_dir is provided
    if config is not None and hasattr(config, 'logs_dir'):
        # File formatter with more details
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = config.logs_dir / f'bert_classifier_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to: {log_file}")
    
    return logger

# Remove global logger - each module should instantiate its own
