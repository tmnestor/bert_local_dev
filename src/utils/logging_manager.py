import logging
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str = __name__,
    level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """Setup and configure logger with consistent formatting"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Only add handlers if they don't exist
        logger.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger

# Global logger instance for common use
logger = setup_logger('bert_classifier')
