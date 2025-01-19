import logging
from pathlib import Path
import sys
from datetime import datetime

def setup_logger(name: str, log_dir: Path = Path("logs")) -> logging.Logger:
    """Setup a logger with both file and console output.

    Sets up a logger with handlers for both file output (with detailed formatting) 
    and console output (with simplified formatting). Creates timestamped log files.

    Args:
        name: Logger name (typically __name__).
        log_dir: Directory to store log files (default: "logs").

    Returns:
        Logger: Configured logging.Logger instance.

    Raises:
        OSError: If log directory cannot be created.
    """
    # Create logs directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler - use timestamp in filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        log_dir / f'bert_classifier_{timestamp}.log'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_log_dir() -> Path:
    """Get the path to the log directory.

    Returns:
        Path: Path object pointing to the logs directory.
    """
    return Path("logs")

# Global logger instance for common use
global_logger = setup_logger('bert_classifier')
