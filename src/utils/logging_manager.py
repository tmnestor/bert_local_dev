import logging
import logging.config
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..config.config import ModelConfig

def setup_logging(config: "ModelConfig") -> None:
    """Initialize logging configuration."""
    # Map verbosity levels to logging levels
    log_levels = {
        0: logging.WARNING,  # Minimal output
        1: logging.INFO,     # Normal operation
        2: logging.DEBUG     # Debug output
    }
    
    log_level = log_levels.get(config.verbosity, logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(message)s',  # Simplified format for evaluation output
        force=True  # Override any existing configuration
    )
    
    # Set level for specific loggers
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    # Only show TQDM output for verbosity > 0
    if config.verbosity == 0:
        logging.getLogger('tqdm').setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)
