import logging
import structlog
import os, sys
from typing import Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def logger_config(process_name: str) -> logging.Logger:
    """
    Configure a logger for structured logging.

    :param process_name: Name of the process for which the logger is being configured.
    :param level: Logging level (default is logging.INFO).
    :return: Configured logger instance.
    """
    # Get log level from environment variable or default to INFO
    level = os.getenv("LOG_LEVEL", "INFO")

    # Validate log level: Check if it's an integer in [10, 20, 30, 40, 50] or a string in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    level = level.upper() if isinstance(level, str) else level
    if not ((isinstance(level, int) and level in logging._levelToName) or (isinstance(level, str) and level in logging._nameToLevel)):
        raise ValueError(f"Invalid log level: {level}. Must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL.")
    
    logging.basicConfig(level=level, stream=sys.stderr)

    structlog.configure(
        processors=[
            # Add timestamp to every log record
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            # Add structured context
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Convert to JSON
            structlog.processors.JSONRenderer()
        ],
        # Use a standard dict for context
        context_class=dict,
        # Use the standard library logger factory
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Use the standard library bound logger
        wrapper_class=structlog.stdlib.BoundLogger,
        # Cache the logger on first use
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger().bind(process_name=process_name)

    return logger