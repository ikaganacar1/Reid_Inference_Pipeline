"""
Centralized Logging Configuration for ReID Pipeline

Provides Docker-style logging with:
- Separate log files for each component (pipeline, evaluation, api, worker)
- Log rotation to prevent disk fill
- Console + file handlers
- Consistent formatting across all components
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

# Default log directory
DEFAULT_LOG_DIR = Path(__file__).parent.parent.parent / "logs"


class LoggerFactory:
    """Factory for creating consistent loggers across components"""

    _initialized_loggers = set()

    @classmethod
    def get_logger(
        cls,
        name: str,
        component: str = "pipeline",
        log_dir: Optional[Path] = None,
        log_level: str = "INFO",
        console: bool = True,
        file: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> logging.Logger:
        """
        Get or create a logger for the specified component.

        Args:
            name: Logger name (usually __name__)
            component: Component name (pipeline, evaluation, api, worker)
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            console: Enable console output
            file: Enable file output
            max_bytes: Max log file size before rotation
            backup_count: Number of backup files to keep

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)

        # Skip if already configured
        if name in cls._initialized_loggers:
            return logger

        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        logger.handlers = []  # Clear existing handlers

        # Formatter with timestamp and component info
        formatter = logging.Formatter(
            f"%(asctime)s | {component.upper():10} | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.DEBUG)
            logger.addHandler(console_handler)

        # File handler with rotation
        if file:
            log_dir = log_dir or DEFAULT_LOG_DIR
            log_dir.mkdir(parents=True, exist_ok=True)

            # Create date-stamped log file
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = log_dir / f"{component}_{date_str}.log"

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

        cls._initialized_loggers.add(name)
        return logger


def setup_pipeline_logging(log_level: str = "INFO", log_dir: Optional[Path] = None) -> logging.Logger:
    """Setup logging for single/multi camera pipeline"""
    return LoggerFactory.get_logger(
        "reid_pipeline",
        component="pipeline",
        log_dir=log_dir,
        log_level=log_level
    )


def setup_evaluation_logging(log_level: str = "INFO", log_dir: Optional[Path] = None) -> logging.Logger:
    """Setup logging for dataset evaluation"""
    return LoggerFactory.get_logger(
        "reid_evaluation",
        component="evaluation",
        log_dir=log_dir,
        log_level=log_level
    )


def setup_api_logging(log_level: str = "INFO", log_dir: Optional[Path] = None) -> logging.Logger:
    """Setup logging for API server"""
    return LoggerFactory.get_logger(
        "reid_api",
        component="api",
        log_dir=log_dir,
        log_level=log_level
    )


def setup_worker_logging(log_level: str = "INFO", log_dir: Optional[Path] = None) -> logging.Logger:
    """Setup logging for worker process"""
    return LoggerFactory.get_logger(
        "reid_worker",
        component="worker",
        log_dir=log_dir,
        log_level=log_level
    )


def configure_all_loggers(log_level: str = "INFO", log_dir: Optional[Path] = None):
    """Configure all component loggers at once"""
    log_dir = log_dir or DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create all component loggers
    setup_pipeline_logging(log_level, log_dir)
    setup_evaluation_logging(log_level, log_dir)
    setup_api_logging(log_level, log_dir)
    setup_worker_logging(log_level, log_dir)


def tail_log(component: str, lines: int = 50, log_dir: Optional[Path] = None) -> str:
    """
    Read last N lines from a component's log file.

    Args:
        component: Component name (pipeline, evaluation, api, worker)
        lines: Number of lines to read
        log_dir: Log directory path

    Returns:
        Last N lines of the log file
    """
    log_dir = log_dir or DEFAULT_LOG_DIR
    date_str = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"{component}_{date_str}.log"

    if not log_file.exists():
        return f"No log file found: {log_file}"

    with open(log_file, 'r') as f:
        all_lines = f.readlines()
        return ''.join(all_lines[-lines:])


if __name__ == "__main__":
    # Test logging
    configure_all_loggers("DEBUG")

    pipeline_logger = logging.getLogger("reid_pipeline")
    eval_logger = logging.getLogger("reid_evaluation")
    api_logger = logging.getLogger("reid_api")
    worker_logger = logging.getLogger("reid_worker")

    pipeline_logger.info("Pipeline test message")
    eval_logger.info("Evaluation test message")
    api_logger.info("API test message")
    worker_logger.info("Worker test message")

    print(f"\nLogs written to: {DEFAULT_LOG_DIR}")
