#!/usr/bin/env python3
"""
CORE LOGGING SYSTEM
===================

Centralized logging configuration for the Enterprise prime aligned compute Platform.
Provides structured logging with proper formatting, rotation, and error handling.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

class ConsciousnessLogger:
    """Centralized logging system for prime aligned compute platform"""

    # Log levels
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    def __init__(self, name: str = "consciousness_platform", log_level: str = "INFO"):
        self.name = name
        self.log_level = self._parse_log_level(log_level)
        self.logger = None
        self._configured = False

    def _parse_log_level(self, level: str) -> int:
        """Parse log level string to logging constant"""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        return level_map.get(level.upper(), logging.INFO)

    def configure(self,
                  log_file: Optional[str] = None,
                  max_bytes: int = 10*1024*1024,  # 10MB
                  backup_count: int = 5,
                  console: bool = True,
                  format_string: Optional[str] = None,
                  log_level: Optional[str] = None) -> logging.Logger:
        """Configure the logger with file and console handlers"""

        if self._configured:
            return self.logger

        # Create logger
        self.logger = logging.getLogger(self.name)

        # Set log level (use parameter if provided, otherwise use instance default)
        if log_level is not None:
            level = self._parse_log_level(log_level)
        else:
            level = self.log_level

        self.logger.setLevel(level)

        # Remove any existing handlers
        self.logger.handlers.clear()

        # Default format
        if format_string is None:
            format_string = "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"

        formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler with rotation
        if log_file:
            try:
                # Create log directory if it doesn't exist
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)

                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
                file_handler.setLevel(self.log_level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

            except Exception as e:
                # Fallback to console only if file logging fails
                self.logger.warning(f"Failed to configure file logging to {log_file}: {e}")

        self._configured = True
        self.logger.info(f"Logging configured for {self.name} at level {logging.getLevelName(self.log_level)}")
        return self.logger

    def get_logger(self) -> logging.Logger:
        """Get the configured logger"""
        if not self._configured:
            return self.configure()
        return self.logger

# Global logger instances
_platform_logger = ConsciousnessLogger("consciousness_platform")
_math_logger = ConsciousnessLogger("consciousness_math")
_security_logger = ConsciousnessLogger("security_system")

def get_platform_logger() -> logging.Logger:
    """Get the main platform logger"""
    return _platform_logger.get_logger()

def get_math_logger() -> logging.Logger:
    """Get the mathematics logger"""
    return _math_logger.get_logger()

def get_security_logger() -> logging.Logger:
    """Get the security logger"""
    return _security_logger.get_logger()

def configure_all_loggers(log_level: str = "INFO",
                         log_dir: str = "logs",
                         enable_file_logging: bool = True) -> Dict[str, logging.Logger]:
    """Configure all loggers with consistent settings"""

    # Create log directory
    if enable_file_logging:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Configure main platform logger
        platform_log_file = os.path.join(log_dir, "consciousness_platform.log")
        _platform_logger.configure(log_file=platform_log_file, log_level=log_level)

        # Configure math logger
        math_log_file = os.path.join(log_dir, "consciousness_math.log")
        _math_logger.configure(log_file=math_log_file, log_level=log_level)

        # Configure security logger
        security_log_file = os.path.join(log_dir, "security_system.log")
        _security_logger.configure(log_file=security_log_file, log_level=log_level)
    else:
        # Console only
        _platform_logger.configure(log_level=log_level)
        _math_logger.configure(log_level=log_level)
        _security_logger.configure(log_level=log_level)

    return {
        "platform": _platform_logger.get_logger(),
        "math": _math_logger.get_logger(),
        "security": _security_logger.get_logger()
    }

class LogContextManager:
    """Context manager for structured logging"""

    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation}", extra={"context": self.context})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        if exc_type:
            self.logger.error(
                f"Failed {self.operation} after {duration:.2f}s",
                extra={
                    "context": self.context,
                    "error": str(exc_val),
                    "duration": duration
                }
            )
        else:
            self.logger.info(
                f"Completed {self.operation} in {duration:.2f}s",
                extra={
                    "context": self.context,
                    "duration": duration
                }
            )

# Utility functions
def log_performance(logger: logging.Logger, operation: str, duration: float, **metrics):
    """Log performance metrics"""
    logger.info(
        f"Performance: {operation}",
        extra={
            "operation": operation,
            "duration": duration,
            "metrics": metrics
        }
    )

def log_error_with_context(logger: logging.Logger, error: Exception,
                          operation: str, **context):
    """Log errors with full context"""
    logger.error(
        f"Error in {operation}: {error}",
        extra={
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        },
        exc_info=True
    )

# Initialize with environment-based configuration
def initialize_logging():
    """Initialize logging system based on environment variables"""
    log_level = os.getenv("CONSCIOUSNESS_LOG_LEVEL", "INFO")
    log_dir = os.getenv("CONSCIOUSNESS_LOG_DIR", "logs")
    enable_file_logging = os.getenv("CONSCIOUSNESS_FILE_LOGGING", "true").lower() == "true"

    loggers = configure_all_loggers(
        log_level=log_level,
        log_dir=log_dir,
        enable_file_logging=enable_file_logging
    )

    # Log initialization
    platform_logger = loggers["platform"]
    platform_logger.info("prime aligned compute Platform logging system initialized")
    platform_logger.info(f"Log level: {log_level}")
    platform_logger.info(f"File logging: {'enabled' if enable_file_logging else 'disabled'}")

    return loggers

# Auto-initialize if this module is imported directly
if __name__ != "__main__":
    try:
        initialize_logging()
    except Exception as e:
        # Fallback to basic console logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        logging.getLogger("consciousness_platform").warning(
            f"Failed to initialize advanced logging: {e}"
        )
