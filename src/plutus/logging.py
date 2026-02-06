"""Structured logging for Plutus.

Uses structlog for clean, structured logs that are:
- JSON format for cloud ingestion
- Console format for local development
- Portable for future cloud logging services
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog.types import Processor

from plutus.config import get_settings


def add_app_context(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add application context to log events."""
    event_dict["app"] = "plutus"
    return event_dict


def setup_logging() -> None:
    """Configure structured logging for the application.
    
    Supports:
    - JSON format (cloud-ready, default)
    - Console format (local development)
    - File output (optional)
    """
    settings = get_settings()
    log_settings = settings.logging
    
    # Shared processors for all outputs
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        add_app_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Format-specific renderer
    if log_settings.is_json:
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )
    
    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.ExceptionRenderer(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Create formatter for stdlib logging
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_settings.level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_settings.file_path:
        file_path = Path(log_settings.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    for logger_name in ["httpx", "httpcore", "urllib3"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# Context manager for adding context to logs
class LogContext:
    """Context manager for adding temporary context to logs.
    
    Example:
        with LogContext(task="portfolio_monitor", ticker="AAPL"):
            logger.info("Processing ticker")
    """
    
    def __init__(self, **context: Any) -> None:
        self.context = context
        self._token: Any = None
    
    def __enter__(self) -> "LogContext":
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self.context.keys())


def set_log_level(level: str) -> None:
    """Change the log level at runtime.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))


def suppress_logs() -> None:
    """Suppress all logs except errors (for clean CLI output)."""
    set_log_level("ERROR")


def restore_logs() -> None:
    """Restore logs to configured level."""
    settings = get_settings()
    set_log_level(settings.logging.level)
