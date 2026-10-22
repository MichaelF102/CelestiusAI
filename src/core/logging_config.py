"""Standard application logging configuration with Streamlit rerun deduplication."""

import logging
import sys

_LOGGING_INITIALIZED = False


def setup_logging(log_level: str | None = None) -> logging.Logger:
    """Configure root logger with console output and consistent formatting.

    Guards against duplicate handlers created during Streamlit script reruns.
    """
    global _LOGGING_INITIALIZED

    level_str = (log_level or "INFO").upper()
    numeric_level = getattr(logging, level_str, logging.INFO)

    root_logger = logging.getLogger("da_agent")
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers on Streamlit reruns
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.propagate = False
    else:
        # Update level in case it changed
        root_logger.setLevel(numeric_level)
        for h in root_logger.handlers:
            h.setLevel(numeric_level)

    _LOGGING_INITIALIZED = True
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve a namespaced child logger under the 'da_agent' hierarchy."""
    if not _LOGGING_INITIALIZED:
        setup_logging()
    clean_name = name if name.startswith("da_agent") else f"da_agent.{name}"
    return logging.getLogger(clean_name)
