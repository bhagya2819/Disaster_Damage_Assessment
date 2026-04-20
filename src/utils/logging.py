"""Project-wide logging setup using rich for readable tracebacks."""

from __future__ import annotations

import logging
import os

from rich.logging import RichHandler

_CONFIGURED = False


def get_logger(name: str = "dda") -> logging.Logger:
    """Return a configured logger. Idempotent: safe to call from anywhere."""
    global _CONFIGURED
    if not _CONFIGURED:
        level = os.environ.get("DDA_LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
        _CONFIGURED = True
    return logging.getLogger(name)
