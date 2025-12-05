"""
Type stubs for bamengine.logging module.

This type stub file provides type hints for the custom logging module,
enabling proper type checking and IDE autocomplete for BamLogger and
the custom TRACE log level.

Type Definitions
----------------
BamLogger : class
    Custom logger with trace() method for TRACE level (5).
getLogger : function
    Factory function returning BamLogger instances.

Constants
---------
CRITICAL, ERROR, WARNING, INFO, DEBUG : int
    Standard Python logging levels (re-exported from logging module).
TRACE : int
    Custom log level (5) for very verbose debugging output.

See Also
--------
bamengine.logging : Implementation module
logging : Python standard library logging module
"""

import logging
from typing import Any

(CRITICAL, ERROR, WARNING, INFO, DEBUG) = (
    logging.CRITICAL,
    logging.ERROR,
    logging.WARNING,
    logging.INFO,
    logging.DEBUG,
)
TRACE: int

class BamLogger(logging.Logger):
    """Custom logger with TRACE level support."""

    def trace(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log message at TRACE level (5)."""
        ...

def getLogger(name: str | None = ...) -> BamLogger:
    """Get a BamLogger instance with trace() method."""
    ...
