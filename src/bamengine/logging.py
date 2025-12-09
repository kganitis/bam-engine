"""
Custom logging configuration for BAM Engine.

Extends Python's standard logging with a custom TRACE level (5)
for very verbose debugging output. Provides BamLogger class with
per-event log level configuration support.

Log Levels
----------
- CRITICAL (50): Critical errors
- ERROR (40): Errors
- WARNING (30): Warnings
- INFO (20): Informational messages (default)
- DEBUG (10): Debug messages
- TRACE (5): Very verbose debug messages

Examples
--------
Use logger in events:

>>> from bamengine import logging
>>> logger = getLogger("bamengine.events.my_event")
>>> logger.info("Event executing")
>>> logger.debug("Detailed debug info")
>>> logger.trace("Very verbose output")

Configure per-event log levels:

>>> import bamengine as be
>>> log_config = {
...     "default_level": "INFO",
...     "events": {"firms_adjust_price": "DEBUG", "workers_send_one_round": "WARNING"},
... }
>>> sim = be.Simulation.init(logging=log_config)

Check if logging level enabled:

>>> if logger.isEnabledFor(logging.DEBUG):
...     expensive_stats = compute_expensive_stats()
...     logger.debug("Stats: %s", expensive_stats)

See Also
--------
Event.get_logger : Get logger for specific event
bamengine.config.schema.LoggingConfig : Logging configuration schema
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
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


class BamLogger(logging.Logger):
    """
    Custom logger with TRACE level support.

    Extends Python's Logger to add the `trace()` method for very verbose
    debugging output (level 5).

    Examples
    --------
    >>> logger = BamLogger("test")
    >>> logger.setLevel(5)  # TRACE
    >>> logger.trace("Very verbose message")

    See Also
    --------
    getLogger : Factory function for obtaining BamLogger instances
    """

    def trace(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log message at TRACE level (5).

        Parameters
        ----------
        msg : str
            Message format string.
        *args : Any
            Arguments for message formatting.
        **kwargs : Any
            Additional logging kwargs.
        """
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, **kwargs)


# Make the logging module hand out our subclass from now on
logging.setLoggerClass(BamLogger)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def getLogger(name: str = "bamengine") -> BamLogger:
    """
    Get a BamLogger instance.

    Convenience wrapper around logging.getLogger() that returns
    a BamLogger instance with TRACE support.

    Parameters
    ----------
    name : str, optional, default="bamengine"
        Logger name. If None, returns bamengine logger.

    Returns
    -------
    BamLogger
        Logger instance with trace() method.

    Examples
    --------
    >>> from bamengine import logging
    >>> logger = logging.getLogger("bamengine.events.my_event")
    >>> logger.info("Message")
    >>> logger.trace("Very verbose")

    See Also
    --------
    BamLogger : Custom logger class
    Event.get_logger : Get logger for specific event
    """
    return logging.getLogger(name)  # type: ignore[return-value]
