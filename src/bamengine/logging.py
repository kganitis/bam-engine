import logging
from typing import Any

(CRITICAL, ERROR, WARNING, INFO, DEBUG) = (
    logging.CRITICAL,
    logging.ERROR,
    logging.WARNING,
    logging.INFO,
    logging.DEBUG,
)
DEEP_DEBUG = 5
logging.addLevelName(DEEP_DEBUG, "DEEP")


class BamLogger(logging.Logger):
    """Application logger that understands the DEEP level."""

    def deep(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(DEEP_DEBUG):
            self._log(DEEP_DEBUG, msg, args, **kwargs)


# Make the logging module hand out our subclass from now on
logging.setLoggerClass(BamLogger)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def getLogger(name: str | None = None) -> BamLogger:  # convenience wrapper
    return logging.getLogger(name)  # type: ignore[return-value]
