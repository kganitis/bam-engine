# bamengine/_logging_ext.pyi
import logging
from typing import Any, Optional

(CRITICAL, ERROR, WARNING, INFO, DEBUG) = (
    logging.CRITICAL,
    logging.ERROR,
    logging.WARNING,
    logging.INFO,
    logging.DEBUG,
)
DEEP_DEBUG: int

class BamLogger(logging.Logger):
    def deep(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

def getLogger(name: Optional[str] = ...) -> BamLogger: ...
