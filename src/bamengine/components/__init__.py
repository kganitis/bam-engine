# src/bamengine/components/__init__.py
"""Internal dataclass bundles."""

from bamengine.components.borrower import Borrower
from bamengine.components.consumer import Consumer
from bamengine.components.economy import Economy, LoanBook
from bamengine.components.employer import Employer
from bamengine.components.lender import Lender
from bamengine.components.producer import Producer
from bamengine.components.worker import Worker

__all__ = [
    "Economy",
    "Producer",
    "Employer",
    "Worker",
    "Borrower",
    "Lender",
    "LoanBook",
    "Consumer",
]
