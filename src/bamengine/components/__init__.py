# src/bamengine/components/__init__.py
"""Internal dataclass bundles."""

from bamengine.components.economy import Economy, LoanBook
from bamengine.components.employer import Employer
from bamengine.components.producer import Producer
from bamengine.components.worker import Worker
from bamengine.components.borrower import Borrower
from bamengine.components.lender import Lender

__all__ = [
    "Economy",
    "Producer",
    "Employer",
    "Worker",
    "Borrower",
    "Lender",
    "LoanBook",
]
