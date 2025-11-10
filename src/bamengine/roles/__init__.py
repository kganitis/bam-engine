"""Internal dataclass bundles."""

from bamengine.roles.borrower import Borrower
from bamengine.roles.consumer import Consumer
from bamengine.roles.employer import Employer
from bamengine.roles.lender import Lender
from bamengine.roles.producer import Producer
from bamengine.roles.worker import Worker

__all__ = [
    "Producer",
    "Employer",
    "Worker",
    "Borrower",
    "Lender",
    "Consumer",
]
