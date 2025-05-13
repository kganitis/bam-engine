# tests/__init__.py

import numpy as np

from bamengine.components.economy import LoanBook
from tests.helpers.factories import mock_employer, mock_producer
from tests.helpers.invariants import assert_basic_invariants

__all__ = [
    "mock_producer",
    "mock_employer",
    "assert_basic_invariants",
]


def create_empty_ledger(n_init: int = 128) -> LoanBook:
    """
    Return a LoanBook pre-allocated for *n_init* rows so unit tests can append
    a handful of loans without triggering a resize.
    """
    led = LoanBook(
        borrower=np.empty(n_init, np.int64),
        lender=np.empty(n_init, np.int64),
        principal=np.empty(n_init, np.float64),
        rate=np.empty(n_init, np.float64),
        interest=np.empty(n_init, np.float64),
        debt=np.empty(n_init, np.float64),
        capacity=n_init,
        size=0,
    )
    assert led.capacity == n_init and led.size == 0
    return led
