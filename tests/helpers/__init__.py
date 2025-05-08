# tests/helpers/__init__.py

import numpy as np
from bamengine.components.credit import LoanBook

def create_empty_ledger(n_init: int = 128) -> LoanBook:
    """
    Return a LoanBook pre-allocated for *n_init* rows so unit tests can append
    a handful of loans without triggering a resize.
    """
    return LoanBook(
        firm=np.empty(n_init, dtype=np.int64),
        bank=np.empty(n_init, dtype=np.int64),
        principal=np.empty(n_init, dtype=np.float64),
        rate=np.empty(n_init, dtype=np.float64),
        interest=np.empty(n_init, dtype=np.float64),
        debt=np.empty(n_init, dtype=np.float64),
        capacity=n_init,
        size=0,
    )
