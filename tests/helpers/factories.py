# tests/helpers/factories.py
"""
Reusable builders for component instances in unit / property tests.

* They construct the **full** dataclasses from `bamengine.components`.
* All vectors are initialised with small, deterministic defaults.
* You can override any field via keyword arguments.

Example
-------
>>> prod = mock_producer(5, desired_production=np.arange(5) + 1.0)
>>> emp  = mock_employer(5, current_labor=np.array([2,1,0,3,4]))
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from bamengine.components import Employer, Producer

Float1D = NDArray[np.float64]
Int1D = NDArray[np.int64]

# deterministic RNG for *test helpers only*  –– do *not* use the global bit-gen
_rng = default_rng(0)

# ───────────────────────── default dictionaries ────────────────────────── #


def _producer_defaults(n: int, *, queue_w: int) -> dict[str, Any]:
    return dict(
        production=np.full(n, 10.0, dtype=np.float64),
        inventory=np.zeros(n, dtype=np.float64),
        expected_demand=np.zeros(n, dtype=np.float64),
        desired_production=np.full(n, 10.0, dtype=np.float64),
        labor_productivity=np.ones(n, dtype=np.float64),
        price=_rng.uniform(1.3, 1.7, size=n),  # deterministic but non-constant
        # scratch (allocated lazily by production rule unless eager=True)
        prod_shock=None,
        prod_mask_up=None,
        prod_mask_dn=None,
    )


def _employer_defaults(n: int, *, queue_w: int) -> dict[str, Any]:
    return dict(
        desired_labor=np.zeros(n, dtype=np.int64),
        current_labor=np.zeros(n, dtype=np.int64),
        wage_offer=np.ones(n, dtype=np.float64),
        wage_bill=np.zeros(n, dtype=np.float64),
        n_vacancies=np.zeros(n, dtype=np.int64),
        total_funds=np.ones(n, dtype=np.float64),
        recv_job_apps_head=np.full(n, -1, dtype=np.int64),
        recv_job_apps=np.full((n, queue_w), -1, dtype=np.int64),
        wage_shock=None,  # scratch
    )


# ───────────────────────── public factory helpers ──────────────────────── #


def mock_producer(
    n: int = 1,
    *,
    queue_w: int = 4,
    alloc_scratch: bool = False,
    **overrides: Any,
) -> Producer:
    """
    Return a fully-typed `Producer`.

    Parameters
    ----------
    n
        Number of producers.
    queue_w
        Placeholder for future outbound queues (kept for symmetry).
    alloc_scratch
        If *True* pre-allocate zeroed scratch buffers so the system call
        does **not** have to create them on first use (useful in some
        buffer-reuse tests).
    **overrides
        Field-value pairs that overwrite defaults.
    """
    cfg = _producer_defaults(n, queue_w=queue_w) | overrides

    if alloc_scratch and cfg["prod_shock"] is None:
        cfg["prod_shock"] = np.zeros(n, dtype=np.float64)
        cfg["prod_mask_up"] = np.zeros(n, dtype=np.bool_)
        cfg["prod_mask_dn"] = np.zeros(n, dtype=np.bool_)

    return Producer(**cfg)


def mock_employer(
    n: int = 1,
    *,
    queue_w: int = 4,
    **overrides: Any,
) -> Employer:
    """
    Return a fully-typed `Employer`.

    Parameters
    ----------
    n
        Number of employers.
    queue_w
        Width of the application queue (`recv_job_apps.shape[1]`).
    **overrides
        Field-value pairs that overwrite defaults.
    """
    cfg = _employer_defaults(n, queue_w=queue_w) | overrides
    return Employer(**cfg)
