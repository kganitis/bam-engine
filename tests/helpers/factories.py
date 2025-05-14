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

from bamengine.components import (
    Economy,
    Producer,
    Employer,
    Worker,
    Borrower,
    Lender,
)

# ───────────────────────── default dictionaries ────────────────────────── #


def _economy_defaults() -> dict[str, Any]:
    return dict(
        avg_mkt_price=1.15,
        avg_mkt_price_history=np.array([1.0, 1.05, 1.10, 1.15]),
        min_wage=1.0,
        min_wage_rev_period=4,
        r_bar=0.07,
        v=0.23,
    )


def _producer_defaults(n: int, *, queue_w: int) -> dict[str, Any]:
    return dict(
        production=np.full(n, 10.0, dtype=np.float64),
        inventory=np.zeros(n, dtype=np.float64),
        expected_demand=np.zeros(n, dtype=np.float64),
        desired_production=np.full(n, 10.0, dtype=np.float64),
        labor_productivity=np.ones(n, dtype=np.float64),
        price=np.full(n, 1.5, dtype=np.float64),
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


def _worker_defaults(n: int, *, queue_w: int) -> dict[str, Any]:
    return dict(
        employed=np.zeros(n, dtype=np.bool_),
        employer=np.full(n, -1, dtype=np.intp),
        employer_prev=np.full(n, -1, dtype=np.intp),
        wage=np.zeros(n, dtype=np.float64),
        periods_left=np.zeros(n, dtype=np.int64),
        contract_expired=np.zeros(n, dtype=np.bool_),
        fired=np.zeros(n, dtype=np.bool_),
        job_apps_head=np.full(n, -1, dtype=np.intp),
        job_apps_targets=np.full((n, queue_w), -1, dtype=np.intp),
    )


def _borrower_defaults(n: int, *, queue_h: int) -> dict[str, Any]:
    return dict(
        net_worth=np.full(n, 10.0, dtype=np.float64),
        total_funds=np.full(n, 10.0, dtype=np.float64),
        wage_bill=np.zeros(n, dtype=np.float64),
        credit_demand=np.zeros(n, dtype=np.float64),
        rnd_intensity=np.ones(n, dtype=np.float64),
        projected_fragility=np.zeros(n, dtype=np.float64),
        loan_apps_head=np.full(n, -1, dtype=np.int64),
        loan_apps_targets=np.full((n, queue_h), -1, dtype=np.int64),
    )


def _lender_defaults(n: int, *, queue_h: int) -> dict[str, Any]:
    return dict(
        equity_base=np.full(n, 10_000.0, dtype=np.float64),
        credit_supply=np.zeros(n, dtype=np.float64),
        interest_rate=np.zeros(n, dtype=np.float64),
        recv_apps_head=np.full(n, -1, dtype=np.int64),
        recv_apps=np.full((n, queue_h), -1, dtype=np.int64),
        opex_shock=None,  # scratch
    )

# ───────────────────────── public factory helpers ──────────────────────── #

def mock_economy(
    **overrides: Any,
) -> Economy:
    """
    Return a fully-typed `Economy`.

    Parameters
    ----------
    **overrides
        Field-value pairs that overwrite defaults.
    """
    cfg = _economy_defaults() | overrides
    return Economy(**cfg)


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


def mock_worker(
    n: int = 1,
    *,
    queue_w: int = 4,
    **overrides: Any,
) -> Worker:
    """
    Return a fully-typed `Worker`.

    Parameters
    ----------
    n
        Number of workers.
    queue_w
        Width of the application queue (`job_apps_targets.shape[1]`).
    **overrides
        Field-value pairs that overwrite defaults.
    """
    cfg = _worker_defaults(n, queue_w=queue_w) | overrides
    return Worker(**cfg)


def mock_borrower(
    n: int = 1,
    *,
    queue_h: int = 2,
    **overrides: Any,
) -> Borrower:
    """
    Return a fully-typed `Borrower`.

    Parameters
    ----------
    n
        Number of borrowers.
    queue_h
        Width of the outbound loan-application queue (`loan_apps_targets.shape[1]`).
    **overrides
        Field-value pairs that overwrite defaults.
    """
    cfg = _borrower_defaults(n, queue_h=queue_h) | overrides
    return Borrower(**cfg)


def mock_lender(
    n: int = 1,
    *,
    queue_h: int = 2,
    **overrides: Any,
) -> Lender:
    """
    Return a fully-typed `Lender`.

    Parameters
    ----------
    n
        Number of lenders.
    queue_h
        Width of the inbound loan queue (`recv_apps.shape[1]`).
    **overrides
        Field-value pairs that overwrite defaults.
    """
    cfg = _lender_defaults(n, queue_h=queue_h) | overrides
    return Lender(**cfg)
