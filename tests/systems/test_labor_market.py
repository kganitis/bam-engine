from typing import Tuple

import numpy as np
from numpy.random import Generator, default_rng

from bamengine.components.economy import Economy
from bamengine.components.firm_labor import FirmHiring, FirmWageOffer
from bamengine.components.worker_job import WorkerJobSearch
from bamengine.systems.labor_market import (
    adjust_minimum_wage,
    decide_wage_offer,
    firms_hire,
    workers_prepare_applications,
    workers_send_one_round,
)


def test_adjust_minimum_wage() -> None:
    # build an economy with known price path
    ec = Economy(
        min_wage=1.0,
        avg_mrkt_price_history=np.array([1.0, 1.05, 1.10, 1.12, 1.15]),  # t = 4
        min_wage_rev_period=4,
    )
    adjust_minimum_wage(ec)
    # inflation between P_0 and P_3 (t-1) : (1.12-1.0)/1.0 = 0.12
    assert np.isclose(ec.min_wage, 1.0 * 1.12)


def test_decide_wage_offer() -> None:
    rng = default_rng(seed=42)

    fw = FirmWageOffer(
        wage_prev=np.array([1.0, 1.2, 1.1, 1.3]),
        n_vacancies=np.array([3, 0, 2, 0]),  # 2 firms hiring
        wage_offer=np.zeros(4),
    )

    decide_wage_offer(fw, w_min=1.05, h_xi=0.1, rng=rng)

    # Firms with no n_vacancies should offer max(w_min , wage_prev) == w_min
    assert np.isclose(fw.wage_offer[1], 1.2)  # higher than min → unchanged
    assert np.isclose(fw.wage_offer[3], 1.3)  # higher than min → unchanged

    # Firms hiring should have wage >= w_min and >= wage_prev
    assert (fw.wage_offer[fw.n_vacancies > 0] >= fw.wage_prev[fw.n_vacancies > 0]).all()
    assert (fw.wage_offer >= 1.05).all()


def _mini_state() -> Tuple[WorkerJobSearch, FirmHiring, Generator, int]:
    rng = default_rng(seed=1)
    n_workers, n_firms, M = 6, 3, 2

    ws = WorkerJobSearch(
        employed=np.array([0, 0, 0, 1, 0, 1], dtype=np.int64),
        employer_prev=np.full(n_workers, -1, dtype=np.int64),
        contract_expired=np.zeros(n_workers, dtype=np.int64),
        fired=np.zeros(n_workers, dtype=np.int64),
        apps_head=np.full(n_workers, -1, dtype=np.int64),
        apps_targets=np.full((n_workers, M), -1, dtype=np.int64),
    )
    fh = FirmHiring(
        wage_offer=np.array([1.2, 1.5, 1.3]),
        n_vacancies=np.array([2, 1, 0], dtype=np.int64),
        recv_apps_head=np.full(n_firms, -1, dtype=np.int64),
        recv_apps=np.full((n_firms, M), -1, dtype=np.int64),
    )
    return ws, fh, rng, M


def test_prepare_applications() -> None:
    ws, fh, rng, M = _mini_state()
    workers_prepare_applications(ws, fh, max_M=M, rng=rng)

    # unemployed workers must have a non-negative head ptr
    heads = ws.apps_head[ws.employed == 0]
    assert (heads >= 0).all()

    # ------------------------------------------------------------------
    # Only check the rows actually used by unemployed workers
    rows = heads // M  # row index in the apps_targets buffer
    targets = ws.apps_targets[rows]  # shape: (U, M)

    assert (targets >= 0).all()  # no sentinel values
    assert (targets < fh.wage_offer.size).all()  # valid firm indices


def test_full_round() -> None:
    ws, fh, rng, M = _mini_state()
    workers_prepare_applications(ws, fh, max_M=M, rng=rng)

    # simulate one full hiring event
    for _ in range(M):
        workers_send_one_round(ws, fh)
        firms_hire(ws, fh, contract_theta=8)

    # every firm received at most its original n_vacancies
    assert (fh.n_vacancies >= 0).all()
    # hired workers are now marked employed
    assert ws.employed.sum() >= 1  # at least one hire
    # no worker keeps a pending head ptr
    assert (ws.apps_head[ws.employed == 1] == -1).all()
