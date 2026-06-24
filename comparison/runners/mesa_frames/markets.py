"""Market matching functions for the mesa-frames (Polars) BAM model.

These mirror the Mesa port's market helpers in
``comparison/runners/mesa/markets.py`` one-to-one, preserving the matching
algorithm, the per-round queue-pop semantics, the priority/ordering rules, and
crucially the RNG draw points and their order so that the mesa-frames port is
structurally equivalent to the Mesa port (Task 10 equivalence gate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from comparison.runners.mesa_frames.model import BAMModel

EPS = 1e-9


def run_labor_market(model: BAMModel) -> None:
    """Event 11 (x max_M): multi-round labor market matching.

    Faithful translation of the Mesa port's ``run_labor_market``.

    Each round:

    1. Every still-unemployed worker (``employer < 0``) with a remaining queue
       entry offers its current front target (``job_app_{head}``) and advances
       its head by one.  Applications to ``-1`` targets or to firms with no
       vacancies are discarded (head still advances), exactly as the Mesa port
       pops then ``continue``-skips zero-vacancy targets.
    2. Remaining applicants are grouped by firm.  Firm processing order is the
       insertion order in which each firm first appears while scanning
       applicants in worker-row order, mirroring the Mesa port's
       ``applicants_by_firm`` dict insertion order.
    3. For each firm with vacancies, its applicant list is shuffled with
       ``model.random.shuffle`` (ONE shuffle per firm-with-applicants, in
       insertion order -- identical draw structure to the Mesa port), and the
       first ``n_vacancies`` are hired.

    On hire: ``employer = firm``, ``wage = firm.wage_offer`` (posted, not
    negotiated), ``periods_left = theta`` (EXACT), ``contract_expired = False``,
    ``fired = False``, queue head exhausted; ``firm.current_labor += 1``,
    ``firm.n_vacancies -= 1``.  Rationing is RANDOM (the per-firm shuffle).

    The DataFrames are read into Python lists/dicts once, the rounds run as a
    plain loop (the Mesa/bamengine version loops too), and the firm- and
    household-side results are written back to Polars once at the end.
    """
    max_M: int = int(model.p["max_M"])
    theta: int = int(model.p["theta"])
    rng = model.random

    hh = model.households
    fdf = model.firms.agents
    hdf = hh.agents

    # --- Firm-side per-round scalar maps keyed by firm unique_id. ---
    firm_ids: list[int] = fdf["unique_id"].to_list()
    vac: dict[int, int] = dict(
        zip(firm_ids, (int(v) for v in fdf["n_vacancies"]), strict=True)
    )
    wage_offer: dict[int, float] = dict(
        zip(firm_ids, (float(w) for w in fdf["wage_offer"]), strict=True)
    )
    cur_labor: dict[int, int] = dict(
        zip(firm_ids, (int(c) for c in fdf["current_labor"]), strict=True)
    )

    # --- Worker-side state (parallel lists in row order). ---
    worker_ids: list[int] = hdf["unique_id"].to_list()
    employer: list[int] = [int(e) for e in hdf["employer"]]
    head: list[int] = [int(h) for h in hdf["job_app_head"]]
    # Application queues: row-major matrix of -1-padded targets.
    queue_cols = [hdf[f"job_app_{k}"].to_list() for k in range(max_M)]

    # Hire results indexed by worker row.
    wage: list[float] = [float(w) for w in hdf["wage"]]
    periods_left: list[int] = [int(p) for p in hdf["periods_left"]]
    contract_expired: list[bool] = list(hdf["contract_expired"])
    fired: list[bool] = list(hdf["fired"])

    for _ in range(max_M):
        # --- Phase A: each active unemployed worker offers its front target. ---
        applicants_by_firm: dict[int, list[int]] = {}
        for i in range(len(worker_ids)):
            if employer[i] >= 0:
                continue
            h = head[i]
            if h >= max_M:
                continue
            target = queue_cols[h][i]
            head[i] = h + 1  # advance head (pop the front)
            if target < 0:
                continue
            if vac.get(target, 0) == 0:
                # Skip this target (advance past it) -- mirrors Mesa continue.
                continue
            applicants_by_firm.setdefault(target, []).append(i)

        # --- Phase B: per firm (insertion order), shuffle then hire. ---
        for firm_id, applicants in applicants_by_firm.items():
            n_vac = vac[firm_id]
            if n_vac == 0:
                continue
            # ONE shuffle per firm-with-applicants, in insertion order --
            # mirrors the Mesa port's model.random.shuffle(applicants).
            rng.shuffle(applicants)
            for i in applicants[:n_vac]:
                employer[i] = firm_id
                wage[i] = wage_offer[firm_id]
                periods_left[i] = theta
                contract_expired[i] = False
                fired[i] = False
                cur_labor[firm_id] += 1
                vac[firm_id] -= 1

    # --- Write firm-side results back (current_labor, n_vacancies). ---
    model.firms.agents = fdf.with_columns(
        pl.col("unique_id")
        .replace_strict(cur_labor, default=None)
        .cast(pl.Int64)
        .alias("current_labor"),
        pl.col("unique_id")
        .replace_strict(vac, default=None)
        .cast(pl.Int64)
        .alias("n_vacancies"),
    )

    # --- Write worker-side results back. ---
    hh.agents = hdf.with_columns(
        pl.Series("employer", employer, dtype=pl.Int64),
        pl.Series("wage", wage, dtype=pl.Float64),
        pl.Series("periods_left", periods_left, dtype=pl.Int64),
        pl.Series("contract_expired", contract_expired, dtype=pl.Boolean),
        pl.Series("fired", fired, dtype=pl.Boolean),
        pl.Series("job_app_head", head, dtype=pl.Int64),
    )
