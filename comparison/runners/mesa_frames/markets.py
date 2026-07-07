"""Market matching functions for the mesa-frames (Polars) BAM model.

These mirror the Mesa port's market helpers in
``comparison/runners/mesa/markets.py`` one-to-one, preserving the matching
algorithm, the per-round queue-pop semantics, the priority/ordering rules, and
crucially the RNG draw points and their order so that the mesa-frames port is
structurally equivalent to the Mesa port (cross-port equivalence gate).
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


def run_credit_market(model: BAMModel) -> None:
    """Event 18 (x max_H): multi-round credit market matching.

    Faithful translation of the Mesa port's ``run_credit_market``.

    Each round:

    1. Every firm with ``credit_demand > 0`` and a remaining queue entry offers
       its current front bank target (``loan_app_{head}``) and advances its head
       by one.  Applications to ``-1`` targets are discarded (head still
       advances).  (The Mesa port pops the front bank unconditionally for every
       firm with demand and a non-empty list; banks with no supply are filtered
       later when their supply is checked, so we mirror that by grouping all
       offered targets and skipping zero-supply banks at grant time.)
    2. Remaining applicants are grouped by bank.  Bank processing order is the
       insertion order in which each bank first appears while scanning applicants
       in firm-row order, mirroring the Mesa port's ``applicants_by_bank`` dict
       insertion order.
    3. For each bank with ``credit_supply > EPS``, its applicant list is RANKED
       ASC by ``projected_fragility`` (safer firms first; NO random tie-break --
       DETERMINISTIC).  Loans are granted walking the ranked list, maintaining a
       running total against ``credit_supply``; the boundary applicant may
       receive a PARTIAL loan.  The loan contract rate is fragility-scaled:
       ``rate = r_bar * (1 + opex_shock[bank] * min(fragility, max_leverage))``
       (DIFFERENT from the posted ranking rate).

    On grant: a loan row (borrower_id, lender_id, principal, interest_rate) is
    appended to ``model.loans``; ``firm.total_funds += amount``;
    ``firm.credit_demand -= amount``; ``bank.credit_supply -= amount`` (applied
    once per bank after all its applicants are processed, matching the Mesa
    port).  Loans accumulate across rounds.

    NO RNG is drawn here (deterministic fragility ranking) -- identical to the
    Mesa port.  The DataFrames are read into Python lists/dicts once, the rounds
    run as a plain loop, and the firm-/bank-side results plus the loan table are
    written back to Polars once at the end.
    """
    max_H: int = int(model.p["max_H"])
    max_loan_to_net_worth: float = float(model.p["max_loan_to_net_worth"])
    r_bar: float = float(model.p["r_bar"])
    max_leverage: float = float(model.p["max_leverage"])

    fdf = model.firms.agents
    bdf = model.banks.agents

    # --- Firm-side state (parallel lists in row order). ---
    firm_ids: list[int] = fdf["unique_id"].to_list()
    credit_demand: list[float] = [float(d) for d in fdf["credit_demand"]]
    fragility: list[float] = [float(f) for f in fdf["projected_fragility"]]
    net_worth: list[float] = [float(nw) for nw in fdf["net_worth"]]
    total_funds: list[float] = [float(t) for t in fdf["total_funds"]]
    head: list[int] = [int(h) for h in fdf["loan_app_head"]]
    queue_cols = [fdf[f"loan_app_{k}"].to_list() for k in range(max_H)]

    # --- Bank-side per-round scalar maps keyed by bank unique_id. ---
    bank_ids: list[int] = bdf["unique_id"].to_list()
    supply: dict[int, float] = dict(
        zip(bank_ids, (float(s) for s in bdf["credit_supply"]), strict=True)
    )
    opex_shock: dict[int, float] = dict(
        zip(bank_ids, (float(o) for o in bdf["opex_shock"]), strict=True)
    )

    # Accumulated loan rows for this period (borrower, lender, principal, rate).
    loan_rows: list[tuple[int, int, float, float]] = []

    for _ in range(max_H):
        # --- Phase A: each active demanding firm offers its front bank. ---
        applicants_by_bank: dict[int, list[int]] = {}
        for r in range(len(firm_ids)):
            if credit_demand[r] <= 0.0:
                continue
            h = head[r]
            if h >= max_H:
                continue
            target = queue_cols[h][r]
            head[r] = h + 1  # advance head (pop the front)
            if target < 0:
                continue
            applicants_by_bank.setdefault(target, []).append(r)

        # --- Phase B: per bank (insertion order), rank then grant. ---
        for bank_id, applicant_rows in applicants_by_bank.items():
            bank_supply = supply.get(bank_id, 0.0)
            if bank_supply <= EPS:
                continue
            # Rank applicants ASC by projected_fragility (safer first; no random
            # tie-break) -- mirrors the Mesa port's applicants.sort(key=...).
            applicant_rows.sort(key=lambda rr: fragility[rr])
            granted_total = 0.0
            for rr in applicant_rows:
                if granted_total >= bank_supply:
                    break
                # Per-loan cap (max_loan_to_net_worth * net_worth) if param > 0.
                if max_loan_to_net_worth > 0:
                    max_grant = min(
                        credit_demand[rr], net_worth[rr] * max_loan_to_net_worth
                    )
                else:
                    max_grant = credit_demand[rr]
                if max_grant <= 0:
                    continue
                remaining = bank_supply - granted_total
                amount = (
                    max_grant
                    if granted_total + max_grant <= bank_supply
                    else max(remaining, 0.0)
                )
                if amount > EPS:
                    frag = min(fragility[rr], max_leverage)
                    rate = r_bar * (1.0 + opex_shock[bank_id] * frag)
                    loan_rows.append((firm_ids[rr], bank_id, amount, rate))
                    total_funds[rr] += amount
                    credit_demand[rr] -= amount
                    granted_total += amount
            # Update bank supply once after all applicants processed.
            supply[bank_id] = bank_supply - granted_total

    # --- Write firm-side results back (credit_demand, total_funds, head). ---
    model.firms.agents = fdf.with_columns(
        pl.Series("credit_demand", credit_demand, dtype=pl.Float64),
        pl.Series("total_funds", total_funds, dtype=pl.Float64),
        pl.Series("loan_app_head", head, dtype=pl.Int64),
    )

    # --- Write bank-side results back (credit_supply). ---
    model.banks.agents = bdf.with_columns(
        pl.col("unique_id")
        .replace_strict(supply, default=None)
        .cast(pl.Float64)
        .alias("credit_supply")
    )

    # --- Append accumulated loans to the model's loan relationship table. ---
    if loan_rows:
        new_loans = pl.DataFrame(
            {
                "borrower_id": [row[0] for row in loan_rows],
                "lender_id": [row[1] for row in loan_rows],
                "principal": [row[2] for row in loan_rows],
                "interest_rate": [row[3] for row in loan_rows],
            },
            schema={
                "borrower_id": pl.Int64,
                "lender_id": pl.Int64,
                "principal": pl.Float64,
                "interest_rate": pl.Float64,
            },
        )
        model.loans = pl.concat([model.loans, new_loans], how="vertical")


def run_goods_market(model: BAMModel) -> None:
    """Event 28 (x1): strictly-sequential goods market matching.

    Faithful translation of the Mesa port's ``run_goods_market`` in
    ``comparison/runners/mesa/markets.py``.

    Buyers are shuffled ONCE via ``model.random.shuffle`` (the same shuffle
    point as the Mesa port), then processed one at a time.  Each buyer walks
    its price-sorted ``shop_visit_0..shop_visit_{max_Z-1}`` targets fully
    before the next buyer acts, decrementing firm inventory IMMEDIATELY so
    subsequent buyers see the current stock.  A rationed buyer overflows to its
    next firm within its own turn.

    RNG: ONE ``model.random.shuffle`` on the list of active buyer row-indices,
    drawn at the same point as the Mesa port's ``model.random.shuffle(buyers)``.
    No per-buyer or per-visit draws.

    The DataFrames are read into Python lists once, the loop runs entirely in
    Python (cannot be vectorized -- see GOODS_MARKET_VECTORIZATION.md), and the
    firm inventory and household income_to_spend columns are written back once
    at the end.
    """
    max_Z: int = int(model.p["max_Z"])
    rng = model.random

    hdf = model.households.agents
    fdf = model.firms.agents

    # --- Household-side state (row-indexed parallel lists). ---
    income_to_spend: list[float] = [float(b) for b in hdf["income_to_spend"]]
    # Shop-visit queues: list of max_Z columns, each a list of firm unique_ids.
    visit_cols: list[list[int]] = [
        hdf[f"shop_visit_{k}"].to_list() for k in range(max_Z)
    ]

    # --- Firm-side state (mapped by firm unique_id). ---
    firm_ids: list[int] = fdf["unique_id"].to_list()
    # inventory and price as dicts for O(1) lookup by firm unique_id.
    inventory: dict[int, float] = dict(
        zip(firm_ids, (float(v) for v in fdf["inventory"]), strict=True)
    )
    price_of: dict[int, float] = dict(
        zip(firm_ids, (float(p) for p in fdf["price"]), strict=True)
    )

    # --- Gather active buyers (those with income_to_spend > EPS). ---
    # Mirror the Mesa port: ``buyers = [h for h in model.households if h.income_to_spend > EPS]``
    buyers: list[int] = [i for i, b in enumerate(income_to_spend) if b > EPS]
    if not buyers:
        return

    # --- Shuffle buyers ONCE via model.random -- IDENTICAL shuffle point to Mesa port. ---
    rng.shuffle(buyers)

    # --- Sequential shopping loop (mirrors Mesa port exactly). ---
    for i in buyers:
        for k in range(max_Z):
            if income_to_spend[i] <= EPS:
                break
            fid = visit_cols[k][i]
            if fid < 0:
                break  # end of this buyer's visit list
            if inventory.get(fid, 0.0) <= EPS:
                continue
            qty = min(income_to_spend[i] / price_of[fid], inventory[fid])
            spent = qty * price_of[fid]
            income_to_spend[i] -= spent
            inventory[fid] -= qty

    # --- Write household-side results back (income_to_spend updated by spending). ---
    model.households.agents = hdf.with_columns(
        pl.Series("income_to_spend", income_to_spend, dtype=pl.Float64)
    )

    # --- Write firm-side results back (inventory depleted by sales). ---
    model.firms.agents = fdf.with_columns(
        pl.col("unique_id")
        .replace_strict(inventory, default=None)
        .cast(pl.Float64)
        .alias("inventory")
    )
