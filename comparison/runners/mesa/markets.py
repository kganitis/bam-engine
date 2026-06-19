"""Market matching functions for the Mesa implementation of the BAM model."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comparison.runners.mesa.model import BamModel

EPS = 1e-9


def run_goods_market(model: BamModel) -> None:
    """Event 28 (REF §6.3): strictly-sequential goods market matching.

    Buyers are shuffled once, then processed one at a time.  Each buyer
    walks its price-sorted shop_visits fully before the next buyer acts,
    decrementing firm inventory IMMEDIATELY so subsequent buyers see the
    current stock.  A rationed buyer overflows to its next firm within its
    own turn.
    """
    buyers = [h for h in model.households if h.income_to_spend > EPS]
    if not buyers:
        return
    model.random.shuffle(buyers)
    for buyer in buyers:
        for firm in buyer.shop_visits:
            if buyer.income_to_spend <= EPS:
                break
            if firm.inventory <= EPS:
                continue
            qty = min(buyer.income_to_spend / firm.price, firm.inventory)
            spent = qty * firm.price
            buyer.income_to_spend -= spent
            firm.inventory -= qty


def run_labor_market(model: BamModel) -> None:
    """Event 11 (x max_M): multi-round labor market matching.

    Each round: still-unemployed workers with non-empty job_apps pop their
    front target; applications to firms with no vacancies are discarded;
    remaining applicants are grouped by firm, shuffled randomly, and the
    first n_vacancies are hired.  Non-hired workers retain their remaining
    job_apps for the next round.
    """
    max_M = model.p["max_M"]
    theta = model.p["theta"]

    for _ in range(max_M):
        # Gather still-unemployed workers who have pending applications.
        applicants_by_firm: dict = {}
        for h in model.households:
            if h.employed or not h.job_apps:
                continue
            target = h.job_apps.pop(0)
            if target.n_vacancies == 0:
                # Skip this target (already advance past it, keep rest).
                continue
            applicants_by_firm.setdefault(target, []).append(h)

        for firm, applicants in applicants_by_firm.items():
            if firm.n_vacancies == 0:
                continue
            model.random.shuffle(applicants)
            for h in applicants[: firm.n_vacancies]:
                # Hire
                h.employer = firm
                h.wage = firm.wage_offer
                h.periods_left = theta
                h.contract_expired = False
                h.fired = False
                h.job_apps = []
                firm.employees[h] = None
                firm.current_labor += 1
                firm.n_vacancies -= 1


def run_credit_market(model: BamModel) -> None:
    """Events 18 (×max_H): multi-round credit market matching.

    Each round: firms with credit_demand>0 and non-empty loan_apps pop their
    front bank; applicants are grouped by bank and ranked by projected_fragility
    ASC (deterministic, no random tie-break); loans are granted walking the
    ranked list maintaining a running total against credit_supply; the boundary
    applicant may receive a partial loan.  Loans accumulate across rounds.
    """
    from comparison.runners.mesa.agents import Loan

    max_H = model.p["max_H"]
    max_loan_to_net_worth = model.p["max_loan_to_net_worth"]
    r_bar = model.p["r_bar"]
    max_leverage = model.p["max_leverage"]

    for _ in range(max_H):
        # Group applications by bank for this round.
        applicants_by_bank: dict = {}
        for f in model.firms:
            if f.credit_demand <= 0 or not f.loan_apps:
                continue
            bank = f.loan_apps.pop(0)
            applicants_by_bank.setdefault(bank, []).append(f)

        for bank, applicants in applicants_by_bank.items():
            supply = bank.credit_supply
            if supply <= EPS:
                continue
            # Rank by projected_fragility ASC (safer firms first, no random tie-break).
            applicants.sort(key=lambda f: f.projected_fragility)
            granted_total = 0.0
            for f in applicants:
                if granted_total >= supply:
                    break
                # Per-loan cap.
                if max_loan_to_net_worth > 0:
                    max_grant = min(
                        f.credit_demand, f.net_worth * max_loan_to_net_worth
                    )
                else:
                    max_grant = f.credit_demand
                if max_grant <= 0:
                    continue
                remaining = supply - granted_total
                amount = (
                    max_grant
                    if granted_total + max_grant <= supply
                    else max(remaining, 0.0)
                )
                if amount > EPS:
                    fragility = min(f.projected_fragility, max_leverage)
                    rate = r_bar * (1.0 + bank.opex_shock * fragility)
                    f.loans.append(Loan(principal=amount, rate=rate, lender=bank))
                    f.total_funds += amount
                    f.credit_demand -= amount
                    granted_total += amount
            # Update bank supply once after all applicants processed.
            bank.credit_supply -= granted_total
