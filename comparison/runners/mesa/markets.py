"""Market matching functions for the Mesa implementation of the BAM model."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comparison.runners.mesa.model import BamModel


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
                firm.employees.add(h)
                firm.current_labor += 1
                firm.n_vacancies -= 1
