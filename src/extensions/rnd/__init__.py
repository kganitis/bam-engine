"""R&D extension for endogenous productivity growth.

This extension implements Section 3.8 of Delli Gatti et al. (2011),
adding endogenous productivity growth through R&D investment.

Usage::

    from extensions.rnd import RND

    sim = bam.Simulation.init(**config)
    sim.use(RND)
    results = sim.run()

Or manually::

    from extensions.rnd import RnD, RND_EVENTS, RND_CONFIG

    sim = bam.Simulation.init(**config)
    sim.use_role(RnD)
    sim.use_events(*RND_EVENTS)
    sim.use_config(RND_CONFIG)
    results = sim.run()

Components:
    - RnD: Role tracking R&D investment and productivity increments
    - FirmsComputeRnDIntensity: Event computing R&D share and intensity
    - FirmsApplyProductivityGrowth: Event applying productivity gains
    - FirmsDeductRnDExpenditure: Event adjusting net profit for R&D expenditure
    - RND_EVENTS: List of all R&D event classes for use with ``sim.use_events()``
    - RND_CONFIG: Default R&D parameters for use with ``sim.use_config()``
    - RND: Pre-built :class:`~bamengine.Extension` bundle for ``sim.use()``
    - RND_COLLECT: Suggested data-collection config for ``sim.run(collect=...)``
"""

from __future__ import annotations

from typing import Any

from bamengine import Extension
from extensions.rnd.events import (
    FirmsApplyProductivityGrowth,
    FirmsComputeRnDIntensity,
    FirmsDeductRnDExpenditure,
)
from extensions.rnd.role import RnD

RND_EVENTS = [
    FirmsComputeRnDIntensity,
    FirmsApplyProductivityGrowth,
    FirmsDeductRnDExpenditure,
]

RND_CONFIG = {
    "sigma_min": 0.0,
    "sigma_max": 0.1,
    "sigma_decay": -1.0,
}

RND = Extension(
    roles={RnD: "firms"},
    events=RND_EVENTS,
    relationships=[],
    config_dict=RND_CONFIG,
)

RND_COLLECT: dict[str, Any] = {
    "Producer": ["production", "labor_productivity", "price", "inventory"],
    "Worker": ["wage", "employed"],
    "Employer": ["n_vacancies"],
    "Borrower": ["net_worth", "gross_profit", "total_funds"],
    "Consumer": ["income_to_spend"],
    "LoanBook": ["principal", "rate", "source_ids"],
    "Economy": True,
    "capture_timing": {
        "Worker.wage": "firms_run_production",
        "Worker.employed": "firms_run_production",
        "Producer.production": "firms_run_production",
        "Producer.labor_productivity": "firms_apply_productivity_growth",
        "Producer.price": "firms_adjust_price",
        "Producer.inventory": "consumers_finalize_purchases",
        "Employer.n_vacancies": "firms_decide_vacancies",
        "Borrower.net_worth": "firms_run_production",
        "Borrower.gross_profit": "firms_collect_revenue",
        "Borrower.total_funds": "firms_collect_revenue",
        "Consumer.income_to_spend": "consumers_decide_income_to_spend",
        "LoanBook.principal": "credit_market_round",
        "LoanBook.rate": "credit_market_round",
        "LoanBook.source_ids": "credit_market_round",
    },
}

__all__ = [
    "RnD",
    "RND",
    "RND_EVENTS",
    "RND_CONFIG",
    "RND_COLLECT",
    "FirmsComputeRnDIntensity",
    "FirmsApplyProductivityGrowth",
    "FirmsDeductRnDExpenditure",
]
