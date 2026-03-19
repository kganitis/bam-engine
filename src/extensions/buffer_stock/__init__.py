"""Buffer-stock consumption extension (Section 3.9.4).

This extension implements Section 3.9.4 of Delli Gatti et al. (2011),
replacing the baseline mean-field MPC with an individual adaptive rule
based on buffer-stock saving theory.

Usage::

    from extensions.buffer_stock import BUFFER_STOCK

    sim = bam.Simulation.init(**config)
    sim.use(BUFFER_STOCK)
    results = sim.run()

Or manually::

    from extensions.buffer_stock import (
        BufferStock,
        BUFFER_STOCK_EVENTS,
        BUFFER_STOCK_CONFIG,
    )

    sim = bam.Simulation.init(**config)
    sim.use_role(BufferStock, n_agents=sim.n_households)
    sim.use_events(*BUFFER_STOCK_EVENTS)
    sim.use_config(BUFFER_STOCK_CONFIG)
    results = sim.run()

Components
----------
BufferStock : role
    Tracks ``prev_income`` and ``propensity`` per household.
ConsumersCalcBufferStockPropensity : event
    Replaces ``consumers_calc_propensity``.
ConsumersDecideBufferStockSpending : event
    Replaces ``consumers_decide_income_to_spend``.

Design Decisions
----------------
**Pipeline hook strategy** — Both ``consumers_calc_propensity`` *and*
``consumers_decide_income_to_spend`` are replaced via ``@event(replace=...)``.
The buffer-stock rule changes how MPC is computed (no mean-field) *and*
how it is applied (income-only for employed, savings-only for unemployed),
so both events must be replaced. Keeping them as two separate events
preserves the baseline pipeline structure (propensity -> spending).

**Income tracking** — ``prev_income`` is stored on the ``BufferStock`` role
(not on ``Consumer``) to keep ``src/`` untouched. ``con.income`` is still
zeroed after consumption, preserving compatibility with the wage payment
mechanism (``workers_receive_wage`` adds wage to ``con.income`` in-place).

**Desired savings-income ratio (h)** — A single homogeneous config
parameter (``buffer_stock_h``) for all households. The book mentions a
"personal desired ratio" but does not specify a distribution. Starting
homogeneous lets validation (Figure 3.8) determine if heterogeneity is
needed.

**MPC bounds and dissaving** — No explicit ``c_max``. The formula
``c_t = 1 + (d - h*g)/(1+g)`` naturally produces ``c > 1`` when income
drops (negative g), which is dissaving from savings. Budget is capped at
total wealth (``savings + income``) to prevent negative savings; ``c`` is
clipped at 0 minimum to prevent negative consumption.

**Unemployed households** — Three-case MPC logic:

=============  =====================  ===========================  ==================
Case           Condition              MPC formula                  Consumption source
=============  =====================  ===========================  ==================
Normal         W > 0 and W_prev > 0   Eq. 3.20 (g clamped -0.99)   Income (c * W)
Fresh start    W > 0 and W_prev <= 0  c = 1 - h + S/W              Income (c * W)
Unemployed     W <= 0                 c = 1/h                      Savings (c * S)
=============  =====================  ===========================  ==================

At the target buffer ``S = h·W``, first-period unemployment spending equals
``(1/h)·h·W = W`` — the last employed income (consumption smoothing). The
drawdown is geometric: after ``h`` periods, ~37% of savings remain (``(1-1/h)^h
→ 1/e``).

**Dividends** — Stay in savings, not counted as income W. The buffer-stock
MPC is driven by labor income dynamics only. The core ``Shareholder`` role
tracks per-period dividends so validation metrics can subtract the dividend
artifact from MPC calculations (``adjusted_c = c - D/W``).

**Composability** — Fully composable with R&D/Growth+. RnD modifies firm
behavior (productivity); BufferStock modifies consumer behavior (MPC).
They operate on different agent types with no conflicts.

**Distribution fitting** — Wealth CCDF is fitted with Singh-Maddala
(``scipy.stats.burr12``), Dagum (``scipy.stats.mielke``), and GB2
(``scipy.stats.betaprime``) distributions for validation against
Figure 3.8. Singh-Maddala and Dagum have ``x^k`` terms in their PDFs;
unconstrained MLE sometimes finds extreme ``k`` (>100), causing overflow.
The validation fitting uses a profile likelihood fallback: when the
default fit overflows, ``k`` is fixed at values in [5..50] and the
remaining parameters are optimized, selecting the ``k`` with maximum
log-likelihood.
"""

from __future__ import annotations

from typing import Any

from bamengine import Extension
from extensions.buffer_stock.events import (
    ConsumersCalcBufferStockPropensity,
    ConsumersDecideBufferStockSpending,
)
from extensions.buffer_stock.role import BufferStock

BUFFER_STOCK_EVENTS = [
    ConsumersCalcBufferStockPropensity,
    ConsumersDecideBufferStockSpending,
]

BUFFER_STOCK_CONFIG = {
    "buffer_stock_h": 2.0,
}

BUFFER_STOCK = Extension(
    roles={BufferStock: "households"},
    events=BUFFER_STOCK_EVENTS,
    relationships=[],
    config_dict=BUFFER_STOCK_CONFIG,
)

BUFFER_STOCK_COLLECT: dict[str, Any] = {
    "Producer": ["production"],
    "Worker": ["wage", "employed"],
    "Employer": ["n_vacancies"],
    "Borrower": ["net_worth"],
    "Consumer": ["savings"],
    "BufferStock": ["propensity"],
    "Shareholder": ["dividends"],
    "LoanBook": ["principal", "rate"],
    "capture_timing": {
        "Worker.wage": "firms_run_production",
        "Worker.employed": "firms_run_production",
        "Producer.production": "firms_run_production",
        "Employer.n_vacancies": "firms_decide_vacancies",
        "Borrower.net_worth": "firms_run_production",
        "Consumer.savings": None,  # end of period
        "BufferStock.propensity": "consumers_calc_buffer_stock_propensity",
        "Shareholder.dividends": "consumers_calc_buffer_stock_propensity",
        "LoanBook.principal": "credit_market_round",
        "LoanBook.rate": "credit_market_round",
    },
}

__all__ = [
    "BufferStock",
    "BUFFER_STOCK",
    "BUFFER_STOCK_EVENTS",
    "BUFFER_STOCK_CONFIG",
    "BUFFER_STOCK_COLLECT",
    "ConsumersCalcBufferStockPropensity",
    "ConsumersDecideBufferStockSpending",
]
