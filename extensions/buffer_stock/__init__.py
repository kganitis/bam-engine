"""Buffer-stock consumption extension (Section 3.9.3).

This extension implements Section 3.9.3 of Delli Gatti et al. (2011),
replacing the baseline mean-field MPC with an individual adaptive rule
based on buffer-stock saving theory.

Usage::

    from extensions.buffer_stock import BufferStock, attach_buffer_stock

    # Import BEFORE creating simulation (registers replacement events)
    sim = bam.Simulation.init(buffer_stock_h=1.0, **config)
    attach_buffer_stock(sim)
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

=============  =====================  ==========================  ==================
Case           Condition              MPC formula                 Consumption source
=============  =====================  ==========================  ==================
Normal         W > 0 and W_prev > 0   Eq. 3.20 (g clamped -0.99) Income (c * W)
Fresh start    W > 0 and W_prev <= 0  c = 1 - h + S/W            Income (c * W)
Unemployed     W <= 0                 c = 1/h                    Savings (c * S)
=============  =====================  ==========================  ==================

At the target buffer ``S = h·W``, first-period unemployment spending equals
``(1/h)·h·W = W`` — the last employed income (consumption smoothing). The
drawdown is geometric: after ``h`` periods, ~37% of savings remain (``(1-1/h)^h
→ 1/e``).

**Dividends** — Stay in savings, not counted as income W. The buffer-stock
MPC is driven by labor income dynamics only.

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

from typing import TYPE_CHECKING

import numpy as np

from extensions.buffer_stock.events import (
    ConsumersCalcBufferStockPropensity,
    ConsumersDecideBufferStockSpending,
)
from extensions.buffer_stock.role import BufferStock

if TYPE_CHECKING:
    import bamengine as bam


def attach_buffer_stock(sim: bam.Simulation) -> BufferStock:
    """Attach the BufferStock role to a simulation with household-sized arrays.

    Unlike ``sim.use_role()`` which defaults to ``n_firms``, this function
    creates the BufferStock role with ``n_households`` arrays, since the
    buffer-stock extension tracks per-household state.

    Parameters
    ----------
    sim : bam.Simulation
        Simulation instance to attach the role to.

    Returns
    -------
    BufferStock
        The attached role instance.
    """
    role_name = "BufferStock"
    if role_name in sim._role_instances:
        return sim._role_instances[role_name]

    instance = BufferStock(
        prev_income=np.zeros(sim.n_households, dtype=np.float64),
        propensity=np.zeros(sim.n_households, dtype=np.float64),
    )
    sim._role_instances[role_name] = instance
    return instance


__all__ = [
    "BufferStock",
    "ConsumersCalcBufferStockPropensity",
    "ConsumersDecideBufferStockSpending",
    "attach_buffer_stock",
]
