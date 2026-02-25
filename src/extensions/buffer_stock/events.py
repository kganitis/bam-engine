"""Buffer-stock consumption events.

This module provides two replacement events for the buffer-stock consumption
extension (Section 3.9.4 of Delli Gatti et al., 2011).

The extension replaces the baseline mean-field MPC with an individual adaptive
rule based on buffer-stock saving theory. Each household maintains a personal
desired savings-income ratio and adjusts consumption to keep that ratio constant.

Key Equations
-------------

**Buffer-Stock MPC (Equation 3.20):**

.. math::

    c_t = 1 + \\frac{d_t - h \\cdot g_t}{1 + g_t}

Where:
- :math:`h` is the desired savings-income ratio (config parameter)
- :math:`g_t = W_t / W_{t-1} - 1` is the income growth rate
- :math:`d_t = S_t / W_{t-1} - h` is the divergence from desired ratio

**Fresh Start Formula (derived):**

.. math::

    c_t = 1 - h + S_t / W_t

Applied when a household has just been re-employed (no previous income).
"""

from __future__ import annotations

import numpy as np

import bamengine as bam
from bamengine import event, ops


@event(replace="consumers_calc_propensity")
class ConsumersCalcBufferStockPropensity:
    """Compute individual MPC using buffer-stock adaptive rule.

    Three-case logic:

    1. **Normal** (prev_income > 0 and income > 0):
       Full buffer-stock formula (Eq. 3.20).
    2. **Fresh start** (prev_income <= 0 and income > 0):
       Just re-employed: ``c = 1 - h + S/W``.
    3. **Unemployed** (income <= 0):
       ``c = 1/h`` (gradual savings drawdown).
    """

    def execute(self, sim: bam.Simulation) -> None:
        """Execute buffer-stock propensity calculation."""
        con = sim.get_role("Consumer")
        buf = sim.get_role("BufferStock")

        income = con.income
        prev_income = buf.prev_income
        savings = con.savings
        h = sim.buffer_stock_h

        # Case masks
        normal = (prev_income > 0) & (income > 0)
        fresh = (prev_income <= 0) & (income > 0)
        # unemployed = income <= 0  (handled by default c=1/h)

        c = np.full(len(income), 1.0 / h)  # default: c=1/h (unemployed drawdown)

        # Normal case: Eq. 3.20
        if np.any(normal):
            g = income[normal] / prev_income[normal] - 1.0
            g = np.maximum(g, -0.99)  # clamp singularity at g=-1
            d = savings[normal] / prev_income[normal] - h
            c[normal] = 1.0 + (d - h * g) / (1.0 + g)

        # Fresh start: just re-employed
        if np.any(fresh):
            c[fresh] = 1.0 - h + savings[fresh] / income[fresh]

        # Floor at 0 (no negative consumption)
        c = np.maximum(c, 0.0)
        ops.assign(buf.propensity, c)


@event(replace="consumers_decide_income_to_spend")
class ConsumersDecideBufferStockSpending:
    """Allocate spending budget using buffer-stock MPC.

    Two-path consumption:

    - **Employed** (income > 0): budget = c * income
    - **Unemployed** (income <= 0): budget = c * savings

    Budget is capped at total wealth (savings + income) and floored at 0.
    Savings are updated and floored at 0 (non-negative).
    """

    def execute(self, sim: bam.Simulation) -> None:
        """Execute buffer-stock spending allocation."""
        con = sim.get_role("Consumer")
        buf = sim.get_role("BufferStock")

        c = buf.propensity
        income = con.income
        savings = con.savings

        # Two-path consumption: employed from income, unemployed from savings
        employed = income > 0
        budget = np.where(employed, c * income, c * savings)
        budget = np.minimum(budget, savings + income)  # cap at wealth
        budget = np.maximum(budget, 0.0)  # safety floor

        # Save prev_income before zeroing
        ops.assign(buf.prev_income, income)

        # Allocate spending budget
        ops.assign(con.income_to_spend, budget)

        # Unified savings update: works for both employed and unemployed
        # For employed: savings + income - budget (retain unspent income)
        # For unemployed: savings + 0 - budget (debit savings)
        new_savings = savings + income - budget
        new_savings = np.maximum(new_savings, 0.0)  # floor at 0
        ops.assign(con.savings, new_savings)

        # Zero income for next period
        ops.assign(con.income, 0.0)
