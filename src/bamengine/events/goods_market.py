"""Goods market events for consumption decisions and shopping.

Each event encapsulates consumption propensity calculation, shopping decisions,
and purchase execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


@event
class ConsumersCalcPropensity:
    """
    Calculate household marginal propensity to consume
    based on current savings relative to average.

    Rule
    ----
        c = 1 / (1 + tanh(SA / SA_avg) ** β)

    Where:
        c: Propensity to consume (0 to 1)
        SA: Current consumer savings
        SA_avg: Average savings across all consumers
        β: Sensitivity parameter for consumption behavior
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.goods_market import consumers_calc_propensity

        _avg_sav = float(sim.con.savings.mean())
        consumers_calc_propensity(sim.con, avg_sav=_avg_sav, beta=sim.config.beta)


@event
class ConsumersDecideIncomeToSpend:
    """
    Determine how much of disposable wealth each consumer will spend this period.

    Rule
    ----
        wealth = savings + income
        income_to_spend = wealth × propensity
        savings = wealth - income_to_spend
        income = 0  (reset after allocation)

    This function converts disposable income and existing savings into spending budget
    and updated savings, based on the consumer's propensity to spend.
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.goods_market import (
            consumers_decide_income_to_spend,
        )

        consumers_decide_income_to_spend(sim.con)


@event
class ConsumersDecideFirmsToVisit:
    """
    Each consumer with spending budget selects up to max_Z firms to potentially visit.

    Selection Strategy

    1. Loyalty Rule:
       Previous period's "largest producer visited" gets slot 0 iff still has inventory
    2. Random Sampling:
       Remaining slots filled by randomly sampling from available firms
    3. Price Sorting:
       Sampled firms sorted by price (cheapest first) for optimal shopping
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.goods_market import (
            consumers_decide_firms_to_visit,
        )

        consumers_decide_firms_to_visit(
            sim.con,
            sim.prod,
            max_Z=sim.config.max_Z,
            rng=sim.rng,
        )


@event
class ConsumersShopOneRound:
    """
    Execute one complete round of shopping across all consumers with spending budget.

    Shopping Process

    1. Identify all consumers with remaining spending budget
    2. Randomize shopping order to ensure fairness
    3. Each consumer attempts to purchase from their next queued firm
    4. Update inventory, spending budget, and loyalty tracking
    5. Advance shopping queue pointers

    A consumer's shopping ends when they either:
    - Exhaust their spending budget, or
    - Visit all firms in their queue
    """

    def execute(self, sim: Simulation) -> None:
        """Execute one shopping round."""
        from bamengine.events._internal.goods_market import consumers_shop_one_round

        consumers_shop_one_round(sim.con, sim.prod, rng=sim.rng)


@event
class ConsumersFinalizePurchases:
    """
    Complete the shopping process by moving any unspent income back to savings.

    Rule
    ----
        savings = savings + income_to_spend
        income_to_spend = 0

    This ensures that any budget not spent during the shopping rounds is preserved
    as savings for future periods, maintaining wealth conservation.
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.goods_market import consumers_finalize_purchases

        consumers_finalize_purchases(sim.con)
