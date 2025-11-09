"""Goods market events for consumption decisions and shopping.

This module contains Event classes that wrap goods market system functions.
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
    Consumers calculate propensity to consume based on savings.

    Propensity to consume is calculated using a logistic function:
        c_j = 1 / (1 + exp(-β · (S_j - S̄)))

    where S_j is individual savings, S̄ is average savings, and β controls
    the sensitivity.

    This event wraps `bamengine.systems.goods_market.consumers_calc_propensity`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute propensity calculation."""
        from bamengine.systems.goods_market import consumers_calc_propensity

        _avg_sav = float(sim.con.savings.mean())
        consumers_calc_propensity(sim.con, avg_sav=_avg_sav, beta=sim.config.beta)


@event
class ConsumersDecideIncomeToSpend:
    """
    Consumers decide how much income to spend on consumption.

    Consumption budget is:
        budget_j = c_j · S_j

    where c_j is propensity to consume and S_j is savings.

    This event wraps `bamengine.systems.goods_market.consumers_decide_income_to_spend`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute spending decision."""
        from bamengine.systems.goods_market import consumers_decide_income_to_spend

        consumers_decide_income_to_spend(sim.con)


@event
class ConsumersDecideFirmsToVisit:
    """
    Consumers select firms to visit for shopping.

    Consumers choose up to max_Z firms with positive inventory, weighted by
    inventory levels (firms with more inventory more likely to be selected).

    This event wraps `bamengine.systems.goods_market.consumers_decide_firms_to_visit`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute firm selection decision."""
        from bamengine.systems.goods_market import consumers_decide_firms_to_visit

        consumers_decide_firms_to_visit(
            sim.con,
            sim.prod,
            max_Z=sim.config.max_Z,
            rng=sim.rng,
        )


@event
class ConsumersShopOneRound:
    """
    Consumers visit one firm and attempt to purchase goods.

    Each consumer processes one firm from their shopping queue, calculates
    desired quantity, and makes purchase if firm has inventory.

    Note: This event is typically repeated max_Z times in the pipeline.

    This event wraps `bamengine.systems.goods_market.consumers_shop_one_round`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute one shopping round."""
        from bamengine.systems.goods_market import consumers_shop_one_round

        consumers_shop_one_round(sim.con, sim.prod, rng=sim.rng)


@event
class ConsumersFinalizePurchases:
    """
    Consumers finalize purchases by deducting spending from savings.

    After all shopping rounds, consumers update savings:
        S_j(t) = S_j(t-1) - total_spending_j

    This event wraps `bamengine.systems.goods_market.consumers_finalize_purchases`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute purchase finalization."""
        from bamengine.systems.goods_market import consumers_finalize_purchases

        consumers_finalize_purchases(sim.con)
