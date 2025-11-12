"""
Goods market events for consumption decisions and shopping.

This module defines the goods market phase events that execute after production.
Households calculate consumption propensity, allocate income to spending,
select firms to visit, and purchase goods through sequential shopping rounds.

Event Sequence
--------------
The goods market events execute in this order:

1. ConsumersCalcPropensity - Calculate propensity to consume based on savings
2. ConsumersDecideIncomeToSpend - Allocate income to spending budget
3. ConsumersDecideFirmsToVisit - Select firms to visit (sorted by price)
4. ConsumersShopOneRound - Execute shopping (repeated max_Z times)
5. ConsumersFinalizePurchases - Move unspent budget back to savings

The shopping rounds are repeated max_Z times to allow consumers to visit
multiple firms and find the best deals.

Design Notes
------------
- Events operate on consumer and producer roles (Consumer, Producer)
- Propensity to consume: c = 1 / (1 + tanh(SA/SA_avg)^β)
- Loyalty rule: consumers visit previous largest producer first (if inventory available)
- Consumers sorted by price (ascending) to prefer cheaper goods
- Shopping order randomized each round for fairness

Examples
--------
Execute goods market events:

>>> import bamengine as be
>>> sim = be.Simulation.init(n_firms=100, n_households=500, seed=42)
>>> # Goods market events run as part of default pipeline
>>> sim.step()

Execute individual goods market event:

>>> event = sim.get_event("consumers_calc_propensity")
>>> event.execute(sim)
>>> sim.con.propensity.mean()  # doctest: +SKIP
0.65

Check consumption:

>>> total_spent = sim.con.total_spent.sum()
>>> total_spent  # doctest: +SKIP
2850.0

See Also
--------
bamengine.events._internal.goods_market : System function implementations
Consumer : Consumption state
Producer : Production state with inventory
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:  # pragma: no cover
    from bamengine.simulation import Simulation


@event
class ConsumersCalcPropensity:
    """
    Calculate marginal propensity to consume based on relative savings.

    Households with below-average savings have higher propensity to consume
    (spend more), while those with above-average savings have lower propensity
    (save more). This implements consumption smoothing behavior.

    Algorithm
    ---------
    For each consumer j:

    1. Calculate relative savings: r_j = SA_j / SA_avg
    2. Apply propensity function: c_j = 1 / (1 + tanh(r_j)^β)

    Mathematical Notation
    ---------------------
    .. math::
        c_j = \\frac{1}{1 + \\tanh\\left(\\frac{SA_j}{\\overline{SA}}\\right)^\\beta}

    where:

    - c_j: propensity to consume (0 < c_j < 1)
    - SA_j: current savings of consumer j
    - SA_avg: average savings across all consumers
    - β: sensitivity parameter controlling consumption response (config)

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_households=500, seed=42)
    >>> event = sim.get_event("consumers_calc_propensity")
    >>> event.execute(sim)

    Check propensity distribution:

    >>> sim.con.propensity.mean()  # doctest: +SKIP
    0.65

    Verify range:

    >>> import numpy as np
    >>> (sim.con.propensity > 0).all() and (sim.con.propensity < 1).all()
    True

    High-savers have lower propensity:

    >>> high_savers = sim.con.savings > sim.con.savings.mean()
    >>> low_savers = sim.con.savings < sim.con.savings.mean()
    >>> sim.con.propensity[low_savers].mean() > sim.con.propensity[high_savers].mean()
    True

    Notes
    -----
    This event must execute first in goods market phase.

    Propensity is bounded: 0 < c < 1 (consumers always save something, always spend something).

    Higher β increases sensitivity to relative savings position.

    See Also
    --------
    ConsumersDecideIncomeToSpend : Uses propensity to allocate spending
    bamengine.events._internal.goods_market.consumers_calc_propensity : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.goods_market import consumers_calc_propensity

        _avg_sav = float(sim.con.savings.mean())
        consumers_calc_propensity(sim.con, avg_sav=_avg_sav, beta=sim.config.beta)


@event
class ConsumersDecideIncomeToSpend:
    """
    Allocate wealth to spending budget based on propensity to consume.

    Consumers combine their savings and income into total wealth, then allocate
    a portion to spending based on their propensity. The remainder stays as savings.

    Algorithm
    ---------
    For each consumer j:

    1. Calculate wealth: W_j = SA_j + I_j
    2. Allocate to spending: B_j = W_j × c_j
    3. Update savings: SA_j = W_j - B_j
    4. Reset income: I_j = 0

    Mathematical Notation
    ---------------------
    .. math::
        W_j = SA_j + I_j

        B_j = W_j \\times c_j

        SA_j = W_j - B_j = W_j(1 - c_j)

        I_j \\leftarrow 0

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_households=500, seed=42)
    >>> # First set propensity
    >>> sim.get_event("consumers_calc_propensity")().execute(sim)
    >>> # Then allocate spending
    >>> initial_wealth = sim.con.savings + sim.con.income
    >>> event = sim.get_event("consumers_decide_income_to_spend")
    >>> event.execute(sim)

    Check spending budget:

    >>> sim.con.income_to_spend.sum()  # doctest: +SKIP
    2950.0

    Verify wealth conservation:

    >>> import numpy as np
    >>> final_wealth = sim.con.savings + sim.con.income_to_spend
    >>> np.allclose(initial_wealth, final_wealth)
    True

    Income reset:

    >>> (sim.con.income == 0).all()
    True

    Notes
    -----
    This event must execute after ConsumersCalcPropensity (need propensity values).

    Wealth is conserved: initial_wealth = final_savings + spending_budget.

    Income is reset to 0 after allocation (will accumulate again next period).

    See Also
    --------
    ConsumersCalcPropensity : Calculates propensity used for allocation
    ConsumersShopOneRound : Uses income_to_spend as shopping budget
    bamengine.events._internal.goods_market.consumers_decide_income_to_spend : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.goods_market import (
            consumers_decide_income_to_spend,
        )

        consumers_decide_income_to_spend(sim.con)


@event
class ConsumersDecideFirmsToVisit:
    """
    Consumers select firms to visit, sorted by price (cheapest first).

    Consumers with spending budget build a shopping queue by sampling firms
    with inventory and sorting them by price. Loyalty rule: consumers prioritize
    the firm they bought most from last period (if inventory available).

    Algorithm
    ---------
    For each consumer j with B_j > 0 (spending budget):

    1. Sample min(max_Z, n_firms_with_inventory) firms randomly
    2. Sort sampled firms by price (ascending - prefer cheaper)
    3. Apply loyalty rule:
       - If prev_largest_producer has inventory:
         Move it to position 0 (top priority)
    4. Store sorted shopping queue

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, n_households=500, seed=42)
    >>> # Allocate spending first
    >>> sim.get_event("consumers_decide_income_to_spend")().execute(sim)
    >>> # Then select firms
    >>> event = sim.get_event("consumers_decide_firms_to_visit")
    >>> event.execute(sim)

    Check consumers with shopping plans:

    >>> import numpy as np
    >>> has_budget = sim.con.income_to_spend > 0
    >>> has_budget.sum()  # doctest: +SKIP
    480

    Verify firms sorted by price:

    >>> # Check one consumer's queue
    >>> consumers_with_budget = np.where(has_budget)[0]
    >>> if len(consumers_with_budget) > 0:
    ...     j = consumers_with_budget[0]
    ...     firm_ids = sim.con.shop_targets[j, :sim.config.max_Z]
    ...     firm_ids = firm_ids[firm_ids >= 0]
    ...     prices = sim.prod.price[firm_ids]
    ...     np.all(prices[:-1] <= prices[1:])  # Prices non-decreasing
    True

    Notes
    -----
    This event must execute after ConsumersDecideIncomeToSpend (need spending budget).

    Only consumers with positive spending budget prepare shopping queues.

    Loyalty rule implements realistic consumer behavior: stick with previous
    main supplier if they still have goods available.

    See Also
    --------
    ConsumersShopOneRound : Processes shopping queue
    bamengine.events._internal.goods_market.consumers_decide_firms_to_visit : Implementation
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
    Execute one shopping round where consumers purchase from one firm each.

    In each round, consumers with remaining budget visit their next queued firm
    and attempt to purchase goods. Shopping order is randomized for fairness.
    This event is repeated max_Z times to allow multiple firm visits.

    Algorithm
    ---------
    1. Randomize consumer shopping order
    2. For each consumer j with budget (B_j > 0):
       - Pop next firm from shopping queue: i = shop_targets[j, head_j]
       - Calculate purchase: Q = min(B_j / P_i, S_i)
       - Update spending: B_j ← B_j - (Q × P_i)
       - Update inventory: S_i ← S_i - Q
       - Track purchase for loyalty: total_spent_j += (Q × P_i)
       - If Q > 0: update prev_largest_producer if this is biggest purchase
       - Advance queue pointer: head_j += 1

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, n_households=500, seed=42)
    >>> # Prepare shopping
    >>> sim.get_event("consumers_decide_firms_to_visit")().execute(sim)
    >>> # Execute one round
    >>> initial_inventory = sim.prod.inventory.sum()
    >>> event = sim.get_event("consumers_shop_one_round")
    >>> event.execute(sim)
    >>> # Inventory decreased
    >>> sim.prod.inventory.sum() < initial_inventory
    True

    Process all shopping rounds:

    >>> max_Z = sim.config.max_Z
    >>> for _ in range(max_Z):
    ...     sim.get_event("consumers_shop_one_round")().execute(sim)

    Notes
    -----
    This event must execute after ConsumersDecideFirmsToVisit (need shopping queues).

    This event is typically repeated max_Z times to process all shopping rounds.

    Shopping order randomized each round to prevent systematic bias (e.g., low-ID
    consumers always shopping first).

    Consumers can partially exhaust inventory: if firm has less than requested
    quantity, consumer buys what's available and moves to next firm.

    See Also
    --------
    ConsumersDecideFirmsToVisit : Prepares shopping queues
    ConsumersFinalizePurchases : Handles unspent budget
    bamengine.events._internal.goods_market.consumers_shop_one_round : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        """Execute one shopping round."""
        from bamengine.events._internal.goods_market import consumers_shop_one_round

        consumers_shop_one_round(sim.con, sim.prod, rng=sim.rng)


@event
class ConsumersFinalizePurchases:
    """
    Return unspent budget to savings after shopping rounds complete.

    Any budget remaining after all shopping rounds is moved back to savings.
    This ensures wealth conservation: no money is lost during shopping.

    Algorithm
    ---------
    For each consumer j:

    .. math::
        SA_j \\leftarrow SA_j + B_j

        B_j \\leftarrow 0

    where SA_j = savings, B_j = income_to_spend (remaining budget).

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_households=500, seed=42)
    >>> # Shop first
    >>> for _ in range(sim.config.max_Z):
    ...     sim.get_event("consumers_shop_one_round")().execute(sim)
    >>> # Track unspent
    >>> unspent = sim.con.income_to_spend.copy()
    >>> initial_savings = sim.con.savings.copy()
    >>> # Finalize
    >>> event = sim.get_event("consumers_finalize_purchases")
    >>> event.execute(sim)

    Verify unspent returned to savings:

    >>> import numpy as np
    >>> np.allclose(sim.con.savings, initial_savings + unspent)
    True

    Budget cleared:

    >>> (sim.con.income_to_spend == 0).all()
    True

    Notes
    -----
    This event must execute after all ConsumersShopOneRound rounds complete.

    Wealth conservation: unspent budget → savings (no money vanishes).

    Consumers with zero unspent budget are unaffected (savings unchanged).

    See Also
    --------
    ConsumersShopOneRound : Spends budget during shopping
    ConsumersDecideIncomeToSpend : Initially allocates budget from wealth
    bamengine.events._internal.goods_market.consumers_finalize_purchases : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.goods_market import consumers_finalize_purchases

        consumers_finalize_purchases(sim.con)
