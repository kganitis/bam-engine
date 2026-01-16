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
- Remaining Z-1 firms selected randomly (preferential attachment mechanism)
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

    1. Calculate relative savings: :math:`r_j = SA_j / \\overline{SA}`
    2. Apply propensity function: :math:`c_j = 1 / (1 + \\tanh(r_j)^\\beta)`

    Mathematical Notation
    ---------------------
    .. math::
        c_j = \\frac{1}{1 + \\tanh\\left(\\frac{SA_j}{\\overline{SA}}\\right)^\\beta}

    where:

    - :math:`c_j`: propensity to consume (:math:`0 < c_j < 1`)
    - :math:`SA_j`: current savings of consumer j
    - :math:`\\overline{SA}`: average savings across all consumers
    - :math:`\\beta`: sensitivity parameter controlling consumption response (config)

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

    1. Calculate wealth: :math:`W_j = SA_j + I_j`
    2. Allocate to spending: :math:`B_j = W_j \\times c_j`
    3. Update savings: :math:`SA_j = W_j - B_j`
    4. Reset income: :math:`I_j = 0`

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
    Consumers select firms to visit and set loyalty BEFORE shopping.

    Consumers with spending budget build a shopping queue by:
    1. Placing their loyalty firm (previous largest producer) in slot 0
    2. Randomly sampling Z-1 additional firms from those with inventory
    3. Sorting the randomly sampled firms by price (cheapest first)
    4. Setting loyalty to the LARGEST producer in the consideration set

    This implements the book's preferential attachment (PA) mechanism matching
    the NetLogo reference implementation. The key insight is that loyalty is
    updated BEFORE shopping based on the consideration set, not during shopping
    based on purchases. This allows the "rich get richer" dynamics to emerge.

    Algorithm
    ---------
    For each consumer j with B_j > 0 (spending budget):

    1. Apply loyalty rule first:
       - If prev_largest_producer has inventory: place in slot 0
    2. Sample remaining slots randomly from firms with inventory
    3. Sort sampled firms by price (cheapest first) for shopping order
    4. Update loyalty to largest producer in consideration set (BEFORE shopping)

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

    Notes
    -----
    This event must execute after ConsumersDecideIncomeToSpend (need spending budget).

    Only consumers with positive spending budget prepare shopping queues.

    The preferential attachment mechanism works as follows:
    - Consumers track the "largest producer" in their consideration set
    - Loyalty is updated BEFORE shopping to the largest in consideration set
    - This firm is visited first (if it has inventory), minimizing rationing risk
    - Even if consumer buys from cheap small firms, they track the large firm
    - Over time, large firms accumulate more loyal customers, creating "rich get richer" dynamics
    - This leads to emergent firm size heterogeneity and business cycle fluctuations

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

    Note: Loyalty (largest_prod_prev) is NOT updated during shopping. It was
    already set BEFORE shopping in ConsumersDecideFirmsToVisit based on the
    consideration set. This matches the NetLogo reference implementation.

    Algorithm
    ---------
    1. Randomize consumer shopping order
    2. For each consumer j with budget (:math:`B_j > 0`):
       - Pop next firm from shopping queue: :math:`i = \\text{shop\\_targets}[j, \\text{head}_j]`
       - Calculate purchase: :math:`Q = \\min(B_j / P_i, S_i)`
       - Update spending: :math:`B_j \\leftarrow B_j - (Q \\times P_i)`
       - Update inventory: :math:`S_i \\leftarrow S_i - Q`
       - Advance queue pointer: :math:`\\text{head}_j \\mathrel{+}= 1`

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
class ConsumersShopSequential:
    """
    Execute sequential shopping where each consumer completes all visits.

    Unlike round-robin shopping (ConsumersShopOneRound called max_Z times),
    this event processes consumers one at a time. Each consumer completes all
    their Z shopping visits before the next consumer starts.

    This matches NetLogo and ABCredit behavior and makes the goods market
    less efficient: early consumers can deplete inventory from multiple firms,
    leaving late consumers with wasted visits on sold-out firms. This results
    in more unsold inventory overall, which affects production decisions.

    Algorithm
    ---------
    1. Randomize consumer order (like NetLogo's `ask workers`)
    2. For each consumer j with budget:
       - Visit up to max_Z firms sequentially
       - At each firm: purchase if inventory available, else wasted visit
       - Continue until budget exhausted or all Z visits complete
    3. Move to next consumer

    Key Differences from Round-Robin
    --------------------------------
    - Round-robin: All 500 consumers visit their 1st firm, then all visit 2nd, etc.
      This spreads purchases evenly across firms.
    - Sequential: Consumer 1 visits all Z firms (may exhaust inventory at several),
      then Consumer 2 visits (some firms already depleted), etc.
      This creates more inventory shortages AND more unsold goods at unpopular firms.

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, n_households=500, seed=42)
    >>> # Prepare shopping
    >>> sim.get_event("consumers_decide_firms_to_visit")().execute(sim)
    >>> # Execute sequential shopping (replaces max_Z rounds of shop_one_round)
    >>> event = sim.get_event("consumers_shop_sequential")
    >>> event.execute(sim)

    Notes
    -----
    This is a single event that replaces `consumers_shop_one_round x max_Z`.

    Early consumers can deplete inventory from multiple firms before later
    consumers get a chance to shop, creating wasted visits.

    See Also
    --------
    ConsumersShopOneRound : Round-robin alternative (one visit per round)
    ConsumersDecideFirmsToVisit : Prepares shopping queues
    bamengine.events._internal.goods_market.consumers_shop_sequential : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        """Execute sequential shopping where each consumer completes all visits."""
        from bamengine.events._internal.goods_market import consumers_shop_sequential

        consumers_shop_sequential(
            sim.con, sim.prod, max_Z=sim.config.max_Z, rng=sim.rng
        )


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

    where :math:`SA_j` = savings, :math:`B_j` = income_to_spend (remaining budget).

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
