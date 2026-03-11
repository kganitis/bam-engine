"""
Goods market events for consumption decisions and shopping.

This module defines the goods market phase events that execute after production.
Households calculate consumption propensity, allocate income to spending,
select firms to visit, and purchase goods through batch-sequential shopping.

Event Sequence
--------------
The goods market events execute in this order:

1. ConsumersCalcPropensity - Calculate propensity to consume based on savings
2. ConsumersDecideIncomeToSpend - Allocate income to spending budget
3. ConsumersDecideFirmsToVisit - Select firms to visit (sorted by price)
4. GoodsMarketRound - Batch-sequential shopping (handles all Z rounds internally)
5. ConsumersFinalizePurchases - Move unspent budget back to savings

Design Notes
------------
- Events operate on consumer and producer roles (Consumer, Producer)
- Propensity to consume: c = 1 / (1 + tanh(SA/SA_avg)^β)
- Loyalty rule: consumers visit previous largest producer first (if inventory available)
- Remaining Z-1 firms selected randomly (preferential attachment mechanism)
- Batch-sequential processing: consumers are shuffled and divided into batches,
  each completing all Z visits before the next batch starts, preserving sequential
  depletion dynamics while using vectorized NumPy operations within each batch

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
    GoodsMarketRound : Uses income_to_spend as shopping budget
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
    the reference implementation. The key insight is that loyalty is
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
    GoodsMarketRound : Processes shopping queue
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
            consumer_matching=sim.config.consumer_matching,
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
    >>> sim.get_event("goods_market_round")().execute(sim)
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
    This event must execute after GoodsMarketRound completes.

    Wealth conservation: unspent budget → savings (no money vanishes).

    Consumers with zero unspent budget are unaffected (savings unchanged).

    See Also
    --------
    GoodsMarketRound : Spends budget during shopping
    ConsumersDecideIncomeToSpend : Initially allocates budget from wealth
    bamengine.events._internal.goods_market.consumers_finalize_purchases : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.goods_market import consumers_finalize_purchases

        consumers_finalize_purchases(sim.con)


@event
class GoodsMarketRound:
    """Batch-sequential goods market matching.

    Consumers are shuffled and divided into batches, each completing all Z
    visits before the next batch starts.  This preserves the sequential
    depletion dynamics while using vectorized NumPy operations within each
    batch.

    This event is called once in the pipeline (it handles all Z rounds
    internally).

    See Also
    --------
    bamengine.events._internal.goods_market.goods_market_round :
        Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.goods_market import goods_market_round

        goods_market_round(
            sim.con,
            sim.prod,
            max_Z=sim.config.max_Z,
            n_batches=sim.config.n_batches,
            rng=sim.rng,
        )
