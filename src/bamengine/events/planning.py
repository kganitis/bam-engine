"""
Planning events for firm production and labor decisions.

This module defines the planning phase events that execute at the start of
each simulation period. Firms make forward-looking decisions about production
targets and labor requirements based on current market conditions.

Event Sequence
--------------
The default planning events execute in this order:

1. FirmsDecideDesiredProduction - Set production targets based on inventory/prices
2. FirmsDecideDesiredLabor - Calculate labor needs from production targets
3. FirmsDecideVacancies - Determine job openings
4. FirmsFireExcessWorkers - Lay off workers when labor exceeds desired

Default planning-phase pricing events (after FirmsDecideDesiredProduction):

- FirmsPlanBreakevenPrice - Breakeven using previous-period costs
- FirmsPlanPrice - Price adjustment with breakeven floor

These are mutually exclusive with the production-phase pricing events
(FirmsCalcBreakevenPrice, FirmsAdjustPrice) defined in
bamengine.events.production. Activate those via ``pricing_phase='production'``.

Design Notes
------------
- Events operate on firm roles (Producer, Employer, Borrower)
- Each event wraps a system function from events._internal.planning
- System functions contain the actual implementation logic
- Events handle simulation state access and parameter passing

Examples
--------
Execute planning events:

>>> import bamengine as be
>>> sim = be.Simulation.init(n_firms=100, seed=42)
>>> # Planning events run as part of default pipeline
>>> sim.step()

Execute individual planning event:

>>> event = sim.get_event("firms_decide_desired_production")
>>> event.execute(sim)
>>> sim.prod.desired_production.mean()  # doctest: +SKIP
105.0

See Also
--------
bamengine.events._internal.planning : System function implementations
bamengine.events.production : Production-phase pricing events
Producer : Production and pricing state
Employer : Labor hiring state
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:  # pragma: no cover
    from bamengine.simulation import Simulation


@event
class FirmsDecideDesiredProduction:
    """
    Set production targets based on inventory levels and market position.

    Firms adjust their production targets adaptively based on whether they
    have unsold inventory and how their prices compare to market average.
    This implements the adaptive production rule from Delli Gatti et al. (2011).

    This event first zeroes out the current ``production`` field (preparing it for
    the new period), then uses ``production_prev`` (the previous period's actual
    production) as the baseline signal for planning decisions.

    Algorithm
    ---------
    For each firm i:

    1. Zero out current production: :math:`Y_i := 0`
    2. Generate random shock: :math:`\\varepsilon_i \\sim U(0, h_\\rho)`
    3. Use previous production (``production_prev``) as baseline and apply rule:

       - If :math:`S_i = 0` and :math:`P_i \\geq \\bar{P}`: :math:`Y^*_i = Y^{prev}_i \\times (1 + \\varepsilon_i)` [increase]
       - If :math:`S_i > 0` and :math:`P_i < \\bar{P}`: :math:`Y^*_i = Y^{prev}_i \\times (1 - \\varepsilon_i)` [decrease]
       - Otherwise: :math:`Y^*_i = Y^{prev}_i` [maintain]

    4. Set desired_production = expected_demand = :math:`Y^*_i`

    Mathematical Notation
    ---------------------
    .. math::
        Y^*_{i,t} = \\begin{cases}
            Y^{prev}_{i}(1 + \\varepsilon_i) & \\text{if } S_{i,t-1}=0 \\land P_i \\geq \\bar{P} \\\\
            Y^{prev}_{i}(1 - \\varepsilon_i) & \\text{if } S_{i,t-1}>0 \\land P_i < \\bar{P} \\\\
            Y^{prev}_{i} & \\text{otherwise}
        \\end{cases}

    where:

    - :math:`Y^*`: desired production for next period
    - :math:`Y^{prev}`: actual production in previous period (``production_prev``)
    - :math:`S`: inventory (unsold goods from previous period)
    - :math:`P`: firm's individual price
    - :math:`\\bar{P}`: average market price across all firms
    - :math:`\\varepsilon`: random shock :math:`\\sim U(0, h_\\rho)`
    - :math:`h_\\rho`: maximum production shock parameter (config)

    Examples
    --------
    Execute this event:

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> event = sim.get_event("firms_decide_desired_production")
    >>> event.execute(sim)

    Inspect production changes:

    >>> prod = sim.prod
    >>> firms_increasing = (prod.inventory == 0) & (prod.price >= sim.ec.avg_mkt_price)
    >>> firms_increasing.sum()  # doctest: +SKIP
    45

    Check desired production distribution:

    >>> prod.desired_production.mean()  # doctest: +SKIP
    52.5
    >>> prod.desired_production.std()  # doctest: +SKIP
    5.2

    Notes
    -----
    This event must execute early in the pipeline as desired production
    determines labor requirements (see FirmsDecideDesiredLabor).

    The production shock introduces randomness that prevents firms from
    settling into static equilibria.

    See Also
    --------
    FirmsDecideDesiredLabor : Calculate labor needs from production targets
    Producer : Production and pricing state
    bamengine.events._internal.planning.firms_decide_desired_production : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.planning import firms_decide_desired_production

        firms_decide_desired_production(
            sim.prod,
            p_avg=sim.ec.avg_mkt_price,
            h_rho=sim.config.h_rho,
            rng=sim.rng,
        )


@event
class FirmsPlanBreakevenPrice:
    """
    Calculate planning-phase breakeven price from previous period's costs.

    Computes minimum cost-covering price using last period's wage bill and
    interest obligations as the numerator, and desired production (the
    firm's current-period output target) as the denominator.
    This provides an early price signal before the labor and credit
    markets operate.

    Algorithm
    ---------
    For each firm i:

    1. Calculate total costs: :math:`C_i = W_{i,t-1} + I_{i,t-1}`
    2. Calculate breakeven: :math:`P_{\\text{breakeven}} = C_i / Y^d_{i,t}`
    3. Apply cap (if configured): :math:`P_{\\text{breakeven}} = \\min(P_{\\text{breakeven}}, P_i \\times \\text{cap\\_factor})`

    where :math:`W_{t-1}` is previous period's wage bill, :math:`I_{t-1}` is previous
    period's interest, and :math:`Y^d_{t}` is the desired production for the
    current period (freshly computed by FirmsDecideDesiredProduction).

    Notes
    -----
    At planning time, wage_bill and LoanBook interest naturally contain
    previous period's values (recomputed only in Phases 2-3). This event
    exploits that timing to avoid introducing separate "previous expenses"
    fields.

    Must run AFTER firms_decide_desired_production (needs desired_production).

    These planning-phase pricing events are **mutually exclusive** with the
    production-phase pair (FirmsCalcBreakevenPrice / FirmsAdjustPrice).
    Use one pair or the other, not both. When using planning-phase pricing,
    remove the production-phase events from the pipeline.

    Not included in the default pipeline. Users can insert it into the
    planning phase via a custom pipeline YAML or programmatically:

        sim.pipeline.insert_after("firms_decide_desired_production",
                                  "firms_plan_breakeven_price")

    See Also
    --------
    FirmsCalcBreakevenPrice : Production-phase alternative (uses projected production)
    FirmsPlanPrice : Planning-phase price adjustment using this breakeven
    bamengine.events._internal.planning.firms_plan_breakeven_price : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.planning import firms_plan_breakeven_price

        firms_plan_breakeven_price(
            prod=sim.prod,
            emp=sim.emp,
            lb=sim.lb,
            cap_factor=sim.cap_factor,
        )


@event
class FirmsPlanPrice:
    """
    Planning-phase price adjustment based on inventory and market position.

    Adjusts prices using the same inventory/market-position rule as
    FirmsAdjustPrice, but runs during the planning phase using the
    breakeven computed from previous period's costs.

    These planning-phase pricing events are mutually exclusive with the
    production-phase pair (FirmsCalcBreakevenPrice / FirmsAdjustPrice).
    Use one pair or the other, not both.

    Not included in the default pipeline. Insert alongside
    FirmsPlanBreakevenPrice as a replacement for the production-phase
    pricing events.

    See Also
    --------
    FirmsAdjustPrice : Production-phase alternative
    FirmsPlanBreakevenPrice : Planning-phase breakeven calculation
    bamengine.events._internal.planning.firms_plan_price : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.planning import firms_plan_price

        firms_plan_price(
            sim.prod,
            p_avg=sim.ec.avg_mkt_price,
            h_eta=sim.config.h_eta,
            price_cut_allow_increase=sim.config.price_cut_allow_increase,
            rng=sim.rng,
        )


@event
class FirmsDecideDesiredLabor:
    """
    Calculate labor requirements from production targets and productivity.

    Firms determine how many workers they need to achieve their desired
    production level, given their labor productivity (output per worker).

    Algorithm
    ---------
    For each firm i:

    .. math::
        L^d_i = \\lceil Y^d_i / \\phi_i \\rceil

    where:

    - :math:`L^d`: desired labor (number of workers needed)
    - :math:`Y^d`: desired production (from FirmsDecideDesiredProduction)
    - :math:`\\phi`: labor productivity (output per worker)

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> event = sim.get_event("firms_decide_desired_labor")
    >>> event.execute(sim)
    >>> sim.emp.desired_labor.sum()  # doctest: +SKIP
    500

    See Also
    --------
    FirmsDecideDesiredProduction : Set production targets
    FirmsDecideVacancies : Calculate job openings
    bamengine.events._internal.planning.firms_decide_desired_labor : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.planning import firms_decide_desired_labor

        firms_decide_desired_labor(sim.prod, sim.emp)


@event
class FirmsDecideVacancies:
    """
    Calculate number of job vacancies to post.

    Firms compare their desired labor force to their current labor force
    and post vacancies for the difference (if positive).

    Algorithm
    ---------
    For each firm i:

    .. math::
        V_i = \\max(L^d_i - L_i, 0)

    where:

    - :math:`V`: number of vacancies to post
    - :math:`L^d`: desired labor (from FirmsDecideDesiredLabor)
    - :math:`L`: current labor force

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> event = sim.get_event("firms_decide_vacancies")
    >>> event.execute(sim)
    >>> (sim.emp.n_vacancies >= 0).all()
    True
    >>> (
    ...     sim.emp.n_vacancies == sim.emp.desired_labor - sim.emp.current_labor
    ... ).all()  # doctest: +SKIP
    True

    See Also
    --------
    FirmsDecideDesiredLabor : Calculate labor requirements
    bamengine.events._internal.planning.firms_decide_vacancies : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.planning import firms_decide_vacancies

        firms_decide_vacancies(sim.emp)


@event
class FirmsFireExcessWorkers:
    """
    Fire workers when current labor exceeds desired labor.

    Firms with more workers than needed (due to reduced production targets)
    lay off the excess workers. By default, firms fire workers randomly,
    but can be configured to fire the most expensive workers first.

    Algorithm
    ---------
    For each firm i with :math:`L_i > L^d_i` (current labor exceeds desired):

    1. Calculate excess: :math:`E_i = L_i - L^d_i`
    2. Select :math:`E_i` workers to fire (by method: random or expensive first)
    3. Fire selected workers:
       - Set worker's employer = -1 (unemployed)
       - Set worker's wage = 0
       - Set worker's fired flag = True
       - Decrement firm's current_labor

    Mathematical Notation
    ---------------------
    For firm i with :math:`L_i > L^d_i`:

    .. math::
        E_i = L_i - L^d_i

    Fire :math:`E_i` workers so that:

    .. math::
        L_i \\leftarrow L^d_i

    Examples
    --------
    Execute this event:

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, n_households=500, seed=42)
    >>> event = sim.get_event("firms_fire_excess_workers")
    >>> event.execute(sim)

    Check that firms now have current_labor <= desired_labor:

    >>> (sim.emp.current_labor <= sim.emp.desired_labor).all()
    True

    Notes
    -----
    This event executes in the Planning phase after vacancies are calculated.
    Workers fired here have their `fired` flag set to True, which affects
    their job search behavior (loyalty rule does not apply).

    See Also
    --------
    FirmsDecideVacancies : Calculate job openings
    FirmsFireWorkers : Fire workers due to financing gaps (credit market)
    bamengine.events._internal.planning.firms_fire_excess_workers : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.planning import firms_fire_excess_workers

        firms_fire_excess_workers(
            sim.emp,
            sim.wrk,
            rng=sim.rng,
        )
