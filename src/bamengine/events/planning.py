"""
Planning events for firm decision-making.

Each event encapsulates a specific planning decision that firms make at the
start of each simulation period.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


@event
class FirmsDecideDesiredProduction:
    """
    Calculate expected demand and update desired production.

    Rule
    ----
        shock ~ U(0, h_ρ)
        if S == 0 and P ≥ P̄   → raise by (1 + shock)
        if S  > 0 and P < P̄   → cut   by (1 − shock)
        otherwise             → keep previous level

    S: Inventory, P: Individual Price, P̄: Avg Market Price, h_ρ: Max Production Shock
    """

    # TODO Address warning: Method 'execute' may be 'static'
    #  across all events in this package
    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.planning import firms_decide_desired_production

        firms_decide_desired_production(
            sim.prod,
            p_avg=sim.ec.avg_mkt_price,
            h_rho=sim.config.h_rho,
            rng=sim.rng,
        )


@event
class FirmsCalcBreakevenPrice:
    """
    Calculate breakeven price based on expected costs.

    Rule
    ----
        P_breakeven = (W + interest) / Y

    P_breakeven: Breakeven Price, W: Wage Bill, Y: Production

    Note
    ----
        Prices are capped at current price × cap_factor to prevent extreme jumps.
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.planning import firms_calc_breakeven_price

        firms_calc_breakeven_price(
            prod=sim.prod,
            emp=sim.emp,
            lb=sim.lb,
            cap_factor=sim.config.cap_factor,
        )


@event
class FirmsAdjustPrice:
    """
    Firms adjust nominal prices based on inventory and market conditions.

    Rule
    ----
        shock ~ U(0, h_η)
        If S == 0 and P < P̄   → multiply by (1 + shock)
        If S > 0 and P >= P̄   → multiply by (1 − shock)

    S: Inventory, P: Individual Price, P̄: Avg Market Price, h_η: Max Price Shock

    Note
    ----
        Prices are floored at breakeven level
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.planning import firms_adjust_price

        firms_adjust_price(
            sim.prod,
            p_avg=sim.ec.avg_mkt_price,
            h_eta=sim.config.h_eta,
            rng=sim.rng,
        )


@event
class FirmsDecideDesiredLabor:
    """
    Firms decide desired labor based on production targets and productivity.

    Rule
    ----
        Ld = ceil(Yd / a)

    Ld: Desired Labour, Yd: Desired Production, a: Labour Productivity
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.planning import firms_decide_desired_labor

        firms_decide_desired_labor(sim.prod, sim.emp)


@event
class FirmsDecideVacancies:
    """
    Firms decide how many job vacancies to post.

    Rule
    ----
        V = max( Ld – L , 0 )

    V: Number of Open Vacancies, Ld: Desired Labour, L: Actual Labour
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.planning import firms_decide_vacancies

        firms_decide_vacancies(sim.emp)
