"""Planning events for firm decision-making.

This module contains Event classes that wrap planning system functions.
Each event encapsulates a specific planning decision that firms make at the
start of each simulation period.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bamengine.core import Event

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


@dataclass(slots=True)
class FirmsDecideDesiredProduction(Event):
    """
    Firms decide desired production based on inventory and market price.

    Implements adaptive expectations rule where firms:
    - Increase production if inventory is zero and price above average
    - Decrease production if inventory is positive and price below average
    - Otherwise keep previous production level

    This event wraps `bamengine.systems.planning.firms_decide_desired_production`.

    Dependencies
    ------------
    None (first event in planning phase)

    See Also
    --------
    bamengine.systems.planning.firms_decide_desired_production : Underlying logic
    """

    def execute(self, sim: Simulation) -> None:
        """Execute desired production decision."""
        from bamengine.systems.planning import firms_decide_desired_production

        firms_decide_desired_production(
            sim.prod,
            p_avg=sim.ec.avg_mkt_price,
            h_rho=sim.config.h_rho,
            rng=sim.rng,
        )


@dataclass(slots=True)
class FirmsCalcBreakevenPrice(Event):
    """
    Firms calculate breakeven price based on expected costs.

    Breakeven price is calculated as:
        P_breakeven = (wage_bill + interest) / production

    This represents the minimum price needed to cover costs.

    This event wraps `bamengine.systems.planning.firms_calc_breakeven_price`.

    Dependencies
    ------------
    - firms_decide_desired_production : Sets expected_demand used in calculation

    See Also
    --------
    bamengine.systems.planning.firms_calc_breakeven_price : Underlying logic
    """

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Events that must run before this one."""
        return ("firms_decide_desired_production",)

    def execute(self, sim: Simulation) -> None:
        """Execute breakeven price calculation."""
        from bamengine.systems.planning import firms_calc_breakeven_price

        firms_calc_breakeven_price(
            prod=sim.prod,
            emp=sim.emp,
            lb=sim.lb,
            cap_factor=sim.config.cap_factor,
        )


@dataclass(slots=True)
class FirmsAdjustPrice(Event):
    """
    Firms adjust nominal prices based on inventory and market conditions.

    Price adjustment rules:
    - If inventory == 0 and price < avg: raise price (multiply by 1 + shock)
    - If inventory > 0 and price >= avg: cut price (multiply by 1 - shock)
    - Prices are floored at breakeven level

    This event wraps `bamengine.systems.planning.firms_adjust_price`.

    Dependencies
    ------------
    - firms_calc_breakeven_price : Sets breakeven_price floor

    See Also
    --------
    bamengine.systems.planning.firms_adjust_price : Underlying logic
    """

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Events that must run before this one."""
        return ("firms_calc_breakeven_price",)

    def execute(self, sim: Simulation) -> None:
        """Execute price adjustment decision."""
        from bamengine.systems.planning import firms_adjust_price

        firms_adjust_price(
            sim.prod,
            p_avg=sim.ec.avg_mkt_price,
            h_eta=sim.config.h_eta,
            rng=sim.rng,
        )


@dataclass(slots=True)
class FirmsDecideDesiredLabor(Event):
    """
    Firms decide desired labor based on production targets and productivity.

    Desired labor is calculated as:
        L_desired = ceil(desired_production / labor_productivity)

    This event wraps `bamengine.systems.planning.firms_decide_desired_labor`.

    Dependencies
    ------------
    - firms_decide_desired_production : Sets desired_production

    See Also
    --------
    bamengine.systems.planning.firms_decide_desired_labor : Underlying logic
    """

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Events that must run before this one."""
        return ("firms_decide_desired_production",)

    def execute(self, sim: Simulation) -> None:
        """Execute desired labor calculation."""
        from bamengine.systems.planning import firms_decide_desired_labor

        firms_decide_desired_labor(sim.prod, sim.emp)


@dataclass(slots=True)
class FirmsDecideVacancies(Event):
    """
    Firms decide how many job vacancies to post.

    Vacancies are calculated as:
        V = max(L_desired - L_current, 0)

    This event wraps `bamengine.systems.planning.firms_decide_vacancies`.

    Dependencies
    ------------
    - firms_decide_desired_labor : Sets desired_labor

    See Also
    --------
    bamengine.systems.planning.firms_decide_vacancies : Underlying logic
    """

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Events that must run before this one."""
        return ("firms_decide_desired_labor",)

    def execute(self, sim: Simulation) -> None:
        """Execute vacancy decision."""
        from bamengine.systems.planning import firms_decide_vacancies

        firms_decide_vacancies(sim.emp)
