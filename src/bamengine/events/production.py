"""
Production events for wage payments, production, and contract updates.

Each event encapsulates production execution, wage payments, and employment
contract management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


@event
class FirmsPayWages:
    """
    Debit firm cash accounts by wage bills and update employer state.
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.production import firms_pay_wages

        firms_pay_wages(sim.emp)


@event
class WorkersReceiveWage:
    """
    Credit household income with wages for employed workers
    and update consumer state.

    Rule
    ----
        income += wage · employed
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.production import workers_receive_wage

        workers_receive_wage(sim.con, sim.wrk)


@event
class FirmsRunProduction:
    """
    Compute production output and update inventory state.

    Rule
    ----
        Y  =  a · L
        S  ←  Y

    Y: Actual Production Output, a: Labor Productivity, L: Actual Labor, S: Inventory
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.production import firms_run_production

        firms_run_production(sim.prod, sim.emp)


@event
class WorkersUpdateContracts:
    """
    Decrease `periods_left` for every employed worker and let contracts that
    reach 0 expire. All worker-side flags are updated and the corresponding
    firm's labor and wage-bill are brought back in sync.

    Rule
    ----
        L_i = Σ {worker employed & employer == i}
        W   = L · w

    L: Actual Labor, W: Wage Bill, w: Individual Wage
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.production import workers_update_contracts

        workers_update_contracts(sim.wrk, sim.emp)
