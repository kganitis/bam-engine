"""Production events for wage payments, production, and contract updates.

This module contains Event classes that wrap production system functions.
Each event encapsulates production execution, wage payments, and employment
contract management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bamengine.core import Event

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


@dataclass(slots=True)
class FirmsPayWages(Event):
    """
    Firms pay wages to employed workers, reducing net worth.

    Net worth is reduced by the total wage bill:
        NW_i(t) = NW_i(t-1) - wage_bill_i

    This event wraps `bamengine.systems.production.firms_pay_wages`.

    Dependencies
    ------------
    - firms_fire_workers : Uses final employment state after layoffs

    See Also
    --------
    bamengine.systems.production.firms_pay_wages : Underlying logic
    """

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Events that must run before this one."""
        return ("firms_fire_workers",)

    def execute(self, sim: Simulation) -> None:
        """Execute wage payment."""
        from bamengine.systems.production import firms_pay_wages

        firms_pay_wages(sim.emp)


@dataclass(slots=True)
class WorkersReceiveWage(Event):
    """
    Workers receive wages as income, adding to savings.

    Savings are increased by wage income:
        S_j(t) = S_j(t-1) + w_j

    This event wraps `bamengine.systems.production.workers_receive_wage`.

    Dependencies
    ------------
    - firms_pay_wages : Wages must be paid first

    See Also
    --------
    bamengine.systems.production.workers_receive_wage : Underlying logic
    """

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Events that must run before this one."""
        return ("firms_pay_wages",)

    def execute(self, sim: Simulation) -> None:
        """Execute wage receipt."""
        from bamengine.systems.production import workers_receive_wage

        workers_receive_wage(sim.con, sim.wrk)


@dataclass(slots=True)
class FirmsRunProduction(Event):
    """
    Firms execute production based on labor and productivity.

    Production is calculated as:
        Q_i = labor_productivity_i · L_i

    Output is added to inventory.

    This event wraps `bamengine.systems.production.firms_run_production`.

    Dependencies
    ------------
    - firms_pay_wages : Production happens after wage payment

    See Also
    --------
    bamengine.systems.production.firms_run_production : Underlying logic
    """

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Events that must run before this one."""
        return ("firms_pay_wages",)

    def execute(self, sim: Simulation) -> None:
        """Execute production."""
        from bamengine.systems.production import firms_run_production

        firms_run_production(sim.prod, sim.emp)


@dataclass(slots=True)
class WorkersUpdateContracts(Event):
    """
    Decrement worker employment contract durations.

    Contract countdown:
        contract_i(t) = contract_i(t-1) - 1

    Workers with expired contracts (contract == 0) become unemployed.

    This event wraps `bamengine.systems.production.workers_update_contracts`.

    Dependencies
    ------------
    - firms_run_production : Production period complete

    See Also
    --------
    bamengine.systems.production.workers_update_contracts : Underlying logic
    """

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Events that must run before this one."""
        return ("firms_run_production",)

    def execute(self, sim: Simulation) -> None:
        """Execute contract update."""
        from bamengine.systems.production import workers_update_contracts

        workers_update_contracts(sim.wrk, sim.emp)
