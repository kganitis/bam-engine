"""
Production events for wage payments, production execution, and contract management.

This module defines the production phase events that execute after credit market
events. Firms pay wages, workers receive income, firms produce goods using labor,
and employment contracts are updated (decrementing periods remaining).

Event Sequence
--------------
The production events execute in this order:

1. FirmsPayWages - Firms deduct wage bill from available funds
2. WorkersReceiveWage - Workers add wages to income
3. FirmsRunProduction - Firms produce goods using labor
4. WorkersUpdateContracts - Decrement contract duration, handle expiration

Design Notes
------------
- Events operate on producer, employer, worker, and consumer roles
- Production function: Y = φ × L (output = productivity × labor)
- Contract expiration: periods_left decremented each period, expires at 0
- Firms with zero labor produce zero output (marked for bankruptcy)
- Wage payments reduce firm funds but increase worker income (money transfer)

Examples
--------
Execute production events:

>>> import bamengine as be
>>> sim = be.Simulation.init(n_firms=100, n_households=500, seed=42)
>>> # Production events run as part of default pipeline
>>> sim.step()

Execute individual production event:

>>> event = sim.get_event("firms_run_production")
>>> event.execute(sim)
>>> sim.prod.production.mean()  # doctest: +SKIP
52.5

Check production output:

>>> sim.prod.inventory.sum()  # doctest: +SKIP
5250.0

See Also
--------
bamengine.events._internal.production : System function implementations
Producer : Production state
Employer : Labor force state
Worker : Employment state
Consumer : Income state
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:  # pragma: no cover
    from bamengine.simulation import Simulation


@event
class FirmsPayWages:
    """
    Firms pay wages by deducting wage bill from available funds.

    Firms transfer funds to cover their wage obligations. This reduces firm cash
    (total_funds) by the wage bill amount. Workers receive these wages in the
    next event (WorkersReceiveWage).

    Algorithm
    ---------
    For each firm i:

    .. math::
        A_i \\leftarrow A_i - W_i

    where :math:`A_i` = total_funds, :math:`W_i` = wage_bill.

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> initial_funds = sim.emp.total_funds.copy()
    >>> event = sim.get_event("firms_pay_wages")
    >>> event.execute(sim)
    >>> # Funds reduced by wage bill
    >>> import numpy as np
    >>> reduction = initial_funds - sim.emp.total_funds
    >>> np.allclose(reduction, sim.emp.wage_bill)
    True

    Notes
    -----
    This event must execute after FirmsFireWorkers (wage_bill finalized).

    See Also
    --------
    WorkersReceiveWage : Workers receive wages (counterpart event)
    bamengine.events._internal.production.firms_pay_wages : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.production import firms_pay_wages

        firms_pay_wages(sim.emp)


@event
class WorkersReceiveWage:
    """
    Workers receive wage income from employment.

    Employed workers add their wages to their income (Consumer role). This is
    the counterpart to FirmsPayWages: money flows from firms to households.

    Algorithm
    ---------
    For each employed worker j (:math:`\\text{employer}_j \\geq 0`):

    .. math::
        I_j \\leftarrow I_j + w_j

    where :math:`I_j` = income, :math:`w_j` = wage.

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_households=500, seed=42)
    >>> initial_income = sim.con.income.copy()
    >>> event = sim.get_event("workers_receive_wage")
    >>> event.execute(sim)
    >>> # Employed workers gained income
    >>> import numpy as np
    >>> employed_mask = sim.wrk.employed
    >>> income_gain = sim.con.income - initial_income
    >>> np.allclose(income_gain[employed_mask], sim.wrk.wage[employed_mask])
    True

    Notes
    -----
    This event must execute after FirmsPayWages (firms have paid).

    Only employed workers receive wages. Unemployed workers gain no income.

    See Also
    --------
    FirmsPayWages : Firms pay wages (counterpart event)
    bamengine.events._internal.production.workers_receive_wage : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.production import workers_receive_wage

        workers_receive_wage(sim.con, sim.wrk)


@event
class FirmsRunProduction:
    """
    Firms produce goods using labor and add output to inventory.

    Production follows a simple linear technology: output equals labor
    productivity times number of workers. The produced goods are added to
    inventory for sale in the goods market.

    The calculated production is saved to both ``production`` (current period's output)
    and ``production_prev`` (for use as next period's planning signal).

    Algorithm
    ---------
    For each firm i:

    1. Calculate output: :math:`Y_i = \\phi_i \\times L_i`
    2. Store production: :math:`\\text{production}_i = Y_i`
    3. Store for next period's planning: :math:`\\text{production\\_prev}_i = Y_i`
    4. Add to inventory: :math:`S_i \\leftarrow Y_i`

    Mathematical Notation
    ---------------------
    .. math::
        Y_i = \\phi_i \\times L_i

        S_i \\leftarrow Y_i

    where:

    - :math:`Y_i`: production output for firm i
    - :math:`\\phi_i`: labor productivity (output per worker)
    - :math:`L_i`: current labor force (number of workers)
    - :math:`S_i`: inventory (replaces previous inventory with current production)

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> event = sim.get_event("firms_run_production")
    >>> event.execute(sim)

    Check production output:

    >>> sim.prod.production.mean()  # doctest: +SKIP
    52.5

    Verify production formula:

    >>> import numpy as np
    >>> expected_output = sim.prod.labor_prod * sim.emp.current_labor
    >>> np.allclose(sim.prod.production, expected_output)
    True

    Check inventory accumulation:

    >>> total_inventory = sim.prod.inventory.sum()
    >>> total_production = sim.prod.production.sum()
    >>> total_inventory >= total_production  # Inventory includes previous unsold goods
    True

    Notes
    -----
    This event must execute after WorkersReceiveWage.

    Firms with zero labor (L_i = 0) produce zero output and are marked for
    bankruptcy in later events.

    Inventory accumulates: S_new = S_old + Y. Unsold goods from previous
    periods remain in inventory.

    See Also
    --------
    FirmsDecideDesiredProduction : Plans production targets
    Producer : Production state with labor_prod, production, inventory
    bamengine.events._internal.production.firms_run_production : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.production import firms_run_production

        firms_run_production(sim.prod, sim.emp)


@event
class WorkersUpdateContracts:
    """
    Decrement contract duration and handle expiration for employed workers.

    Each employed worker's contract duration (periods_left) decrements by 1.
    Contracts that reach 0 expire: workers become unemployed but remain loyal
    (can reapply to previous employer). Firm labor counts are recalculated
    to reflect expired contracts.

    Algorithm
    ---------
    For each employed worker j:

    1. Decrement contract: :math:`\\text{periods\\_left}_j \\leftarrow \\text{periods\\_left}_j - 1`
    2. If :math:`\\text{periods\\_left}_j = 0`:
       - Set :math:`\\text{contract\\_expired}_j = \\text{True}`
       - Set :math:`\\text{employer\\_prev}_j = \\text{employer}_j` (store for loyalty)
       - Set :math:`\\text{employer}_j = -1` (unemployed)
       - Set :math:`w_j = 0`
    3. Recalculate firm labor counts:
       - For each firm i: :math:`L_i = |\\{j : \\text{employer}_j = i\\}|`
    4. Recalculate firm wage bills:
       - For each firm i: :math:`W_i = \\sum_{j : \\text{employer}_j = i} w_j`

    Mathematical Notation
    ---------------------
    For firm i after contract updates:

    .. math::
        L_i = |\\{j : \\text{employer}_j = i\\}|

        W_i = \\sum_{j : \\text{employer}_j = i} w_j

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_households=500, seed=42)
    >>> event = sim.get_event("workers_update_contracts")
    >>> event.execute(sim)

    Check contracts decremented:

    >>> import numpy as np
    >>> # All employed workers should have periods_left >= 0
    >>> employed_mask = sim.wrk.employed
    >>> (sim.wrk.periods_left[employed_mask] >= 0).all()
    True

    Check expired contracts:

    >>> expired_mask = sim.wrk.contract_expired
    >>> expired_mask.sum()  # doctest: +SKIP
    12

    Verify expired workers are unemployed:

    >>> (sim.wrk.employer[expired_mask] == -1).all()
    True

    Verify labor counts match:

    >>> worker_counts = np.bincount(
    ...     sim.wrk.employer[sim.wrk.employed], minlength=sim.emp.current_labor.size
    ... )
    >>> np.array_equal(worker_counts, sim.emp.current_labor)
    True

    Notes
    -----
    This event must execute after FirmsRunProduction (end of period).

    Workers with expired contracts have contract_expired flag set, which
    triggers loyalty behavior in the next labor market phase (workers apply
    to previous employer first if not fired).

    Firm labor and wage bill are recalculated to reflect contract expirations.
    This ensures consistency between Worker and Employer states.

    See Also
    --------
    FirmsHireWorkers : Sets initial contract duration
    WorkersDecideFirmsToApply : Uses contract_expired flag for loyalty
    Worker : Employment state with contract fields
    Employer : Labor force state with current_labor, wage_bill
    bamengine.events._internal.production.workers_update_contracts : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.production import workers_update_contracts

        workers_update_contracts(sim.wrk, sim.emp)
