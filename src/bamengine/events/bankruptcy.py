"""
Bankruptcy events for insolvency detection and agent replacement.

This module defines the bankruptcy phase events that execute at the end of each
period. Firms update net worth with retained earnings, insolvent agents are
detected and removed, and replacement agents are spawned to maintain population.

Event Sequence
--------------
The bankruptcy events execute in this order:

1. FirmsUpdateNetWorth - Add retained profits/losses to net worth
2. MarkBankruptFirms - Detect insolvent firms (A < 0 or Y = 0)
3. MarkBankruptBanks - Detect insolvent banks (E < 0)
4. SpawnReplacementFirms - Create new firms to replace bankrupt ones
5. SpawnReplacementBanks - Create new banks to replace bankrupt ones

Design Notes
------------
- Bankruptcy criteria: firms (A < 0 or Y = 0), banks (E < 0)
- Bankrupt firms: fire all workers, purge loans
- Bankrupt banks: purge all loans
- Replacement firms: inherit trimmed mean of survivors × scale factor
- Replacement banks: clone random surviving bank equity
- Population constant: n_firms, n_banks unchanged

Examples
--------
Execute bankruptcy events:

>>> import bamengine as be
>>> sim = be.Simulation.init(n_firms=100, n_banks=10, seed=42)
>>> # Bankruptcy events run as part of default pipeline
>>> sim.step()

Check bankruptcies:

>>> sim.ec.n_firm_failures  # doctest: +SKIP
2
>>> sim.ec.n_bank_failures  # doctest: +SKIP
0

See Also
--------
bamengine.events._internal.bankruptcy : System function implementations
Economy : Tracks bankruptcy counts
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:  # pragma: no cover
    from bamengine.simulation import Simulation


@event
class FirmsUpdateNetWorth:
    """
    Update firm net worth with retained profits/losses.

    Net worth accumulates retained earnings from each period. Negative net worth
    (insolvency) leads to bankruptcy in the next event.

    Algorithm
    ---------
    For each firm i:

    .. math::
        A_i \\leftarrow A_i + RP_i

        \\text{total\\_funds}_i = \\max(0, A_i)

    where :math:`A_i` = net_worth, :math:`RP_i` = retained_profit.

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> event = sim.get_event("firms_update_net_worth")
    >>> event.execute(sim)

    See Also
    --------
    FirmsPayDividends : Calculates retained_profit
    bamengine.events._internal.bankruptcy.firms_update_net_worth : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.bankruptcy import firms_update_net_worth

        firms_update_net_worth(sim.bor)


@event
class MarkBankruptFirms:
    """
    Detect insolvent firms and remove them from the economy.

    Firms are bankrupt if net_worth < 0 or production_prev = 0 (ghost firm rule).
    Bankrupt firms fire all workers and have all loans purged from LoanBook.

    Note: We check ``production_prev`` (not ``production``) because ``production``
    is zeroed at the start of each period's planning phase. ``production_prev``
    holds the previous period's actual production.

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> event = sim.get_event("mark_bankrupt_firms")
    >>> event.execute(sim)
    >>> sim.ec.n_firm_failures  # doctest: +SKIP
    2

    See Also
    --------
    SpawnReplacementFirms : Creates replacements
    bamengine.events._internal.bankruptcy.mark_bankrupt_firms : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.bankruptcy import mark_bankrupt_firms

        mark_bankrupt_firms(
            sim.ec,
            sim.emp,
            sim.bor,
            sim.prod,
            sim.wrk,
            sim.lb,
        )


@event
class MarkBankruptBanks:
    """
    Detect insolvent banks and remove them from the economy.

    Banks are bankrupt if equity_base < 0. Bankrupt banks have all loans purged.

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_banks=10, seed=42)
    >>> event = sim.get_event("mark_bankrupt_banks")
    >>> event.execute(sim)
    >>> sim.ec.n_bank_failures  # doctest: +SKIP
    0

    See Also
    --------
    SpawnReplacementBanks : Creates replacements
    bamengine.events._internal.bankruptcy.mark_bankrupt_banks : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.bankruptcy import mark_bankrupt_banks

        mark_bankrupt_banks(sim.ec, sim.lend, sim.lb)


@event
class SpawnReplacementFirms:
    """
    Create new firms to replace bankrupt ones.

    Replacement firms inherit attributes (price, wage, net worth) from
    trimmed mean of survivors × scale factor (smaller than average).

    For production fields:
    - ``production = 0`` (no workers yet, keeps end-period aggregate stats clean)
    - ``production_prev = mean_prod × new_firm_production_factor`` (planning signal)

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> event = sim.get_event("spawn_replacement_firms")
    >>> event.execute(sim)

    See Also
    --------
    MarkBankruptFirms : Detects bankruptcies
    bamengine.events._internal.bankruptcy.spawn_replacement_firms : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.bankruptcy import spawn_replacement_firms

        spawn_replacement_firms(
            sim.ec,
            sim.prod,
            sim.emp,
            sim.bor,
            sim.wrk,
            new_firm_size_factor=sim.config.new_firm_size_factor,
            new_firm_production_factor=sim.config.new_firm_production_factor,
            new_firm_wage_factor=sim.config.new_firm_wage_factor,
            new_firm_price_markup=sim.config.new_firm_price_markup,
            rng=sim.rng,
        )


@event
class SpawnReplacementBanks:
    """
    Create new banks to replace bankrupt ones.

    Replacement banks clone equity from random surviving bank. If no banks survive,
    simulation terminates (systemic collapse).

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_banks=10, seed=42)
    >>> event = sim.get_event("spawn_replacement_banks")
    >>> event.execute(sim)

    See Also
    --------
    MarkBankruptBanks : Detects bankruptcies
    bamengine.events._internal.bankruptcy.spawn_replacement_banks : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.bankruptcy import spawn_replacement_banks

        spawn_replacement_banks(
            sim.ec,
            sim.lend,
            rng=sim.rng,
        )
