"""Bankruptcy events for insolvency detection and agent replacement.

This module contains Event classes that wrap bankruptcy system functions.
Each event encapsulates net worth updates, bankruptcy detection, and
spawning of replacement agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


@event
class FirmsUpdateNetWorth:
    """
    Firms update net worth based on current financial position.

    Net worth accumulates profits/losses over the period. This event
    calculates final net worth after all financial transactions.

    This event wraps `bamengine.systems.bankruptcy.firms_update_net_worth`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute net worth update."""
        from bamengine.systems.bankruptcy import firms_update_net_worth

        firms_update_net_worth(sim.bor)


@event
class MarkBankruptFirms:
    """
    Identify and mark bankrupt firms.

    Firms with negative net worth or zero production are marked as bankrupt.
    Bankrupt firms lay off all workers and are removed from the loan book.

    This event wraps `bamengine.systems.bankruptcy.mark_bankrupt_firms`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute bankruptcy marking for firms."""
        from bamengine.systems.bankruptcy import mark_bankrupt_firms

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
    Identify and mark bankrupt banks.

    Banks with negative equity are marked as bankrupt and removed from
    the loan book.

    This event wraps `bamengine.systems.bankruptcy.mark_bankrupt_banks`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute bankruptcy marking for banks."""
        from bamengine.systems.bankruptcy import mark_bankrupt_banks

        mark_bankrupt_banks(sim.ec, sim.lend, sim.lb)


@event
class SpawnReplacementFirms:
    """
    Spawn replacement firms to replace bankrupt ones.

    New firms are created with randomized initial conditions, maintaining
    constant population size.

    This event wraps `bamengine.systems.bankruptcy.spawn_replacement_firms`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute firm replacement spawning."""
        from bamengine.systems.bankruptcy import spawn_replacement_firms

        spawn_replacement_firms(
            sim.ec,
            sim.prod,
            sim.emp,
            sim.bor,
            rng=sim.rng,
        )


@event
class SpawnReplacementBanks:
    """
    Spawn replacement banks to replace bankrupt ones.

    New banks are created with randomized initial conditions, maintaining
    constant population size.

    This event wraps `bamengine.systems.bankruptcy.spawn_replacement_banks`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute bank replacement spawning."""
        from bamengine.systems.bankruptcy import spawn_replacement_banks

        spawn_replacement_banks(
            sim.ec,
            sim.lend,
            rng=sim.rng,
        )
