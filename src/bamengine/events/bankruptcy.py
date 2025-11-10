"""Bankruptcy events for insolvency detection and agent replacement.

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
    Update firm net worth with retained earnings from the current period.

    Rule
    ----
        A_t = A_{t-1} + retained_t
        funds_t = max(0, A_t)

    t: Current Period, A: Net Worth,
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.bankruptcy import firms_update_net_worth

        firms_update_net_worth(sim.bor)


@event
class MarkBankruptFirms:
    """
    Detect and process firm bankruptcies based on net worth or production.

    Rule
    ----
    A firm is marked as bankrupt if either:
      • Net Worth (A) < 0
      • Current Production (Y) <= 0

    Note
    ----
    For bankrupt firms, all workers are fired and loans are purged.
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
    Detect and process bank bankruptcies based on negative equity.

    Rule
    ----
        A bank is marked as bankrupt if equity (E) < 0.

    Note
    ----
    For bankrupt banks, all associated loans are purged from the loan book.
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.bankruptcy import mark_bankrupt_banks

        mark_bankrupt_banks(sim.ec, sim.lend, sim.lb)


@event
class SpawnReplacementFirms:
    """
    Create one brand-new firm to replace each firm that went bankrupt.

    Rule
    ----
    New firms inherit attributes based on the trimmed mean of surviving firms,
    scaled by a factor `s`.
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.bankruptcy import spawn_replacement_firms

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
    Create one brand-new bank to replace each bank that went bankrupt.

    Rule
    ----
    New banks clone the equity of a random surviving peer. If no peers exist,
    the simulation is terminated.
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.bankruptcy import spawn_replacement_banks

        spawn_replacement_banks(
            sim.ec,
            sim.lend,
            rng=sim.rng,
        )
