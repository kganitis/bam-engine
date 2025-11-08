"""Revenue events for collection, debt repayment, and dividends.

This module contains Event classes that wrap revenue system functions.
Each event encapsulates revenue collection, debt validation and repayment,
and dividend distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bamengine.core import Event

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


@dataclass(slots=True)
class FirmsCollectRevenue(Event):
    """
    Firms collect revenue from goods sold during the period.

    Revenue is calculated from the sales buffer accumulated during shopping:
        R_i = Σ (price_i · quantity_sold)

    Revenue is added to net worth.

    This event wraps `bamengine.systems.revenue.firms_collect_revenue`.

    Dependencies
    ------------
    - consumers_finalize_purchases : All sales complete

    See Also
    --------
    bamengine.systems.revenue.firms_collect_revenue : Underlying logic
    """

    def execute(self, sim: Simulation) -> None:
        """Execute revenue collection."""
        from bamengine.systems.revenue import firms_collect_revenue

        firms_collect_revenue(sim.prod, sim.bor)


@dataclass(slots=True)
class FirmsValidateDebtCommitments(Event):
    """
    Firms repay loans and update debt obligations.

    Firms repay principal and interest on outstanding loans. Banks collect
    payments and update equity. The loan book is updated to reflect repayments
    and remove fully paid loans.

    This event wraps `bamengine.systems.revenue.firms_validate_debt_commitments`.

    Dependencies
    ------------
    - firms_collect_revenue : Uses revenue for debt repayment

    See Also
    --------
    bamengine.systems.revenue.firms_validate_debt_commitments : Underlying logic
    """

    def execute(self, sim: Simulation) -> None:
        """Execute debt validation and repayment."""
        from bamengine.systems.revenue import firms_validate_debt_commitments

        firms_validate_debt_commitments(sim.bor, sim.lend, sim.lb)


@dataclass(slots=True)
class FirmsPayDividends(Event):
    """
    Firms distribute dividends to owners (shareholders).

    Dividends are paid as a fraction of net worth:
        dividends_i = δ · max(NW_i, 0)

    where δ is the dividend payout ratio.

    This event wraps `bamengine.systems.revenue.firms_pay_dividends`.

    Dependencies
    ------------
    - firms_validate_debt_commitments : After debt repayment

    See Also
    --------
    bamengine.systems.revenue.firms_pay_dividends : Underlying logic
    """

    def execute(self, sim: Simulation) -> None:
        """Execute dividend payment."""
        from bamengine.systems.revenue import firms_pay_dividends

        firms_pay_dividends(sim.bor, delta=sim.config.delta)
