"""Taxation events for the entry neutrality experiment.

Implements a simple profit tax that deducts from net_profit and total_funds
without redistributing the revenue. This creates fiscal drag that increases
bankruptcy rates, testing whether the automatic firm entry mechanism
artificially drives recovery (Section 3.10.2).
"""

from __future__ import annotations

import bamengine as bam
from bamengine import event, ops


@event(
    name="firms_tax_profits",
    after="firms_validate_debt_commitments",
)
class FirmsTaxProfits:
    """Apply profit taxation to firms with positive net profit.

    Tax revenue vanishes (no redistribution). This is intentional:
    the experiment tests whether firm entry alone can sustain recovery
    when profits are heavily taxed.

    Tax = rate * max(0, net_profit)
    net_profit -= tax
    total_funds -= tax

    Requires extension parameter: ``profit_tax_rate`` (float, 0 to 1).

    Note: Positioned after ``firms_validate_debt_commitments`` (before
    ``firms_pay_dividends``) so tax is deducted before dividend distribution.
    Apply with ``sim.use_events(*TAXATION_EVENTS)``.
    """

    def execute(self, sim: bam.Simulation) -> None:
        rate = sim.profit_tax_rate
        if rate <= 0.0:
            return

        bor = sim.get_role("Borrower")

        # Tax only positive profits
        taxable = ops.where(ops.greater(bor.net_profit, 0.0), bor.net_profit, 0.0)
        tax = ops.multiply(rate, taxable)

        # Deduct from net_profit and total_funds
        ops.assign(bor.net_profit, ops.subtract(bor.net_profit, tax))
        ops.assign(bor.total_funds, ops.subtract(bor.total_funds, tax))
