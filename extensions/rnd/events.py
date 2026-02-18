"""R&D events for Growth+ extension.

This module provides the three custom events for the Growth+ scenario
from section 3.8 of Delli Gatti et al. (2011).

The extension adds endogenous productivity growth via R&D investment:
- Firms invest a portion of profits in R&D (sigma = RnD share)
- R&D intensity determines expected productivity gains
- Productivity increments are drawn from exponential distribution
- Higher financial fragility leads to lower R&D investment

Key Equations
-------------

**Productivity Evolution (Equation 3.15):**

.. math::

    \\alpha_{t+1} = \\alpha_t + z_t

Where :math:`z_t \\sim \\text{Exponential}(\\mu)` represents the productivity
increment drawn from an exponential distribution with scale parameter :math:`\\mu`.

**R&D Intensity (expected productivity gain):**

.. math::

    \\mu = \\sigma \\cdot \\frac{\\pi}{p \\cdot Y}

**R&D Share (parameterized):**

.. math::

    \\sigma = \\sigma_{min} + (\\sigma_{max} - \\sigma_{min}) \\cdot \\exp(k \\cdot \\text{fragility})

**Net Worth Evolution (Equation 3.16):**

.. math::

    A_t = A_{t-1} + (1-\\sigma)(1-\\delta)\\pi_{t-1}
"""

from __future__ import annotations

import bamengine as bam
from bamengine import event, ops


@event(
    name="firms_compute_rnd_intensity",
    after="firms_validate_debt_commitments",
)
class FirmsComputeRnDIntensity:
    """Compute R&D share and intensity for firms.

    Calculates:
    - fragility = wage_bill / net_worth
    - sigma = sigma_min + (sigma_max - sigma_min) * exp(sigma_decay * fragility)
    - mu = sigma * net_profit / (price * production)

    Requires extension parameters: sigma_min, sigma_max, sigma_decay
    Firms with non-positive profits have sigma = 0 (no R&D).

    Note: This event is positioned after 'firms_validate_debt_commitments'
    (before 'firms_pay_dividends') via the ``@event(after=...)`` hook so that
    R&D deducts from net profit *before* dividend distribution. Apply with
    ``sim.use_events(*RND_EVENTS)``.
    """

    def execute(self, sim: bam.Simulation) -> None:
        """Execute R&D intensity computation."""
        bor = sim.get_role("Borrower")
        prod = sim.get_role("Producer")
        emp = sim.get_role("Employer")
        rnd = sim.get_role("RnD")

        # Access extension parameters directly via sim.param_name
        sigma_min = sim.sigma_min
        sigma_max = sim.sigma_max
        sigma_decay = sim.sigma_decay

        # Calculate fragility = W/A (wage_bill / net_worth)
        # Use safe division with small epsilon to avoid division by zero
        eps = 1e-10
        safe_net_worth = ops.where(ops.greater(bor.net_worth, eps), bor.net_worth, eps)
        fragility = ops.divide(emp.wage_bill, safe_net_worth)

        # Store fragility
        ops.assign(rnd.fragility, fragility)

        # Calculate sigma = sigma_min + (sigma_max - sigma_min) * exp(sigma_decay * fragility)
        decay_factor = ops.exp(ops.multiply(sigma_decay, fragility))
        sigma_range = sigma_max - sigma_min
        sigma = ops.add(sigma_min, ops.multiply(sigma_range, decay_factor))

        # Set sigma = 0 for firms with non-positive net profit
        sigma = ops.where(ops.greater(bor.net_profit, 0.0), sigma, 0.0)
        ops.assign(rnd.sigma, sigma)

        # Calculate mu = sigma * net_profit / (price * production)
        # This is the expected productivity gain (scale parameter for exponential)
        revenue = ops.multiply(prod.price, prod.production)
        safe_revenue = ops.where(ops.greater(revenue, eps), revenue, eps)
        mu = ops.divide(ops.multiply(sigma, bor.net_profit), safe_revenue)

        # Clamp mu to reasonable range
        mu = ops.where(ops.greater(mu, 0.0), mu, 0.0)
        ops.assign(rnd.rnd_intensity, mu)


@event(after="firms_compute_rnd_intensity")
class FirmsApplyProductivityGrowth:
    """Apply productivity growth based on R&D.

    For firms with positive R&D intensity (mu > 0):
    - Draw z from Exponential(scale=mu)
    - Update: labor_productivity += z

    This implements equation 3.15 from Macroeconomics from the Bottom-up.

    Note: This event is positioned after 'firms_compute_rnd_intensity' via the
    ``@event(after=...)`` hook. Apply with ``sim.use_events(*RND_EVENTS)``.
    """

    def execute(self, sim: bam.Simulation) -> None:
        """Execute productivity growth."""
        prod = sim.get_role("Producer")
        rnd = sim.get_role("RnD")

        # Draw productivity increments from exponential distribution
        # z ~ Exponential(scale=mu), where E[z] = mu
        # Only for firms with mu > 0
        n_firms = sim.n_firms
        mu = rnd.rnd_intensity

        # Draw from exponential - use sim.rng for reproducibility
        # For firms with mu=0, we set z=0
        z = ops.zeros(n_firms)
        active = ops.greater(mu, 0.0)
        if ops.any(active):
            # Draw from exponential with scale=mu for active firms
            # Note: sim.rng.exponential is proper RNG usage
            z[active] = sim.rng.exponential(scale=mu[active])

        # Store the increment
        ops.assign(rnd.productivity_increment, z)

        # Apply to labor productivity: alpha_{t+1} = alpha_t + z
        new_productivity = ops.add(prod.labor_productivity, z)
        ops.assign(prod.labor_productivity, new_productivity)


@event(after="firms_apply_productivity_growth")
class FirmsDeductRnDExpenditure:
    """Adjust net profit for R&D expenditure.

    Modifies net profit before dividend distribution:
    - new_net_profit = old_net_profit * (1 - sigma)

    This implements the (1-sigma) factor in equation 3.16. By reducing
    net_profit before ``firms_pay_dividends``, dividends correctly equal
    delta * (1-sigma) * pi, matching book Section 3.8.

    Note: This event is positioned after 'firms_apply_productivity_growth' via the
    ``@event(after=...)`` hook. Apply with ``sim.use_events(*RND_EVENTS)``.
    """

    def execute(self, sim: bam.Simulation) -> None:
        """Execute R&D expenditure deduction."""
        bor = sim.get_role("Borrower")
        rnd = sim.get_role("RnD")

        # Adjust net profit: multiply by (1 - sigma)
        # This captures the R&D expenditure before dividend distribution
        one_minus_sigma = ops.subtract(1.0, rnd.sigma)
        new_net_profit = ops.multiply(bor.net_profit, one_minus_sigma)
        ops.assign(bor.net_profit, new_net_profit)
