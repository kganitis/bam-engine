"""
=============================
Growth+ Model Extension
=============================

This example implements the Growth+ extension from chapter 3.8 of Macroeconomics from
the Bottom-up, demonstrating endogenous productivity growth based on R&D investment.

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

Where:
- :math:`\\sigma` = R&D share of profits (varies with fragility)
- :math:`\\pi` = net profit (positive only)
- :math:`p` = firm's selling price
- :math:`Y` = production quantity

**R&D Share (parameterized):**

.. math::

    \\sigma = \\sigma_{min} + (\\sigma_{max} - \\sigma_{min}) \\cdot \\exp(k \\cdot \\text{fragility})

Where:
- :math:`\\sigma_{min}` = R&D share for poorest firms (default: 0.0)
- :math:`\\sigma_{max}` = R&D share for richest firms (default: 0.1)
- :math:`k` = decay rate (default: -1.0, negative means higher fragility → lower R&D)
- fragility = W/A (wage_bill / net_worth)

Firms with higher financial fragility invest less in R&D.

**Net Worth Evolution (Equation 3.16):**

.. math::

    A_t = A_{t-1} + (1-\\sigma)(1-\\delta)\\pi_{t-1}

Where :math:`\\delta` is the dividend payout ratio.

This example demonstrates:

- Defining custom roles with the ``@role`` decorator
- Creating custom events with the ``@event`` decorator
- Using pipeline hooks via ``@event(after=...)`` for automatic event positioning
- Attaching custom roles to simulations via ``sim.use_role()``
- Passing extension parameters to ``Simulation.init()``
- Accessing extension parameters directly as ``sim.param_name``
- Collecting custom role data in simulation results
- Using ``results.get_array()`` for easy data access
"""

# %%
# Import Dependencies
# -------------------
#
# We import BAM Engine and the decorators needed to define custom components.

import bamengine as bam
from bamengine import Float, event, logging, ops, role
from bamengine.logging import getLogger

# %%
# Define Custom Role: RnD
# -----------------------
#
# The RnD role tracks R&D-related state for each firm. This extends firms
# with productivity growth capabilities.


@role
class RnD:
    """R&D state for Growth+ extension.

    Tracks R&D investment decisions and productivity increments for firms.

    Parameters
    ----------
    sigma : Float
        R&D share of profits (0.0 to 0.1). Higher values mean more
        investment in R&D. Decreases with financial fragility.
    rnd_intensity : Float
        Expected productivity gain (mu). Scale parameter for the
        exponential distribution from which actual gains are drawn.
    productivity_increment : Float
        Actual productivity increment (z) drawn each period.
        Added to labor_productivity.
    fragility : Float
        Financial fragility metric (W/A = wage_bill / net_worth).
        High fragility leads to lower R&D investment.
    """

    sigma: Float
    rnd_intensity: Float
    productivity_increment: Float
    fragility: Float


print(f"Custom {RnD.name} role defined!")

# %%
# Define Custom Events
# --------------------
#
# We define three events that implement the Growth+ mechanism:
#
# 1. ``FirmsComputeRnDIntensity``: Calculate R&D share and intensity
# 2. ``FirmsApplyProductivityGrowth``: Draw and apply productivity increments
# 3. ``FirmsDeductRnDExpenditure``: Adjust retained profits for R&D spending


@event(after="firms_pay_dividends")
class FirmsComputeRnDIntensity:
    """Compute R&D share and intensity for firms.

    Calculates:
    - fragility = wage_bill / net_worth
    - sigma = sigma_min + (sigma_max - sigma_min) * exp(sigma_decay * fragility)
    - mu = sigma * net_profit / (price * production)

    Requires extension parameters: sigma_min, sigma_max, sigma_decay
    Firms with non-positive profits have sigma = 0 (no R&D).

    Note: This event is automatically inserted after 'firms_pay_dividends'
    via the ``@event(after=...)`` hook.
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


@event(after="firms_compute_rn_d_intensity")
class FirmsApplyProductivityGrowth:
    """Apply productivity growth based on R&D.

    For firms with positive R&D intensity (mu > 0):
    - Draw z from Exponential(scale=mu)
    - Update: labor_productivity += z

    This implements equation 3.15 from Macroeconomics from the Bottom-up.

    Note: This event is automatically inserted after 'firms_compute_rn_d_intensity'
    via the ``@event(after=...)`` hook.
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
    """Adjust retained profits for R&D expenditure.

    Modifies retained profit calculation:
    - new_retained = old_retained * (1 - sigma)

    This implements the (1-sigma) factor in equation 3.16,
    ensuring retained profits account for R&D spending.

    Note: This event is automatically inserted after 'firms_apply_productivity_growth'
    via the ``@event(after=...)`` hook.
    """

    def execute(self, sim: bam.Simulation) -> None:
        """Execute R&D expenditure deduction."""
        bor = sim.get_role("Borrower")
        rnd = sim.get_role("RnD")

        # Adjust retained profit: multiply by (1 - sigma)
        # This captures the R&D expenditure before profit retention
        one_minus_sigma = ops.subtract(1.0, rnd.sigma)
        new_retained = ops.multiply(bor.retained_profit, one_minus_sigma)
        ops.assign(bor.retained_profit, new_retained)


print("Custom events defined:")
print(f"  - {FirmsComputeRnDIntensity.name}")
print(f"  - {FirmsApplyProductivityGrowth.name}")
print(f"  - {FirmsDeductRnDExpenditure.name}")

# %%
# Initialize Simulation
# ---------------------
#
# Create a baseline simulation and attach the custom RnD role using ``use_role()``.

sim = bam.Simulation.init(
    n_firms=100,
    n_households=500,
    n_banks=10,
    n_periods=1000,
    seed=42,
    # Growth+ R&D parameters (extension parameters)
    sigma_min=0.0,  # R&D share for poorest (highest fragility) firms
    sigma_max=0.1,  # R&D share for richest (lowest fragility) firms
    sigma_decay=-1.0,  # Decay rate: negative means higher fragility → lower sigma
)

# Attach custom RnD role using use_role() - automatically initializes with zeros
rnd = sim.use_role(RnD)

print("\nGrowth+ simulation initialized:")
print(f"  - {sim.n_firms} firms")
print(f"  - {sim.n_households} households")
print(f"  - {sim.n_banks} banks")
print(f"  - Custom RnD role attached: {rnd is not None}")
print(
    f"  - Extension params: sigma_min={sim.sigma_min}, sigma_max={sim.sigma_max}, "
    f"sigma_decay={sim.sigma_decay}"
)

# Verify role is accessible via get_role() too
assert sim.get_role("RnD") is rnd

# %%
# Run Growth+ Simulation
# ----------------------
#
# Run the simulation for 1000 periods, collecting both standard and custom role data.

log = getLogger()
log.setLevel(logging.ERROR)

print("\nRunning Growth+ simulation...")

results = sim.run(
    collect={
        "roles": ["Producer", "Worker", "Borrower"],
        "variables": {
            "Producer": ["labor_productivity", "production", "price"],
            "Worker": ["wage", "employed"],
            "Borrower": ["net_worth", "net_profit"],
        },
        "include_economy": True,
    }
)

print(f"Simulation completed: {results.metadata['n_periods']} periods")
print(f"Runtime: {results.metadata['runtime_seconds']:.2f} seconds")

# %%
# Extract and Analyze Results
# ---------------------------
#
# Compare productivity growth over time using ``get_array()`` for easy data access.

# Get productivity data using get_array() - cleaner than navigating nested dicts
productivity = results.get_array("Producer", "labor_productivity")
production = results.get_array("Producer", "production")
price = results.get_array("Producer", "price")
net_worth = results.get_array("Borrower", "net_worth")
wages = results.get_array("Worker", "wage")
employed = results.get_array("Worker", "employed")

# Economy data - use "Economy" as role name
unemployment = results.get_array("Economy", "unemployment_rate")
inflation = results.get_array("Economy", "inflation")
avg_price = results.get_array("Economy", "avg_price")

print("Data shapes:")
print(f"  productivity: {productivity.shape}")
print(f"  production: {production.shape}")
print(f"  unemployment: {unemployment.shape}")

# Calculate aggregates (axis=1 for per-period mean/sum across agents)
if productivity.ndim == 2:
    avg_productivity = ops.mean(productivity, axis=1)
    gdp = ops.sum(production, axis=1)
    total_net_worth = ops.sum(net_worth, axis=1)
else:
    # If data is already aggregated or 1D
    avg_productivity = productivity
    gdp = production
    total_net_worth = net_worth

# Calculate productivity growth rate
prod_growth = ops.divide(
    avg_productivity[1:] - avg_productivity[:-1],
    ops.where(avg_productivity[:-1] > 0, avg_productivity[:-1], 1.0),
)

print("\nGrowth+ Results Summary:")
print(f"  Initial avg productivity: {avg_productivity[0]:.4f}")
print(f"  Final avg productivity: {avg_productivity[-1]:.4f}")
print(
    f"  Productivity growth: {(avg_productivity[-1] / avg_productivity[0] - 1) * 100:.1f}%"
)
print(
    f"  Mean productivity growth rate: {float(ops.mean(prod_growth)) * 100:.2f}% per period"
)

# %%
# Visualization
# -------------
#
# Plot the key Growth+ dynamics.

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Apply burn-in
burn_in = min(500, len(avg_productivity) // 2)  # Ensure burn_in is valid
periods = range(burn_in, len(avg_productivity))

# 1. Productivity Evolution
ax1 = axes[0, 0]
ax1.plot(periods, avg_productivity[burn_in:], "b-", linewidth=1)
ax1.set_xlabel("Period")
ax1.set_ylabel("Average Labor Productivity")
ax1.set_title("Endogenous Productivity Growth")
ax1.grid(True, alpha=0.3)

# 2. GDP (Real Output)
ax2 = axes[0, 1]
ax2.plot(periods, gdp[burn_in:], "g-", linewidth=1)
ax2.set_xlabel("Period")
ax2.set_ylabel("Total Production (GDP)")
ax2.set_title("Real GDP")
ax2.grid(True, alpha=0.3)

# 3. Unemployment Rate
ax3 = axes[1, 0]
ax3.plot(periods, unemployment[burn_in:] * 100, "r-", linewidth=1)
ax3.set_xlabel("Period")
ax3.set_ylabel("Unemployment Rate (%)")
ax3.set_title("Unemployment")
ax3.grid(True, alpha=0.3)

# 4. Inflation Rate
ax4 = axes[1, 1]
ax4.plot(periods, inflation[burn_in:] * 100, "purple", linewidth=1)
ax4.set_xlabel("Period")
ax4.set_ylabel("Annual Inflation Rate (%)")
ax4.set_title("Inflation")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("growth_plus_dynamics.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nFigure saved as 'growth_plus_dynamics.png'")

# %%
# Summary Statistics
# ------------------
#
# Report key statistics from the Growth+ simulation.

# Post burn-in statistics
post_burn = burn_in

print("\n" + "=" * 50)
print("Growth+ Model Summary Statistics")
print("=" * 50)
print("\nProductivity:")
print(f"  Start (period {burn_in}): {avg_productivity[post_burn]:.4f}")
print(f"  End (period {len(avg_productivity) - 1}): {avg_productivity[-1]:.4f}")
print(
    f"  Total growth: {(avg_productivity[-1] / avg_productivity[post_burn] - 1) * 100:.1f}%"
)

print("\nOutput (GDP):")
print(f"  Mean: {float(ops.mean(gdp[post_burn:])):.1f}")
print(f"  Std Dev: {float(ops.std(gdp[post_burn:])):.1f}")

print("\nUnemployment Rate:")
print(f"  Mean: {float(ops.mean(unemployment[post_burn:])) * 100:.1f}%")
print(f"  Std Dev: {float(ops.std(unemployment[post_burn:])) * 100:.2f}%")

print("\nInflation Rate:")
print(f"  Mean: {float(ops.mean(inflation[post_burn:])) * 100:.2f}%")
print(f"  Std Dev: {float(ops.std(inflation[post_burn:])) * 100:.2f}%")

print("\n" + "=" * 50)
print("Growth+ extension demonstrates endogenous productivity")
print("growth through R&D investment by financially healthy firms.")
print("=" * 50)
