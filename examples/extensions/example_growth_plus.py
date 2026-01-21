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

This example demonstrates:

- Defining custom roles with the ``@role`` decorator
- Creating custom events with the ``@event`` decorator
- Using pipeline hooks via ``@event(after=...)`` for automatic event positioning
- Attaching custom roles to simulations via ``sim.use_role()``

For detailed validation with bounds and statistical annotations, run:
    python -m validation.scenarios.growth_plus
"""

# %%
# Import Dependencies
# -------------------
#
# We import BAM Engine and the decorators needed to define custom components.

import bamengine as bam
from bamengine import Float, event, ops, role

# %%
# Define Custom Role: RnD
# -----------------------
#
# The RnD role tracks R&D-related state for each firm.


@role
class RnD:
    """R&D state for Growth+ extension."""

    sigma: Float  # R&D share of profits
    rnd_intensity: Float  # Expected productivity gain (mu)
    productivity_increment: Float  # Actual productivity increment (z)
    fragility: Float  # Financial fragility (wage_bill / net_worth)


# %%
# Define Custom Events
# --------------------
#
# Three events implement the Growth+ mechanism.


@event(name="firms_compute_rnd_intensity", after="firms_pay_dividends")
class FirmsComputeRnDIntensity:
    """Compute R&D share and intensity for firms."""

    def execute(self, sim: bam.Simulation) -> None:
        bor = sim.get_role("Borrower")
        prod = sim.get_role("Producer")
        emp = sim.get_role("Employer")
        rnd = sim.get_role("RnD")

        sigma_min, sigma_max, sigma_decay = (
            sim.sigma_min,
            sim.sigma_max,
            sim.sigma_decay,
        )
        eps = 1e-10

        safe_net_worth = ops.where(ops.greater(bor.net_worth, eps), bor.net_worth, eps)
        fragility = ops.divide(emp.wage_bill, safe_net_worth)
        ops.assign(rnd.fragility, fragility)

        decay_factor = ops.exp(ops.multiply(sigma_decay, fragility))
        sigma = ops.add(sigma_min, ops.multiply(sigma_max - sigma_min, decay_factor))
        sigma = ops.where(ops.greater(bor.net_profit, 0.0), sigma, 0.0)
        ops.assign(rnd.sigma, sigma)

        revenue = ops.multiply(prod.price, prod.production)
        safe_revenue = ops.where(ops.greater(revenue, eps), revenue, eps)
        mu = ops.divide(ops.multiply(sigma, bor.net_profit), safe_revenue)
        mu = ops.where(ops.greater(mu, 0.0), mu, 0.0)
        ops.assign(rnd.rnd_intensity, mu)


@event(after="firms_compute_rnd_intensity")
class FirmsApplyProductivityGrowth:
    """Apply productivity growth based on R&D."""

    def execute(self, sim: bam.Simulation) -> None:
        prod = sim.get_role("Producer")
        rnd = sim.get_role("RnD")

        z = ops.zeros(sim.n_firms)
        active = ops.greater(rnd.rnd_intensity, 0.0)
        if ops.any(active):
            z[active] = sim.rng.exponential(scale=rnd.rnd_intensity[active])

        ops.assign(rnd.productivity_increment, z)
        ops.assign(prod.labor_productivity, ops.add(prod.labor_productivity, z))


@event(after="firms_apply_productivity_growth")
class FirmsDeductRnDExpenditure:
    """Adjust retained profits for R&D expenditure."""

    def execute(self, sim: bam.Simulation) -> None:
        bor = sim.get_role("Borrower")
        rnd = sim.get_role("RnD")
        new_retained = ops.multiply(bor.retained_profit, ops.subtract(1.0, rnd.sigma))
        ops.assign(bor.retained_profit, new_retained)


# %%
# Main Execution Guard
# --------------------
#
# Run the simulation only when executed directly.

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Initialize simulation with Growth+ parameters
    sim = bam.Simulation.init(
        n_firms=100,
        n_households=500,
        n_banks=10,
        n_periods=1000,
        seed=0,
        logging={"default_level": "ERROR"},
        new_firm_size_factor=0.5,
        new_firm_production_factor=0.5,
        new_firm_wage_factor=0.5,
        new_firm_price_markup=1.5,
        sigma_min=0.0,
        sigma_max=0.1,
        sigma_decay=-1.0,
    )

    sim.use_role(RnD)
    print(f"Growth+ simulation: {sim.n_firms} firms, {sim.n_households} households")

    # Run simulation
    COLLECT_CONFIG = {
        "Producer": ["production", "labor_productivity"],
        "Worker": ["wage", "employed"],
        "Employer": ["n_vacancies"],
        "Economy": True,
        "aggregate": None,  # Collect per-agent data (2D arrays)
        "capture_timing": {
            "Worker.wage": "workers_receive_wage",
            "Worker.employed": "firms_run_production",
            "Producer.production": "firms_run_production",
            "Employer.n_vacancies": "firms_decide_vacancies",
        },
    }
    results = sim.run(collect=COLLECT_CONFIG)
    print(f"Completed: {results.metadata['runtime_seconds']:.2f}s")

    # Compute metrics
    burn_in = 500
    n_periods = sim.n_periods

    # Extract raw data from results
    avg_price = results.economy_data["avg_price"]
    production = results.get_array("Producer", "production")
    productivity = results.get_array("Producer", "labor_productivity")
    wages = results.get_array("Worker", "wage")
    employed_arr = results.get_array("Worker", "employed")
    n_vacancies = results.get_array("Employer", "n_vacancies")

    # Compute total production (GDP)
    total_production = ops.sum(production, axis=1)

    # Unemployment rate
    unemployment = 1 - ops.mean(employed_arr.astype(float), axis=1)

    # Log GDP
    log_gdp = ops.log(total_production + 1e-10)

    # Inflation
    inflation = results.economy_data.get("inflation", np.zeros(n_periods))

    # Average wage for employed workers
    employed_wages_sum = ops.sum(ops.where(employed_arr, wages, 0.0), axis=1)
    employed_count = ops.sum(employed_arr, axis=1)
    avg_wage = ops.where(
        ops.greater(employed_count, 0),
        ops.divide(employed_wages_sum, employed_count),
        0.0,
    )

    # Real wage
    real_wage = ops.divide(avg_wage, avg_price)

    # Production-weighted average productivity
    weighted_prod = ops.sum(ops.multiply(productivity, production), axis=1)
    avg_productivity = ops.divide(weighted_prod, total_production)

    # Wage inflation for Phillips curve
    wage_inflation = ops.divide(
        avg_wage[1:] - avg_wage[:-1],
        ops.where(ops.greater(avg_wage[:-1], 0), avg_wage[:-1], 1.0),
    )

    # GDP growth for Okun curve
    gdp_growth = ops.divide(
        total_production[1:] - total_production[:-1], total_production[:-1]
    )

    # Unemployment growth for Okun curve
    unemployment_growth = ops.divide(
        unemployment[1:] - unemployment[:-1],
        ops.where(ops.greater(unemployment[:-1], 0), unemployment[:-1], 1.0),
    )

    # Vacancy rate
    total_vacancies = ops.sum(n_vacancies, axis=1)
    vacancy_rate = ops.divide(total_vacancies, sim.n_households)

    prod = sim.get_role("Producer")
    final_production = prod.production.copy()

    phillips_corr = np.corrcoef(unemployment[burn_in:], wage_inflation[burn_in - 1 :])[
        0, 1
    ]
    okun_corr = np.corrcoef(
        unemployment_growth[burn_in - 1 :], gdp_growth[burn_in - 1 :]
    )[0, 1]
    beveridge_corr = np.corrcoef(unemployment[burn_in:], vacancy_rate[burn_in:])[0, 1]

    prod_growth = (avg_productivity[-1] - avg_productivity[burn_in]) / avg_productivity[
        burn_in
    ]
    print(f"Productivity growth: {prod_growth * 100:.0f}%")

    # Visualize
    periods = ops.arange(burn_in, n_periods)
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    fig.suptitle("Growth+ Model (Section 3.8)", fontsize=16, y=0.995)

    axes[0, 0].plot(periods, log_gdp[burn_in:], linewidth=1, color="#2E86AB")
    axes[0, 0].set_title("Real GDP (Growing)", fontweight="bold")
    axes[0, 0].set_ylabel("Log output")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].grid(True, linestyle="--", alpha=0.3)

    axes[0, 1].plot(periods, unemployment[burn_in:] * 100, linewidth=1, color="#A23B72")
    axes[0, 1].set_title("Unemployment Rate", fontweight="bold")
    axes[0, 1].set_ylabel("Unemployment (%)")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylim(bottom=0)
    axes[0, 1].grid(True, linestyle="--", alpha=0.3)

    axes[1, 0].plot(periods, inflation[burn_in:] * 100, linewidth=1, color="#F18F01")
    axes[1, 0].axhline(0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
    axes[1, 0].set_title("Inflation Rate", fontweight="bold")
    axes[1, 0].set_ylabel("Yearly inflation (%)")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].grid(True, linestyle="--", alpha=0.3)

    axes[1, 1].plot(
        periods,
        avg_productivity[burn_in:],
        linewidth=1,
        color="#E74C3C",
        label="Productivity",
    )
    axes[1, 1].plot(
        periods, real_wage[burn_in:], linewidth=1, color="#6A994E", label="Real Wage"
    )
    axes[1, 1].set_title("Productivity & Real Wage", fontweight="bold")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].legend(loc="lower right", fontsize=8)
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)

    axes[2, 0].scatter(
        unemployment[burn_in:],
        wage_inflation[burn_in - 1 :],
        s=10,
        alpha=0.5,
        color="#2E86AB",
    )
    axes[2, 0].set_title(f"Phillips Curve (r={phillips_corr:.2f})", fontweight="bold")
    axes[2, 0].set_xlabel("Unemployment Rate")
    axes[2, 0].set_ylabel("Wage Inflation Rate")
    axes[2, 0].grid(True, linestyle="--", alpha=0.3)

    axes[2, 1].scatter(
        unemployment_growth[burn_in - 1 :],
        gdp_growth[burn_in - 1 :],
        s=2,
        alpha=0.5,
        color="#A23B72",
    )
    axes[2, 1].set_title(f"Okun Curve (r={okun_corr:.2f})", fontweight="bold")
    axes[2, 1].set_xlabel("Unemployment Growth")
    axes[2, 1].set_ylabel("Output Growth")
    axes[2, 1].grid(True, linestyle="--", alpha=0.3)

    axes[3, 0].scatter(
        unemployment[burn_in:], vacancy_rate[burn_in:], s=10, alpha=0.5, color="#F18F01"
    )
    axes[3, 0].set_title(f"Beveridge Curve (r={beveridge_corr:.2f})", fontweight="bold")
    axes[3, 0].set_xlabel("Unemployment Rate")
    axes[3, 0].set_ylabel("Vacancy Rate")
    axes[3, 0].grid(True, linestyle="--", alpha=0.3)

    axes[3, 1].hist(
        final_production, bins=15, edgecolor="black", alpha=0.7, color="#6A994E"
    )
    axes[3, 1].set_title("Firm Size Distribution", fontweight="bold")
    axes[3, 1].set_xlabel("Production")
    axes[3, 1].set_ylabel("Frequency")
    axes[3, 1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()
