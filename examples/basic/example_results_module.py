"""
==================
Simulation Results
==================

This example demonstrates how to collect and analyze simulation results
using the ``SimulationResults`` class. Learn how to:

- Collect data during simulation runs
- Access data via ``results["Producer.price"]`` or ``results.Producer.price``
- Use ``results.get()`` for programmatic access with optional aggregation
- Use ``results.available()`` to discover collected variables
- Export data to pandas DataFrames
- Generate summary statistics

The results module makes it easy to extract insights from simulations
without manual data collection.
"""

# %%
# Basic Data Collection
# ---------------------
#
# Pass ``collect=True`` to ``sim.run()`` to collect data automatically.
# This returns a ``SimulationResults`` object containing time series data.

import numpy as np

import bamengine as bam

# Run simulation with data collection
# We collect Worker employed data without aggregation to calculate unemployment
sim = bam.Simulation.init(n_firms=100, n_households=500, seed=42)
results = sim.run(
    n_periods=50,
    collect={
        "Producer": True,
        "Worker": ["employed"],  # Boolean employed status for each worker
        "Employer": True,
        "Borrower": True,
        "Lender": True,
        "Consumer": True,
        # Capture timing: when to snapshot each variable during the period
        # Worker.employed should be captured after production runs (steady state)
        "capture_timing": {
            "Worker.employed": "firms_run_production",
        },
    },
)

print(f"Collected results: {results}")
print("\nMetadata:")
print(f"  Periods simulated: {results.metadata.get('n_periods', 'N/A')}")
print(f"  Firms: {results.metadata.get('n_firms', 'N/A')}")
print(f"  Households: {results.metadata.get('n_households', 'N/A')}")

# %%
# Discovering Available Data
# --------------------------
#
# Use ``results.available()`` to list all collected variables as
# ``"Name.variable"`` strings.

print("All available data:")
for key in results.available():
    print(f"  {key}")

# %%
# Primary Data Access
# -------------------
#
# The primary API uses bracket notation ``results["Name.variable"]``
# or attribute access ``results.Name.variable``.  Economy metrics are
# always collected automatically (no need to request them).

# Bracket access (recommended for most use cases)
avg_price = results["Economy.avg_price"]
inflation = results["Economy.inflation"]

# Attribute access (convenient for interactive exploration)
avg_price_attr = results.Economy.avg_price
print(f"\nAverage price (bracket): {bam.ops.mean(avg_price):.3f}")
print(f"Average price (attr):   {bam.ops.mean(avg_price_attr):.3f}")

# Role data works the same way
worker_employed = results["Worker.employed"]
print(f"Worker employed shape:  {worker_employed.shape}")


# Helper function to calculate unemployment rate from Worker employed data
def calc_unemployment_rate(worker_employed: np.ndarray) -> np.ndarray:
    """Calculate unemployment rate per period from Worker employed boolean array.

    Args:
        worker_employed: 2D boolean array of shape (n_periods, n_workers) where
            True indicates employed, False indicates unemployed.

    Returns:
        1D array of unemployment rates per period.
    """
    # Employment rate = mean of employed (True=1, False=0) across workers
    # Unemployment rate = 1 - employment rate
    return 1.0 - np.mean(worker_employed.astype(float), axis=1)


# Calculate unemployment rate from Worker employed data
unemployment = calc_unemployment_rate(worker_employed)
print("\nUnemployment rate:")
print(f"  Mean: {bam.ops.mean(unemployment):.2%}")
print(f"  Final: {unemployment[-1]:.2%}")

print("\nAverage price:")
print(f"  Mean: {bam.ops.mean(avg_price):.3f}")
print(f"  Final: {avg_price[-1]:.3f}")

# %%
# Accessing Role Data
# -------------------
#
# Role data contains per-period snapshots of agent states.
# With dict-form collect, data is full per-agent arrays (periods x agents) by default.

# Access Producer price data - shape is (periods, firms) with full per-agent data
price_data = results["Producer.price"]
print(f"Price data shape: {price_data.shape}")
# Calculate mean price per period
avg_prices = np.mean(price_data, axis=1)
print(f"Price trend (first 5): {avg_prices[:5].round(3)}")

# %%
# Programmatic Access with get()
# --------------------------------
#
# Use ``results.get()`` for programmatic access, especially when the role
# or variable name comes from a variable.  It also supports on-the-fly
# aggregation.

# get() for role data
prices_via_get = results.get("Producer", "price")
print("\nUsing get():")
print(f"  results.get('Producer', 'price').shape: {prices_via_get.shape}")

# get() for economy data - use "Economy" as the name
price_via_get = results.get("Economy", "avg_price")
print(f"  results.get('Economy', 'avg_price').shape: {price_via_get.shape}")

# Aggregation on-the-fly (useful when you have full per-agent data)
full_data_sim = bam.Simulation.init(n_firms=50, n_households=250, seed=42)
full_data_results = full_data_sim.run(
    n_periods=20,
    collect={
        "Producer": ["price"],  # Specific variables for Producer
    },
)

# Get full 2D data (periods x firms)
prices_2d = full_data_results.get("Producer", "price")
print(f"\n  Full data shape: {prices_2d.shape}")

# Get mean aggregated on-the-fly
prices_mean = full_data_results.get("Producer", "price", aggregate="mean")
print(f"  With aggregate='mean': {prices_mean.shape}")

# %%
# Legacy Access
# -------------
#
# The underlying data is also available via ``role_data``, ``economy_data``,
# and ``relationship_data`` dictionaries.  The ``data`` property merges
# them into a single dict with an "Economy" key.

print("\nLegacy dict access:")
print(f"  results.role_data keys: {list(results.role_data.keys())}")
print(f"  results.economy_data keys: {list(results.economy_data.keys())}")

# The unified data property
all_data = results.data
print(f"\nresults.data keys: {list(all_data.keys())}")

# %%
# Visualizing Results
# -------------------
#
# Plot key economic indicators over time.

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Unemployment rate
ax1 = axes[0, 0]
if len(unemployment) > 0:
    ax1.plot(unemployment * 100, linewidth=2, color="tab:blue")
    ax1.set_ylabel("Unemployment Rate (%)")
    ax1.set_title("Unemployment")
    ax1.grid(True, alpha=0.3)

# Average price
ax2 = axes[0, 1]
if len(avg_price) > 0:
    ax2.plot(avg_price, linewidth=2, color="tab:green")
    ax2.set_ylabel("Price Level")
    ax2.set_title("Average Market Price")
    ax2.grid(True, alpha=0.3)

# Inflation
ax3 = axes[1, 0]
if len(inflation) > 0:
    ax3.plot(inflation * 100, linewidth=2, color="tab:orange")
    ax3.set_ylabel("Inflation Rate (%)")
    ax3.set_title("Annual Inflation")
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax3.grid(True, alpha=0.3)

# Production - compute mean across firms since data is 2D
ax4 = axes[1, 1]
production_data = results["Producer.production"]
# Average across firms (axis=1) since data is (periods, firms)
avg_production = np.mean(production_data, axis=1)
ax4.plot(avg_production, linewidth=2, color="tab:red")
ax4.set_ylabel("Avg Production")
ax4.set_title("Average Firm Production")
ax4.grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel("Period")

plt.tight_layout()
plt.show()

# %%
# Custom Data Collection
# ----------------------
#
# Use a dictionary to specify exactly what data to collect.
# Keys are role names (or "Economy"), values are ``True`` for all variables
# or a list of specific variable names.

# Collect only specific roles (economy metrics are always included automatically)
custom_results = bam.Simulation.init(n_firms=100, n_households=500, seed=42).run(
    n_periods=30,
    collect={
        "Producer": True,  # All Producer variables
        "Worker": True,  # All Worker variables
        "aggregate": "mean",  # Average across agents
    },
)

print("Custom collection:")
print(f"  Roles collected: {list(custom_results.role_data.keys())}")

# %%
# Full Agent-Level Data
# ---------------------
#
# Dict-form ``collect`` returns full per-agent data by default (larger arrays).

# Warning: This collects full arrays - can be memory intensive!
full_results = bam.Simulation.init(n_firms=50, n_households=250, seed=42).run(
    n_periods=20,
    collect={
        "Producer": ["price", "production"],  # Specific variables only
    },
)

prices_full = full_results["Producer.price"]
print(f"Full price data shape: {prices_full.shape}")
print(f"  (periods x firms): ({prices_full.shape[0]} x {prices_full.shape[1]})")

# Access individual firm's price history
firm_0_prices = prices_full[:, 0]
print(f"Firm 0 price history: {firm_0_prices[:5].round(3)}...")

# %%
# Export to pandas DataFrame
# --------------------------
#
# Convert results to pandas DataFrames for further analysis.
# Note: pandas is an optional dependency.

try:
    import pandas

    # Run a new simulation for DataFrame export
    df_results = bam.Simulation.init(n_firms=100, n_households=500, seed=42).run(
        n_periods=50,
        collect=True,
    )

    # Get all data as a single DataFrame (aggregated)
    df = df_results.to_dataframe(aggregate="mean")
    print("Full DataFrame:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)[:5]}...")

    # Get economy metrics only
    df_economy = df_results.economy_metrics
    print("\nEconomy metrics DataFrame:")
    print(df_economy.head())

    # Get specific role data
    df_producer = df_results.get_role_data("Producer", aggregate="mean")
    print(f"\nProducer DataFrame columns: {list(df_producer.columns)}")

except ImportError:
    print("pandas not installed. Install with: pip install pandas")

# %%
# Summary Statistics
# ------------------
#
# Get descriptive statistics for all collected metrics.

try:
    import pandas  # noqa: F401 - check if pandas is installed

    summary_results = bam.Simulation.init(n_firms=100, n_households=500, seed=42).run(
        n_periods=100,
        collect=True,
    )

    # Get summary statistics
    summary = summary_results.summary
    print("Summary Statistics:")
    print(summary[["mean", "std", "min", "max"]].round(4))

except ImportError:
    print("pandas not installed for summary statistics")

# %%
# Comparing Multiple Simulation Runs
# ----------------------------------
#
# Run multiple simulations and compare their results.

# Run with different random seeds
n_runs = 5
all_unemployment = []

print("Running ensemble of simulations...")
for i in range(n_runs):
    run_results = bam.Simulation.init(n_firms=100, n_households=500, seed=42 + i).run(
        n_periods=50,
        collect={
            "Worker": ["employed"],
            "capture_timing": {"Worker.employed": "firms_run_production"},
        },
    )
    # Calculate unemployment from Worker employed data
    unemp_rate = calc_unemployment_rate(run_results["Worker.employed"])
    all_unemployment.append(unemp_rate)

# Plot ensemble results
if all_unemployment:
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, unemp in enumerate(all_unemployment):
        ax.plot(bam.ops.multiply(unemp, 100), alpha=0.5, label=f"Run {i + 1}")

    # Calculate and plot mean unemployment across runs
    # Stack arrays and compute mean along axis 0
    stacked = bam.ops.asarray([list(u) for u in all_unemployment])
    mean_unemployment = bam.ops.multiply(bam.ops.mean(stacked, axis=0), 100)
    ax.plot(mean_unemployment, "k-", linewidth=2, label="Mean")

    ax.set_xlabel("Period")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.set_title(f"Ensemble of {n_runs} Simulation Runs")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Summary across runs
    final_rates = bam.ops.asarray([unemp[-1] * 100 for unemp in all_unemployment])
    print("\nFinal unemployment rates:")
    print(f"  Mean: {bam.ops.mean(final_rates):.2f}%")
    print(f"  Std:  {bam.ops.std(final_rates):.2f}%")
    print(f"  Range: {bam.ops.min(final_rates):.2f}% - {bam.ops.max(final_rates):.2f}%")

# %%
# Collecting Relationship Data
# ----------------------------
#
# Relationships (like ``LoanBook``) can be collected alongside role data.
# Unlike roles, relationships are **opt-in only** and NOT included with ``collect=True``.

# Manually add some loans to demonstrate relationship data collection
rel_sim = bam.Simulation.init(n_firms=50, n_households=250, seed=42)
loans = rel_sim.get_relationship("LoanBook")
loans.append_loans_for_lender(
    lender_idx=np.intp(0),
    borrower_indices=np.array([0, 1, 2, 3, 4], dtype=np.int64),
    amount=np.array([1000.0, 1500.0, 2000.0, 500.0, 750.0]),
    rate=np.array([0.02, 0.03, 0.025, 0.018, 0.022]),
)

rel_results = rel_sim.run(
    n_periods=20,
    collect={
        "Producer": ["price"],  # Role data
        "LoanBook": ["principal", "rate", "debt"],  # Relationship data
        "aggregate": "sum",  # Sum across all active loans
    },
)

print("Relationship data collection:")
print(f"  Available: {rel_results.available()}")

# Access via bracket notation
total_principal = rel_results["LoanBook.principal"]
print(f"  Total principal over time shape: {total_principal.shape}")
print(f"  Initial total principal: {total_principal[0]:.2f}")

# Access via get()
total_debt = rel_results.get("LoanBook", "debt")
print(f"  Total debt (last period): {total_debt[-1]:.2f}")

# %%
# Analyzing Loan Distribution
# ---------------------------
#
# Without aggregation (the default for dict-form collect), you get full edge data per
# period as variable-length arrays. Useful for analyzing distributions but cannot be
# exported to DataFrame.

loan_dist_sim = bam.Simulation.init(n_firms=50, n_households=250, seed=42)
loans = loan_dist_sim.get_relationship("LoanBook")
# Add loans with varying amounts
loans.append_loans_for_lender(
    lender_idx=np.intp(0),
    borrower_indices=np.array([0, 1, 2], dtype=np.int64),
    amount=np.array([100.0, 200.0, 300.0]),
    rate=np.array([0.02, 0.03, 0.025]),
)

dist_results = loan_dist_sim.run(
    n_periods=5,
    collect={
        "LoanBook": ["principal"],  # Full edge data (variable-length per period)
    },
)

principal_per_period = dist_results["LoanBook.principal"]
print("\nLoan distribution (full per-edge data):")
print(f"  Type: {type(principal_per_period).__name__}")
print(f"  Number of periods: {len(principal_per_period)}")
if principal_per_period:
    print(f"  Period 0 loans: {len(principal_per_period[0])} active")
    print(f"  Period 0 principals: {principal_per_period[0]}")

# %%
# Key Takeaways
# -------------
#
# **Basic collection:**
#
# - Use ``collect=True`` (the default) for full per-agent data collection
# - Economy metrics are always collected automatically
# - Use ``collect={"Producer": ["price"]}`` for specific variables
# - Use ``True`` for all variables: ``{"Worker": True}``
#
# **Data access (primary API):**
#
# - ``results["Producer.price"]`` -- bracket notation (recommended)
# - ``results.Producer.price`` -- attribute access (interactive use)
# - ``results.get("Producer", "price")`` -- programmatic access
# - ``results.get("Producer", "price", aggregate="mean")`` -- with aggregation
# - ``results.available()`` -- discover all collected variables
#
# **Per-agent data and capture timing:**
#
# - Dict-form ``collect`` returns full per-agent data by default (shape: periods x agents)
# - Use ``capture_timing`` to control when variables are captured during each period
# - Example: ``"capture_timing": {"Worker.employed": "firms_run_production"}``
#
# **Computing derived metrics:**
#
# - Unemployment rate can be computed from ``Worker.employed``:
#   ``1.0 - np.mean(employed.astype(float), axis=1)``
# - When computing from per-agent data, aggregate across agents with ``axis=1``
#
# **Relationship data:**
#
# - Relationships (like ``LoanBook``) are opt-in: use ``"LoanBook": ["principal"]``
# - NOT included with ``collect=True`` (must specify explicitly)
# - Aggregations: ``sum`` (total), ``mean`` (average), ``std`` (variation)
# - Without aggregation (default): list of variable-length arrays (can't export to DataFrame)
# - Access via ``results["LoanBook.principal"]`` or ``results.get("LoanBook", "principal")``
#
# **Legacy access (still works):**
#
# - ``results.economy_data``, ``results.role_data``, ``results.relationship_data``
# - ``results.data`` for unified dict access (includes "Economy" key and relationships)
# - Export to pandas with ``to_dataframe()`` for further analysis
# - Get quick statistics with ``results.summary``
