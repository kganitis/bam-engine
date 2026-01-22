"""
==================
Simulation Results
==================

This example demonstrates how to collect and analyze simulation results
using the ``SimulationResults`` class. Learn how to:

- Collect data during simulation runs
- Access role and economy time series
- Use ``get_array()`` for cleaner data access
- Use the ``data`` property for unified access
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
        "Economy": True,
        "aggregate": None,  # Full per-agent data for Worker
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
# Accessing Economy Metrics
# -------------------------
#
# Economy-wide metrics are stored as arrays in ``economy_data``.
# We can also compute derived metrics from role data.

print("Available economy metrics:", list(results.economy_data.keys()))


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


# Access specific metrics
avg_price = results.economy_data.get("avg_price", bam.ops.zeros(0))
inflation = results.economy_data.get("inflation", bam.ops.zeros(0))

# Calculate unemployment rate from Worker employed data
worker_employed = results.role_data.get("Worker", {}).get("employed", None)
if worker_employed is not None:
    unemployment = calc_unemployment_rate(worker_employed)
    print("\nUnemployment rate (calculated from Worker employed data):")
    print(f"  Mean: {bam.ops.mean(unemployment):.2%}")
    print(f"  Final: {unemployment[-1]:.2%}")
else:
    unemployment = bam.ops.zeros(0)

if len(avg_price) > 0:
    print("\nAverage price:")
    print(f"  Mean: {bam.ops.mean(avg_price):.3f}")
    print(f"  Final: {avg_price[-1]:.3f}")

# %%
# Accessing Role Data
# -------------------
#
# Role data contains per-period snapshots of agent states.
# When ``aggregate=None``, data is full per-agent arrays (periods x agents).

print("Available roles:", list(results.role_data.keys()))

# Access Producer role data
if "Producer" in results.role_data:
    producer_data = results.role_data["Producer"]
    print(f"\nProducer variables: {list(producer_data.keys())}")

    # Get price data - shape is (periods, firms) with aggregate=None
    if "price" in producer_data:
        price_data = producer_data["price"]
        print(f"  Price data shape: {price_data.shape}")
        # Calculate mean price per period
        avg_prices = np.mean(price_data, axis=1)
        print(f"  Price trend (first 5): {avg_prices[:5].round(3)}")

# %%
# Easy Data Access with get_array()
# ---------------------------------
#
# Use ``get_array()`` for cleaner data access without navigating nested dicts.
# This method also supports aggregation on-the-fly.

# get_array() for role data - cleaner than results.role_data["Producer"]["price"]
if "Producer" in results.role_data and "price" in results.role_data["Producer"]:
    prices_via_get_array = results.get_array("Producer", "price")
    print("\nUsing get_array():")
    print(
        f"  results.get_array('Producer', 'price').shape: {prices_via_get_array.shape}"
    )

# get_array() for economy data - use "Economy" as the role name
if len(avg_price) > 0:
    price_via_get_array = results.get_array("Economy", "avg_price")
    print(
        f"  results.get_array('Economy', 'avg_price').shape: {price_via_get_array.shape}"
    )

# Aggregation on-the-fly (useful when you have full per-agent data)
full_data_sim = bam.Simulation.init(n_firms=50, n_households=250, seed=42)
full_data_results = full_data_sim.run(
    n_periods=20,
    collect={
        "Producer": ["price"],  # Specific variables for Producer
        "aggregate": None,  # Full per-agent data
    },
)

if "Producer" in full_data_results.role_data:
    # Get full 2D data (periods x firms)
    prices_2d = full_data_results.get_array("Producer", "price")
    print(f"\n  Full data shape: {prices_2d.shape}")

    # Get mean aggregated on-the-fly
    prices_mean = full_data_results.get_array("Producer", "price", aggregate="mean")
    print(f"  With aggregate='mean': {prices_mean.shape}")

# %%
# Unified Data Access with data Property
# --------------------------------------
#
# The ``data`` property combines role_data and economy_data into one dict.
# Economy data is accessible under the "Economy" key.

print("\nUsing results.data property:")
all_data = results.data
print(f"  Available keys: {list(all_data.keys())}")

# Access role data
if "Producer" in all_data:
    print(f"  Producer variables: {list(all_data['Producer'].keys())}")

# Access economy data via "Economy" key
if "Economy" in all_data:
    print(f"  Economy variables: {list(all_data['Economy'].keys())}")

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

# Production (if available) - compute mean across firms since data is 2D
ax4 = axes[1, 1]
if "Producer" in results.role_data and "production" in results.role_data["Producer"]:
    production_data = results.role_data["Producer"]["production"]
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

# Collect only specific roles and economy metrics
custom_results = bam.Simulation.init(n_firms=100, n_households=500, seed=42).run(
    n_periods=30,
    collect={
        "Producer": True,  # All Producer variables
        "Worker": True,  # All Worker variables
        "Economy": True,  # All economy metrics
        "aggregate": "mean",  # Average across agents
    },
)

print("Custom collection:")
print(f"  Roles collected: {list(custom_results.role_data.keys())}")

# %%
# Full Agent-Level Data
# ---------------------
#
# Set ``aggregate=None`` to get per-agent data (larger arrays).

# Warning: This collects full arrays - can be memory intensive!
full_results = bam.Simulation.init(n_firms=50, n_households=250, seed=42).run(
    n_periods=20,
    collect={
        "Producer": ["price", "production"],  # Specific variables only
        "aggregate": None,  # Full per-agent data
    },
)

if "Producer" in full_results.role_data:
    producer_full = full_results.role_data["Producer"]
    if "price" in producer_full:
        prices_full = producer_full["price"]
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
            "aggregate": None,
            "capture_timing": {"Worker.employed": "firms_run_production"},
        },
    )
    # Calculate unemployment from Worker employed data
    worker_employed = run_results.role_data.get("Worker", {}).get("employed", None)
    if worker_employed is not None:
        unemp_rate = calc_unemployment_rate(worker_employed)
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
        "Economy": True,
        "aggregate": "sum",  # Sum across all active loans
    },
)

print("Relationship data collection:")
print(f"  Relationships collected: {list(rel_results.relationship_data.keys())}")
if "LoanBook" in rel_results.relationship_data:
    loanbook = rel_results.relationship_data["LoanBook"]
    print(f"  LoanBook fields: {list(loanbook.keys())}")
    print(f"  Total principal over time shape: {loanbook['principal'].shape}")
    print(f"  Initial total principal: {loanbook['principal'][0]:.2f}")

# Access via get_array()
if "LoanBook" in rel_results.relationship_data:
    total_debt = rel_results.get_array("LoanBook", "debt")
    print(f"  Total debt (last period): {total_debt[-1]:.2f}")

# %%
# Analyzing Loan Distribution
# ---------------------------
#
# With ``aggregate=None``, you get full edge data per period as variable-length arrays.
# This is useful for analyzing loan distributions but cannot be exported to DataFrame.

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
        "LoanBook": ["principal"],
        "aggregate": None,  # Full edge data (variable-length per period)
    },
)

if "LoanBook" in dist_results.relationship_data:
    principal_per_period = dist_results.relationship_data["LoanBook"]["principal"]
    print("\nLoan distribution (aggregate=None):")
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
# - Use ``collect=True`` for basic data collection with aggregated means
# - Use ``collect=["Producer", "Worker", "Economy"]`` for specific roles
# - Use ``collect={"Producer": ["price"], "Economy": True}`` for specific variables
# - Use ``True`` for all variables: ``{"Worker": True}``
#
# **Per-agent data and capture timing:**
#
# - Use ``aggregate=None`` to get full per-agent data (shape: periods × agents)
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
# - With ``aggregate=None``: list of variable-length arrays (can't export to DataFrame)
# - Access via ``results.relationship_data["LoanBook"]["principal"]``
# - Or via ``results.get_array("LoanBook", "principal")``
#
# **Data access:**
#
# - Access raw data via ``results.economy_data``, ``results.role_data``, and
#   ``results.relationship_data``
# - Use ``results.get_array()`` for cleaner access: ``get_array("Producer", "price")``
# - Use ``results.get_array("Economy", "metric")`` for economy data
# - Use ``results.get_array("LoanBook", "field")`` for relationship data
# - Use ``results.data`` for unified access (includes "Economy" key and relationships)
# - Export to pandas with ``to_dataframe()`` for further analysis
# - Get quick statistics with ``results.summary``
