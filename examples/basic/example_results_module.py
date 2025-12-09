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

import bamengine as bam

# Run simulation with data collection
sim = bam.Simulation.init(n_firms=100, n_households=500, seed=42)
results = sim.run(n_periods=50, collect=True)

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

print("Available economy metrics:", list(results.economy_data.keys()))

# Access specific metrics (returns empty array if key not found)
empty_array = bam.ops.zeros(0)  # Fallback empty array
unemployment = results.economy_data.get("unemployment_rate", empty_array)
avg_price = results.economy_data.get("avg_price", empty_array)
inflation = results.economy_data.get("inflation", empty_array)

if len(unemployment) > 0:
    print("\nUnemployment rate:")
    print(f"  Mean: {bam.ops.mean(unemployment):.2%}")
    print(f"  Final: {unemployment[-1]:.2%}")

if len(avg_price) > 0:
    print("\nAverage price:")
    print(f"  Mean: {bam.ops.mean(avg_price):.3f}")
    print(f"  Final: {avg_price[-1]:.3f}")

# %%
# Accessing Role Data
# -------------------
#
# Role data contains per-period snapshots of agent states.
# By default, data is aggregated (mean across agents).

print("Available roles:", list(results.role_data.keys()))

# Access Producer role data
if "Producer" in results.role_data:
    producer_data = results.role_data["Producer"]
    print(f"\nProducer variables: {list(producer_data.keys())}")

    # Get average price over time (already aggregated by default)
    if "price" in producer_data:
        avg_prices = producer_data["price"]
        print(f"  Average price shape: {avg_prices.shape}")
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
if len(unemployment) > 0:
    unemp_via_get_array = results.get_array("Economy", "unemployment_rate")
    print(
        f"  results.get_array('Economy', 'unemployment_rate').shape: {unemp_via_get_array.shape}"
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

# Production (if available)
ax4 = axes[1, 1]
if "Producer" in results.role_data and "production" in results.role_data["Producer"]:
    production = results.role_data["Producer"]["production"]
    ax4.plot(production, linewidth=2, color="tab:red")
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
        collect=True,
    )
    if "unemployment_rate" in run_results.economy_data:
        all_unemployment.append(run_results.economy_data["unemployment_rate"])

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
# Key Takeaways
# -------------
#
# - Use ``collect=True`` for basic data collection with aggregated means
# - Use ``collect=["Producer", "Worker", "Economy"]`` for specific roles
# - Use ``collect={"Producer": ["price"], "Economy": True}`` for specific variables
# - Use ``True`` for all variables: ``{"Worker": True}``
# - Access raw data via ``results.economy_data`` and ``results.role_data``
# - Use ``results.get_array()`` for cleaner access: ``get_array("Producer", "price")``
# - Use ``results.get_array("Economy", "metric")`` for economy data
# - Use ``results.data`` for unified access (includes "Economy" key)
# - Export to pandas with ``to_dataframe()`` for further analysis
# - Get quick statistics with ``results.summary``
