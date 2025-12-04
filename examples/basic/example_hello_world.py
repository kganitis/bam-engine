"""
================
Hello BAM Engine
================

This is the simplest possible BAM Engine example. It shows how to initialize
a simulation, run it for a few periods, and display basic results.

If you're new to BAM Engine, start here!
"""

# %%
# Initialize and Run
# ------------------
#
# Create a simulation with default parameters and run it for 50 periods.

import bamengine as bam

# Initialize simulation with 100 firms and 500 households
sim = bam.Simulation.init(n_firms=100, n_households=500, seed=42)

# Run for 50 periods
for _ in range(50):
    sim.step()

print(f"Simulation completed! Ran {sim.t} periods.")
print(f"Final unemployment rate: {sim.ec.unemp_rate_history[-1]:.2%}")
print(f"Final average price: {sim.ec.avg_mkt_price:.2f}")

# %%
# Visualize Unemployment
# ----------------------
#
# Plot the unemployment rate over time.

import matplotlib.pyplot as plt

# Get unemployment history (convert list to array and scale to percentage)
unemployment_history = bam.ops.multiply(bam.ops.asarray(sim.ec.unemp_rate_history), 100)

# Create simple plot
plt.figure(figsize=(10, 6))
plt.plot(unemployment_history, linewidth=2)
plt.xlabel("Period")
plt.ylabel("Unemployment Rate (%)")
plt.title("Unemployment Rate Over Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
