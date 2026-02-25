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
sim = bam.Simulation.init()

# Run for 50 periods
sim.run(n_periods=50)

print(f"Simulation completed! Ran {sim.t} periods.")

# %%
# Visualize Unemployment
# ----------------------
#
# Plot the average market price over time.

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(sim.ec.avg_mkt_price_history, linewidth=2)
plt.xlabel("Period")
plt.ylabel("Average Market Price")
plt.title("Average Market Price Over Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
