"""
=======
Logging
=======

This example demonstrates BAM Engine's logging system. Learn how to:

- Configure global log levels
- Set per-event log levels
- Use the custom TRACE level for verbose debugging
- Create loggers in your custom code

Proper logging configuration helps with debugging, performance analysis,
and understanding simulation behavior.
"""

# %%
# Default Logging Behavior
# ------------------------
#
# By default, BAM Engine logs at INFO level. This provides a good balance
# between information and noise.

import bamengine as bam
from bamengine import logging

print("Default log level is INFO")
print("Running a short simulation with default logging...")

sim = bam.Simulation.init(n_firms=50, n_households=250, seed=42)
sim.run(n_periods=5)

print(f"Simulation complete. Final unemployment: {sim.ec.unemp_rate_history[-1]:.2%}")

# %%
# Log Levels Overview
# -------------------
#
# BAM Engine supports standard Python log levels plus a custom TRACE level:
#
# - CRITICAL (50): Critical errors that may cause failure
# - ERROR (40): Errors that affect operation
# - WARNING (30): Warnings about potential issues
# - INFO (20): General information (default)
# - DEBUG (10): Detailed debugging information
# - TRACE (5): Very verbose output for deep debugging

print("\nAvailable log levels:")
print(f"  CRITICAL: {logging.CRITICAL}")
print(f"  ERROR:    {logging.ERROR}")
print(f"  WARNING:  {logging.WARNING}")
print(f"  INFO:     {logging.INFO}")
print(f"  DEBUG:    {logging.DEBUG}")
print(f"  TRACE:    {logging.TRACE}")

# %%
# Setting Global Log Level
# ------------------------
#
# Configure logging via the ``logging`` parameter at initialization.

# Set to WARNING to reduce output (only warnings and above)
sim_quiet = bam.Simulation.init(
    n_firms=50,
    n_households=250,
    seed=42,
    logging={"default_level": "WARNING"},
)

print("\nRunning with WARNING level (quiet mode)...")
sim_quiet.run(n_periods=5)
print("Done (you should see fewer log messages)")

# Set to DEBUG for more detail
sim_debug = bam.Simulation.init(
    n_firms=50,
    n_households=250,
    seed=42,
    logging={"default_level": "DEBUG"},
)

print("\nRunning with DEBUG level (verbose mode)...")
sim_debug.run(n_periods=2)
print("Done (you should see more log messages)")

# %%
# Per-Event Log Levels
# --------------------
#
# Fine-tune logging by setting different levels for specific events.
# This is useful when debugging a particular part of the simulation.

# Only show detailed output for labor market events
log_config = {
    "default_level": "WARNING",  # Base level (quiet)
    "events": {
        "workers_send_one_round": "DEBUG",  # Verbose for this event
        "firms_hire_workers": "DEBUG",  # And this one
    },
}

sim_selective = bam.Simulation.init(
    n_firms=50,
    n_households=250,
    seed=42,
    logging=log_config,
)

print("\nRunning with selective logging (labor market events only)...")
sim_selective.run(n_periods=3)

# %%
# Using TRACE Level
# -----------------
#
# TRACE (level 5) provides the most verbose output. Use it for
# deep debugging when you need to understand exactly what's happening.

# TRACE level - very verbose!
sim_trace = bam.Simulation.init(
    n_firms=20,  # Small simulation
    n_households=100,
    seed=42,
    logging={
        "default_level": "INFO",
        "events": {
            "firms_decide_desired_production": "TRACE",
        },
    },
)

print("\nRunning with TRACE level for production planning...")
sim_trace.run(n_periods=1)
print("Check the output for very detailed production planning info")

# %%
# Creating Custom Loggers
# -----------------------
#
# When writing custom events or extensions, create your own loggers.

# Get a logger for your module
logger = logging.getLogger("my_custom_module")

# Log at different levels
logger.info("This is an info message")
logger.debug("This is a debug message (may not show at INFO level)")
logger.warning("This is a warning")

# The custom TRACE method
logger.trace("This is a trace message (very verbose)")

print("\n(Custom logger messages shown above if level allows)")

# %%
# Conditional Logging
# -------------------
#
# For performance-sensitive code, check if logging is enabled before
# computing expensive debug information.


def expensive_computation():
    """Simulate an expensive debugging calculation."""
    rng = bam.make_rng()
    arr = bam.ops.uniform(rng, 0, 1, 1000000)
    return bam.ops.mean(arr)


# Create a logger
perf_logger = logging.getLogger("performance_example")
perf_logger.setLevel(logging.DEBUG)

# Good practice: check before expensive operations
if perf_logger.isEnabledFor(logging.DEBUG):
    result = expensive_computation()
    perf_logger.debug(f"Expensive result: {result:.6f}")

# This is more efficient than:
# perf_logger.debug(f"Expensive result: {expensive_computation()}")
# Because expensive_computation() would run even if DEBUG is disabled

print("\nConditional logging demonstrated (check code for pattern)")

# %%
# Silencing Verbose Events
# ------------------------
#
# Some events run many times per period (like market matching rounds).
# Silence them to focus on other parts of the simulation.

# Silence repetitive events
quiet_config = {
    "default_level": "INFO",
    "events": {
        # Market matching events run multiple times per period
        "workers_send_one_round": "WARNING",
        "firms_hire_workers": "WARNING",
        "firms_send_one_loan_app": "WARNING",
        "banks_provide_loans": "WARNING",
        "consumers_shop_one_round": "WARNING",
    },
}

sim_focused = bam.Simulation.init(
    n_firms=50,
    n_households=250,
    seed=42,
    logging=quiet_config,
)

print("\nRunning with market matching events silenced...")
sim_focused.run(n_periods=5)
print("Done (market matching events suppressed)")

# %%
# Logging in Custom Events
# ------------------------
#
# When creating custom events, use the built-in logger pattern.

from bamengine import event, ops


@event
class TaxCollectionEvent:
    """Custom event that collects taxes from firms."""

    def execute(self, sim):
        # Get a logger for this event
        logger = logging.getLogger("bamengine.events.tax_collection")

        logger.info("Collecting taxes from firms...")

        # Access firm data
        borr = sim.get_role("Borrower")

        # Calculate tax (10% of net worth)
        tax_rate = 0.10
        positive_nw = ops.maximum(borr.net_worth, 0.0)
        tax_amount = ops.multiply(positive_nw, tax_rate)

        # Log details at debug level
        total_tax = ops.sum(tax_amount)
        logger.debug(f"Total tax collected: {total_tax:.2f}")

        # Very detailed logging at trace level
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(f"Individual tax amounts: {tax_amount[:5]}... (first 5)")

        # Apply tax (reduce net worth)
        borr.net_worth[:] = borr.net_worth - tax_amount

        logger.info(f"Tax collection complete. Total: {total_tax:.2f}")


print("\nCustom TaxCollectionEvent defined (see code for logging pattern)")

# %%
# YAML Logging Configuration
# --------------------------
#
# Logging can also be configured in YAML config files:
#
# .. code-block:: yaml
#
#     # In your config.yml file
#     logging:
#       default_level: INFO
#       events:
#         workers_send_one_round: WARNING
#         firms_hire_workers: DEBUG
#         firms_adjust_price: TRACE
#
# Then load it:
#
# .. code-block:: python
#
#     sim = bam.Simulation.init(config="config.yml")

print(
    """
YAML logging configuration example:

logging:
  default_level: INFO
  events:
    workers_send_one_round: WARNING
    firms_hire_workers: DEBUG
    firms_adjust_price: TRACE
"""
)

# %%
# Key Takeaways
# -------------
#
# - Default level is INFO (balanced output)
# - Use WARNING or ERROR for quiet runs
# - Use DEBUG or TRACE for debugging
# - Configure per-event levels to focus on specific areas
# - Use ``isEnabledFor()`` before expensive debug computations
# - Custom events should use ``logging.getLogger()``
