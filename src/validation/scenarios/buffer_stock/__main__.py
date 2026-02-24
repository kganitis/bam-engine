"""Entry point for: python -m validation.scenarios.buffer_stock"""

from validation.scenarios.buffer_stock import run_scenario

run_scenario(seed=0, n_periods=1000, burn_in=500, show_plot=True)
