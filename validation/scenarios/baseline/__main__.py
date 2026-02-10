"""Entry point for: python -m validation.scenarios.baseline"""

from validation.scenarios.baseline import run_scenario

run_scenario(seed=0, n_periods=1000, burn_in=500, show_plot=True)
