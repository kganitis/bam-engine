"""Entry point for: python -m validation.scenarios.growth_plus"""

from validation.scenarios.growth_plus import run_scenario

run_scenario(seed=0, n_periods=1000, burn_in=500, show_plot=True)
