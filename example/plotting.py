# example/plotting.py
import logging

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


logging.getLogger("matplotlib").setLevel(logging.INFO)


def plot_results(data: dict[str, NDArray[np.float64]], burn_in: int) -> None:
    periods = np.arange(burn_in, len(data["gdp"]))

    log_gdp = np.log(data["gdp"][burn_in:])
    unemployment = data["unemployment_rate"][burn_in:] * 100  # as percentage
    annual_inflation = data["inflation_rate"][burn_in:] * 100  # as percentage
    prod_wage_ratio = data["productivity_wage_ratio"][burn_in:]

    # Create Plots
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    fig.suptitle("BAM Model Baseline Scenario Results", fontsize=16)

    # (log) Real GDP
    axes[0].plot(periods, log_gdp)
    axes[0].set_title("Log Real GDP")
    axes[0].set_xlabel("Time (periods)")
    axes[0].set_ylabel("Log Output")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Unemployment Rate
    axes[1].plot(periods, unemployment)
    axes[1].set_title("Unemployment Rate (%)")
    axes[1].set_xlabel("Time (periods)")
    axes[1].set_ylabel("Unemployment Rate (%)")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Annual Inflation Rate
    axes[2].plot(periods, annual_inflation)
    axes[2].set_title("Annual Inflation Rate (%)")
    axes[2].set_xlabel("Time (periods)")
    axes[2].set_ylabel("Inflation Rate (%)")
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    # Productivity / Real Wage Ratio
    axes[3].plot(periods, prod_wage_ratio)
    axes[3].set_title("Productivity / Real Wage Ratio")
    axes[3].set_xlabel("Time (periods)")
    axes[3].set_ylabel("Ratio")
    axes[3].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    plt.show()
