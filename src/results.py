# src/results.py
"""
Runs a baseline simulation of the BAM Engine and plots key macroeconomic time series.
"""

import logging
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from bamengine.scheduler import Scheduler
from diagnostics import log_firm_strategy_distribution

logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class DataCollector:
    """A simple class to collect and store time-series data from the simulation."""

    # Raw data collected each period
    gdp: list[float] = field(default_factory=list)
    unemployment_rate: list[float] = field(default_factory=list)
    inflation_rate: list[float] = field(default_factory=list)
    productivity_wage_ratio: list[float] = field(default_factory=list)

    def capture(self, sched: Scheduler) -> None:
        """Calculates and stores metrics for the current period."""

        # --- Real GDP ---
        current_real_gdp = float(sched.prod.production.sum() / sched.ec.avg_mkt_price)
        self.gdp.append(current_real_gdp)

        # --- Unemployment Rate ---
        current_unemployment = float(sched.ec.unemp_rate_history[-1])
        self.unemployment_rate.append(current_unemployment)

        # --- Annual Inflation Rate ---
        current_annual_inflation = float(sched.ec.inflation_history[-1])
        self.inflation_rate.append(current_annual_inflation)

        # --- Productivity to Real Wage Ratio ---
        # (Avg Labor Productivity) / (Avg Nominal Wage / Price Level)
        avg_productivity = float(sched.prod.labor_productivity.mean())
        # Avoid division by zero if prices or wages are zero
        if sched.ec.avg_mkt_price > 0:
            # Use only wages of employed workers for a more stable average
            employed_wages = sched.wrk.wage[sched.wrk.employed]
            avg_nominal_wage = (
                float(employed_wages.mean()) if len(employed_wages) > 0 else 0.0
            )
            avg_real_wage = avg_nominal_wage / sched.ec.avg_mkt_price
            ratio = avg_productivity / avg_real_wage if avg_real_wage > 0 else 0.0
        else:
            ratio = 0.0
        self.productivity_wage_ratio.append(ratio)

        log.debug(
            f"Real GDP={current_real_gdp:.2f}, "
            f"Unemployment={current_unemployment:.3f}, "
            f"Inflation={current_annual_inflation * 100:.2f}, "
            f"Prod / Real Wage={ratio:.2f}, "
        )

    def get_arrays(self) -> dict[str, NDArray[np.float64]]:
        """Converts collected lists to NumPy arrays for analysis."""
        return {
            "gdp": np.array(self.gdp),
            "unemployment_rate": np.array(self.unemployment_rate),
            "inflation_rate": np.array(self.inflation_rate),
            "productivity_wage_ratio": np.array(self.productivity_wage_ratio),
        }


def run_baseline_simulation() -> dict[str, NDArray[np.float64]]:
    """Initializes and runs the simulation, returning the collected data."""
    log.info("Initializing baseline simulation...")
    # --- Simulation Parameters ---
    n = 20
    params = {
        "n_households": n * 5,
        "n_firms": n,
        "n_banks": max(int(n / 10), 3),
        "periods": 20,
        "seed": 42,
    }

    sched = Scheduler.init(
        n_firms=params["n_firms"],
        n_households=params["n_households"],
        n_banks=params["n_banks"],
        periods=params["periods"],
        seed=params["seed"],
    )
    collector = DataCollector()

    for t in range(params["periods"]):
        log.info(
            f"\n\n--> Simulating period {t+1}/{params['periods']} "
            f"---------------------------------------------------------------"
            f"---------------------------------------------------------------"
            f"---------------------------------------------------------------"
            f"---------------------------------------------------------------\n"
        )
        log_firm_strategy_distribution(sched)
        sched.step()
        collector.capture(sched)

    log.info("Simulation finished.")
    return collector.get_arrays()


def plot_results(data: dict[str, NDArray[np.float64]], burn_in: int) -> None:
    """Processes the data and generates the 4-panel plot."""
    log.info(f"Generating plots, discarding first {burn_in} periods as transient.")

    # --- Prepare data for plotting ---
    periods = np.arange(burn_in, len(data["gdp"]))

    # Slice data to remove the burn-in period
    log_gdp = np.log(data["gdp"][burn_in:])
    unemployment = data["unemployment_rate"][burn_in:]
    annual_inflation = data["inflation_rate"][burn_in:] * 100  # as percentage
    prod_wage_ratio = data["productivity_wage_ratio"][burn_in:]

    # --- Create Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("BAM Model Baseline Scenario Results", fontsize=16)

    # 1. (log) Real GDP
    axes[0, 0].plot(periods, log_gdp)
    axes[0, 0].set_title("Log Real GDP")
    axes[0, 0].set_xlabel("Time (periods)")
    axes[0, 0].set_ylabel("Log Output")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    # 2. Unemployment Rate
    axes[0, 1].plot(periods, unemployment)
    axes[0, 1].set_title("Unemployment Rate")
    axes[0, 1].set_xlabel("Time (periods)")
    axes[0, 1].set_ylabel("Unemployment Rate")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    # 3. Annual Inflation Rate
    axes[1, 0].plot(periods, annual_inflation)
    axes[1, 0].set_title("Annual Inflation Rate (%)")
    axes[1, 0].set_xlabel("Time (periods)")
    axes[1, 0].set_ylabel("Inflation Rate (%)")
    axes[1, 0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    # 4. Productivity / Real Wage Ratio
    axes[1, 1].plot(periods, prod_wage_ratio)
    axes[1, 1].set_title("Productivity / Real Wage Ratio")
    axes[1, 1].set_xlabel("Time (periods)")
    axes[1, 1].set_ylabel("Ratio")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    plt.show()


if __name__ == "__main__":
    log.setLevel(logging.DEBUG)

    sim_data = run_baseline_simulation()
    plot_results(data=sim_data, burn_in=0)
