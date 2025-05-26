# src/bamengine/results.py
"""
Runs a baseline simulation of the BAM Engine and plots key macroeconomic time series.
"""

import logging
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from bamengine.scheduler import Scheduler

# --- Basic logging setup ---
logging.basicConfig(
    level=logging.INFO,
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
    avg_price_level: list[float] = field(default_factory=list)
    productivity_wage_ratio: list[float] = field(default_factory=list)

    def capture(self, sched: Scheduler) -> None:
        """Calculates and stores metrics for the current period."""
        # --- Real GDP ---
        # Sum of all firm production in the period
        current_gdp = float(sched.prod.production.sum())
        self.gdp.append(current_gdp)

        # --- Unemployment Rate ---
        # Number of unemployed agents / total household population
        unemployed_count = sched.n_households - sched.wrk.employed.sum()
        current_unemployment = unemployed_count / sched.n_households
        self.unemployment_rate.append(current_unemployment)

        # --- Average Price Level ---
        # Use the average market price calculated by the engine
        self.avg_price_level.append(sched.ec.avg_mkt_price)

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
            self.productivity_wage_ratio.append(ratio)
        else:
            self.productivity_wage_ratio.append(0.0)

        log.debug(
            f"GDP={current_gdp:.2f}, "
            f"Unemployment={current_unemployment:.3f}, "
            f"P_avg={sched.ec.avg_mkt_price:.2f}"
        )

    def get_arrays(self) -> dict[str, NDArray[np.float64]]:
        """Converts collected lists to NumPy arrays for analysis."""
        return {
            "gdp": np.array(self.gdp),
            "unemployment_rate": np.array(self.unemployment_rate),
            "avg_price_level": np.array(self.avg_price_level),
            "productivity_wage_ratio": np.array(self.productivity_wage_ratio),
        }


def run_baseline_simulation() -> dict[str, NDArray[np.float64]]:
    """Initializes and runs the simulation, returning the collected data."""
    log.info("Initializing baseline simulation...")
    # --- Simulation Parameters ---
    params = {
        "n_households": 500,
        "n_firms": 100,
        "n_banks": 10,
        "periods": 100,
        "seed": 42,  # for reproducibility
    }

    sched = Scheduler.init(**params)
    collector = DataCollector()

    log.info(
        f"Starting simulation for {params['periods']} periods. "
        f"This may take a moment..."
    )
    for t in range(params["periods"]):
        if t % 100 == 0:
            log.info(f"--> Simulating period {t}/{params['periods']}")
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
    prod_wage_ratio = data["productivity_wage_ratio"][burn_in:]

    # Calculate annual inflation (assuming 4 periods = 1 year, e.g., quarterly)
    # Inflation at t = (P_t / P_{t-4}) - 1
    price_level = data["avg_price_level"]
    # Ensure we have enough data points to look back 4 periods
    inflation = np.zeros_like(price_level)
    for t in range(4, len(price_level)):
        if price_level[t - 4] > 0:
            inflation[t] = (price_level[t] / price_level[t - 4]) - 1
        else:
            inflation[t] = 0.0
    annual_inflation = inflation[burn_in:] * 100  # As percentage

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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # To see debug messages, change level to logging.DEBUG
    log.setLevel(logging.DEBUG)

    sim_data = run_baseline_simulation()
    plot_results(data=sim_data, burn_in=0)
