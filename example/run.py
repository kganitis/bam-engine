# example/run.py
"""
Runs a baseline simulation of the BAM Engine and plots key macroeconomic time series.
"""

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from bamengine.helpers import sample_beta_with_mean
from bamengine.scheduler import Scheduler
from diagnostics import log_firm_strategy_distribution
from example.data_collector import DataCollector
from example.plotting import plot_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_baseline_simulation(n=100, seed=42) -> dict[str, NDArray[np.float64]]:
    # --- Simulation Parameters ---
    params = {
        "n_households": n * 5,
        "n_firms": n,
        "n_banks": max(int(n / 10), 3),
        "periods": 1000,
        "seed": np.random.default_rng(seed)
    }

    savings_seed = sample_beta_with_mean(
        mean=3.0,
        n=params["n_households"],
        low=2.0, high=10.0,
        concentration=6,
        rng=params["seed"])

    equity_base_seed = sample_beta_with_mean(
        mean=10_000.0,
        n=params["n_banks"],
        low=5_000.0,
        high=15_000.0,
        concentration=20,
        rng=params["seed"])

    # Broadcast scalar/array to the correct length.
    params["savings_init"] = np.broadcast_to(savings_seed,
                                             params["n_households"]).copy()
    params["equity_base_init"] = np.broadcast_to(equity_base_seed,
                                                 params["n_banks"]).copy()

    sched = Scheduler.init(
        Path("config.yml"),
        n_firms=params["n_firms"],
        n_households=params["n_households"],
        n_banks=params["n_banks"],
        n_periods=params["periods"],
        savings_init=params["savings_init"],
        equity_base_init=params["equity_base_init"],
        seed=params["seed"],
    )
    collector = DataCollector()

    for t in range(params["periods"]):
        log.info(
            f"--> Simulating period {t + 1}/{params['periods']} "
            f"---------------------------------------------------------------"
            f"---------------------------------------------------------------"
        )
        log_firm_strategy_distribution(sched)
        sched.step()
        collector.capture(sched)

    return collector.get_arrays()


if __name__ == "__main__":
    log.setLevel(logging.INFO)

    sim_data = run_baseline_simulation()
    plot_results(data=sim_data, burn_in=500)
