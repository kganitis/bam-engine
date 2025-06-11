# example/run.py
"""
Runs a baseline simulation of the BAM Engine and plots key macroeconomic time series.
"""

import logging

import numpy as np

from bamengine.helpers import sample_beta_with_mean
from bamengine.scheduler import Scheduler
from data_collector import DataCollector
from plotting import plot_results

log = logging.getLogger(__name__)
logging.getLogger("bamengine").setLevel(logging.WARNING)


def run_baseline_simulation(n_firms=100, seed=0):
    # --- Simulation Parameters ---
    params = {
        "n_households": n_firms * 5,
        "n_firms": n_firms,
        "n_banks": max(int(n_firms / 10), 3),
        "periods": 500,
        "seed": np.random.default_rng(seed)
    }

    # params["savings_init"] = sample_beta_with_mean(
    #     mean=3.0,
    #     n=params["n_households"],
    #     low=0.5, high=10.0,
    #     concentration=12,
    #     rng=params["seed"]
    # )
    params["savings_init"] = 1 + params["seed"].poisson(2)

    # params["equity_base_init"] = sample_beta_with_mean(
    #     mean=5.0,
    #     n=params["n_banks"],
    #     low=1.0,
    #     high=10.0,
    #     concentration=20,
    #     rng=params["seed"]
    # )
    params["equity_base_init"] = 10 + params["seed"].poisson(10_000)

    sched = Scheduler.init(
        config="config.yml",
        n_firms=params["n_firms"],
        n_households=params["n_households"],
        n_banks=params["n_banks"],
        n_periods=params["periods"],
        savings_init=params["savings_init"],
        equity_base_init=params["equity_base_init"],
        seed=params["seed"],
    )
    collector = DataCollector()

    for _ in range(sched.n_periods):
        if not sched.ec.destroyed:
            log.info(
                f"--> Simulating period {sched.t + 1}/{sched.n_periods} "
                f"---------------------------------------------------------------"
                f"---------------------------------------------------------------"
            )
            sched.step()
            collector.capture(sched)

    return collector.get_arrays()


if __name__ == "__main__":
    sim_data = run_baseline_simulation()
    plot_results(data=sim_data, burn_in=100)
