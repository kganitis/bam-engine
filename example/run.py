# example/run.py
import logging

import numpy as np

from bamengine.simulation import Simulation
from data_collector import DataCollector
from plotting import plot_results

log = logging.getLogger(__name__)
logging.getLogger("bamengine").setLevel(logging.CRITICAL)


def run_example_simulation(n_firms=100, seed=0):
    params = {
        "n_households": n_firms * 5,
        "n_firms": n_firms,
        "n_banks": max(n_firms // 10, 3),
        "periods": 1000,
        "seed": np.random.default_rng(seed)
    }
    params["savings_init"] = 1 + params["seed"].poisson(2)
    params["equity_base_init"] = 10 + params["seed"].poisson(10_000)

    sim = Simulation.init(
        config="config.yml",
        n_firms=params["n_firms"],
        n_households=params["n_households"],
        n_banks=params["n_banks"],
        n_periods=params["periods"],
        savings_init=params["savings_init"],
        equity_base_init=params["equity_base_init"],
        seed=0,
    )
    collector = DataCollector()

    for _ in range(sim.n_periods):
        if not sim.ec.destroyed:
            log.info(
                f"--> Simulating period {sim.t + 1}/{sim.n_periods} "
                f"---------------------------------------------------------------"
            )
            sim.step()
            collector.capture(sim)

    return collector.get_arrays()


if __name__ == "__main__":
    sim_data = run_example_simulation()
    plot_results(data=sim_data, burn_in=500)
