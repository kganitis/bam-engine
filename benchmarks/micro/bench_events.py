"""Micro-benchmarks for individual events.

Tests performance of individual event executions to identify
which events are most expensive.
"""

import timeit
import numpy as np
import logging
from bamengine import Simulation

# Disable logging for benchmarks
logging.getLogger('bamengine').setLevel(logging.ERROR)


def bench_event(event_name, sim, n_warmup=10, n_runs=100):
    """Benchmark a single event execution.

    Parameters
    ----------
    event_name : str
        Name of event to benchmark
    sim : Simulation
        Simulation instance
    n_warmup : int
        Number of warmup runs
    n_runs : int
        Number of benchmark runs

    Returns
    -------
    mean : float
        Mean execution time in milliseconds
    std : float
        Standard deviation in milliseconds
    """
    event = sim.get_event(event_name)

    # Warmup
    for _ in range(n_warmup):
        event.execute(sim)

    # Benchmark
    times = timeit.repeat(
        lambda: event.execute(sim),
        repeat=n_runs,
        number=1
    )

    times = np.array(times) * 1000  # Convert to milliseconds
    mean = times.mean()
    std = times.std()

    return mean, std


def run_micro_benchmarks():
    """Run micro-benchmarks for all events."""
    print("=" * 70)
    print("BAM Engine - Micro-benchmarks (Individual Events)")
    print("=" * 70)
    print()

    # Initialize simulation
    sim = Simulation.init(n_firms=100, n_households=500, seed=42)

    # Run one step to get realistic state
    sim.step()

    # All unique events to benchmark (in order of execution)
    events = [
        # Planning phase
        "firms_decide_desired_production",
        "firms_calc_breakeven_price",
        "firms_adjust_price",
        "update_avg_mkt_price",
        "calc_annual_inflation_rate",
        "firms_decide_desired_labor",
        "firms_decide_vacancies",

        # Labor market
        "adjust_minimum_wage",
        "firms_decide_wage_offer",
        "workers_decide_firms_to_apply",
        "workers_send_one_round",
        "firms_hire_workers",
        "firms_calc_wage_bill",

        # Credit market
        "banks_decide_credit_supply",
        "banks_decide_interest_rate",
        "firms_decide_credit_demand",
        "firms_calc_credit_metrics",
        "firms_prepare_loan_applications",
        "firms_send_one_loan_app",
        "banks_provide_loans",
        "firms_fire_workers",

        # Production
        "firms_pay_wages",
        "workers_receive_wage",
        "firms_run_production",
        "workers_update_contracts",

        # Goods market
        "consumers_calc_propensity",
        "consumers_decide_income_to_spend",
        "consumers_decide_firms_to_visit",
        "consumers_shop_one_round",
        "consumers_finalize_purchases",

        # Revenue & bankruptcy
        "firms_collect_revenue",
        "firms_validate_debt_commitments",
        "firms_pay_dividends",
        "firms_update_net_worth",
        "mark_bankrupt_firms",
        "mark_bankrupt_banks",

        # Entry
        "spawn_replacement_firms",
        "spawn_replacement_banks",

        # Stats
        "calc_unemployment_rate",
    ]

    results = []

    for event_name in events:
        try:
            mean, std = bench_event(event_name, sim)
            results.append((event_name, mean, std))
            print(f"{event_name:45s}: {mean:7.3f} ± {std:6.3f} ms")
        except Exception as e:
            print(f"{event_name:45s}: ERROR - {e}")

    print()
    print("=" * 70)
    print("Top 10 Most Expensive Events:")
    print("=" * 70)

    # Sort by mean time
    results.sort(key=lambda x: x[1], reverse=True)

    for i, (event_name, mean, std) in enumerate(results[:10], 1):
        print(f"{i:2d}. {event_name:43s}: {mean:7.3f} ± {std:6.3f} ms")


if __name__ == "__main__":
    run_micro_benchmarks()
