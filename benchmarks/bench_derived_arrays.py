"""Benchmark the performance impact of computing vs storing derived arrays.

This script measures:
1. Current implementation (stored arrays)
2. Computed-on-read implementation (properties)
3. Hybrid implementation (cached computation with invalidation)

Focus on the most frequently accessed derived arrays:
- Worker.employed
- Employer.current_labor
- Employer.wage_bill
- Employer.n_vacancies
"""

import time
from typing import Callable

import numpy as np

import bamengine as be


def time_simulation(
    sim: be.Simulation, n_periods: int, description: str
) -> tuple[float, be.Simulation]:
    """Time a simulation run and return elapsed time and final state."""
    start = time.perf_counter()
    sim.run(n_periods=n_periods)
    elapsed = time.perf_counter() - start
    print(f"{description:50s} {elapsed:8.4f}s ({n_periods / elapsed:6.1f} periods/s)")
    return elapsed, sim


def benchmark_baseline(n_firms: int, n_households: int, n_periods: int, seed: int = 42):
    """Benchmark current implementation with stored arrays."""
    print(f"\n{'=' * 80}")
    print(f"BASELINE: Stored Arrays (current implementation)")
    print(f"Config: {n_firms} firms, {n_households} households, {n_periods} periods")
    print(f"{'=' * 80}")

    sim = be.Simulation.init(
        n_firms=n_firms,
        n_households=n_households,
        n_banks=10,
        seed=seed,
    )

    elapsed, sim = time_simulation(sim, n_periods, "Baseline (stored arrays)")
    return elapsed, sim


def benchmark_read_patterns(n_firms: int = 200, n_households: int = 1000, seed: int = 42):
    """Benchmark different read patterns for derived arrays.

    This simulates what would happen if we replace stored arrays with computed properties.
    We'll measure:
    1. Reading from stored arrays (baseline)
    2. Computing employed mask from employer array
    3. Computing current_labor via bincount
    4. Computing wage_bill via weighted bincount
    5. Computing n_vacancies from desired_labor - current_labor
    """
    print(f"\n{'=' * 80}")
    print(f"READ PATTERN MICRO-BENCHMARKS")
    print(f"Config: {n_firms} firms, {n_households} households")
    print(f"{'=' * 80}\n")

    # Create simulation and run one period to get realistic state
    sim = be.Simulation.init(
        n_firms=n_firms,
        n_households=n_households,
        n_banks=10,
        seed=seed,
    )
    sim.step()

    wrk = sim.get_role("Worker")
    emp = sim.get_role("Employer")

    # Number of iterations for micro-benchmarks
    n_iter = 10_000

    # 1. Worker.employed - current vs computed
    print("1. Worker.employed (bool array)")
    print("-" * 80)

    # Baseline: read from stored array
    start = time.perf_counter()
    for _ in range(n_iter):
        employed_mask = wrk.employed == 1
        _ = employed_mask.sum()  # force evaluation
    baseline_time = time.perf_counter() - start
    print(f"  Stored array read:        {baseline_time:8.4f}s ({n_iter / baseline_time:,.0f} ops/s)")

    # Computed: derive from employer
    start = time.perf_counter()
    for _ in range(n_iter):
        employed_mask = wrk.employer >= 0
        _ = employed_mask.sum()  # force evaluation
    computed_time = time.perf_counter() - start
    print(f"  Computed from employer:   {computed_time:8.4f}s ({n_iter / computed_time:,.0f} ops/s)")
    print(f"  Overhead: {((computed_time / baseline_time - 1) * 100):+.1f}%\n")

    # 2. Employer.current_labor - current vs computed
    print("2. Employer.current_labor (int array)")
    print("-" * 80)

    # Baseline: read from stored array
    start = time.perf_counter()
    for _ in range(n_iter):
        labor = emp.current_labor.copy()  # copy to simulate usage
        _ = labor.sum()
    baseline_time = time.perf_counter() - start
    print(f"  Stored array read:        {baseline_time:8.4f}s ({n_iter / baseline_time:,.0f} ops/s)")

    # Computed: bincount on employer IDs
    start = time.perf_counter()
    for _ in range(n_iter):
        employed_workers = wrk.employer >= 0
        labor = np.bincount(
            wrk.employer[employed_workers],
            minlength=n_firms
        )
        _ = labor.sum()
    computed_time = time.perf_counter() - start
    print(f"  Computed via bincount:    {computed_time:8.4f}s ({n_iter / computed_time:,.0f} ops/s)")
    print(f"  Overhead: {((computed_time / baseline_time - 1) * 100):+.1f}%\n")

    # 3. Employer.wage_bill - current vs computed
    print("3. Employer.wage_bill (float array)")
    print("-" * 80)

    # Baseline: read from stored array
    start = time.perf_counter()
    for _ in range(n_iter):
        wage_bill = emp.wage_bill.copy()
        _ = wage_bill.sum()
    baseline_time = time.perf_counter() - start
    print(f"  Stored array read:        {baseline_time:8.4f}s ({n_iter / baseline_time:,.0f} ops/s)")

    # Computed: weighted bincount
    start = time.perf_counter()
    for _ in range(n_iter):
        employed_workers = wrk.employer >= 0
        wage_bill = np.bincount(
            wrk.employer[employed_workers],
            weights=wrk.wage[employed_workers],
            minlength=n_firms
        )
        _ = wage_bill.sum()
    computed_time = time.perf_counter() - start
    print(f"  Computed via bincount:    {computed_time:8.4f}s ({n_iter / computed_time:,.0f} ops/s)")
    print(f"  Overhead: {((computed_time / baseline_time - 1) * 100):+.1f}%\n")

    # 4. Employer.n_vacancies - current vs computed
    print("4. Employer.n_vacancies (int array)")
    print("-" * 80)

    # Baseline: read from stored array
    start = time.perf_counter()
    for _ in range(n_iter):
        vacancies = emp.n_vacancies.copy()
        _ = vacancies.sum()
    baseline_time = time.perf_counter() - start
    print(f"  Stored array read:        {baseline_time:8.4f}s ({n_iter / baseline_time:,.0f} ops/s)")

    # Computed: desired - current (assuming we still compute current_labor)
    start = time.perf_counter()
    for _ in range(n_iter):
        # Assume current_labor is also computed
        employed_workers = wrk.employer >= 0
        current_labor = np.bincount(
            wrk.employer[employed_workers],
            minlength=n_firms
        )
        vacancies = np.maximum(0, emp.desired_labor - current_labor)
        _ = vacancies.sum()
    computed_time = time.perf_counter() - start
    print(f"  Computed from desired:    {computed_time:8.4f}s ({n_iter / computed_time:,.0f} ops/s)")
    print(f"  Overhead: {((computed_time / baseline_time - 1) * 100):+.1f}%\n")


def estimate_full_simulation_impact():
    """Estimate the performance impact on a full simulation.

    Based on micro-benchmark results, we can estimate:
    - How many times each array is accessed per period
    - Total overhead from computing vs storing
    """
    print(f"\n{'=' * 80}")
    print(f"ESTIMATED FULL SIMULATION IMPACT")
    print(f"{'=' * 80}\n")

    # These are rough estimates based on code inspection
    # Each count is per period
    access_counts = {
        "Worker.employed": {
            "reads_per_period": 20,  # Used in many filters across labor, production, credit
            "writes_per_period": 3,  # Hiring, firing, contract expiration
        },
        "Employer.current_labor": {
            "reads_per_period": 5,  # Mostly validation and capacity checks
            "writes_per_period": 5,  # Hiring, firing, bankruptcy, expiration
        },
        "Employer.wage_bill": {
            "reads_per_period": 4,  # Production payments, credit checks
            "writes_per_period": 2,  # Labor market calc, firing adjustment
        },
        "Employer.n_vacancies": {
            "reads_per_period": 8,  # Labor market matching, multiple rounds
            "writes_per_period": 2,  # Planning calc, hiring decrement
        },
    }

    print("Access frequency estimates (per period):")
    print("-" * 80)
    for array_name, counts in access_counts.items():
        total = counts["reads_per_period"] + counts["writes_per_period"]
        print(f"{array_name:30s} Reads: {counts['reads_per_period']:3d}  "
              f"Writes: {counts['writes_per_period']:3d}  Total: {total:3d}")

    print("\nNote: These are estimates. Actual counts vary by market parameters (max_M, max_H, max_Z).")
    print("      max_M=20, max_H=10, max_Z=7 → ~37 labor market rounds per period")


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 80)
    print("DERIVED ARRAY BENCHMARK SUITE")
    print("=" * 80)

    # Configuration
    configs = [
        {"n_firms": 100, "n_households": 500, "n_periods": 100, "label": "Small"},
        {"n_firms": 200, "n_households": 1000, "n_periods": 100, "label": "Medium"},
        {"n_firms": 500, "n_households": 2500, "n_periods": 100, "label": "Large"},
    ]

    # 1. Baseline benchmarks
    baseline_times = {}
    for config in configs:
        elapsed, _ = benchmark_baseline(
            n_firms=config["n_firms"],
            n_households=config["n_households"],
            n_periods=config["n_periods"],
        )
        baseline_times[config["label"]] = elapsed

    # 2. Micro-benchmarks for read patterns
    benchmark_read_patterns(n_firms=200, n_households=1000)

    # 3. Impact estimation
    estimate_full_simulation_impact()

    # 4. Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}\n")
    print("Baseline Times:")
    for label, elapsed in baseline_times.items():
        print(f"  {label:10s} {elapsed:8.4f}s")

    print("\nRecommendations will be based on:")
    print("  1. Micro-benchmark overhead percentages")
    print("  2. Access frequency estimates")
    print("  3. Code complexity impact")
    print("  4. Memory savings vs performance trade-off")


if __name__ == "__main__":
    main()
