"""Full simulation benchmarking (macro-benchmarks).

Benchmarks end-to-end performance of full simulation runs
with various configurations.
"""

import time
import numpy as np
import logging
from bamengine import Simulation

# Disable logging for benchmarks
logging.getLogger('bamengine').setLevel(logging.ERROR)


def benchmark_simulation(n_firms, n_households, n_periods, n_runs=10):
    """Benchmark full simulation runs.

    Parameters
    ----------
    n_firms : int
        Number of firms
    n_households : int
        Number of households
    n_periods : int
        Number of periods to simulate
    n_runs : int
        Number of benchmark runs (default: 10)

    Returns
    -------
    mean : float
        Mean execution time in seconds
    std : float
        Standard deviation of execution times
    """
    times = []

    for i in range(n_runs):
        sim = Simulation.init(
            n_firms=n_firms,
            n_households=n_households,
            seed=42 + i
        )

        start = time.perf_counter()
        sim.run(n_periods)
        elapsed = time.perf_counter() - start

        times.append(elapsed)

    times = np.array(times)
    mean = times.mean()
    std = times.std()

    print(f"Configuration: {n_firms} firms, {n_households} households, {n_periods} periods")
    print(f"Average time: {mean:.3f} ± {std:.3f} seconds")
    print(f"Throughput: {n_periods / mean:.1f} periods/second")
    print(f"Min/Max: {times.min():.3f}s / {times.max():.3f}s")

    return mean, std


def run_all_benchmarks():
    """Run benchmarks for various configurations."""
    print("=" * 70)
    print("BAM Engine - Full Simulation Benchmarks")
    print("=" * 70)
    print()

    # Benchmark configurations
    # Note: Using 1000 periods as recommended in original BAM paper
    configs = [
        ("Small", 100, 500, 1000),
        ("Medium", 200, 1000, 1000),
        ("Large", 500, 2500, 1000),
    ]

    results = {}

    for name, n_firms, n_households, n_periods in configs:
        print(f"Benchmarking {name} configuration...")
        mean, std = benchmark_simulation(n_firms, n_households, n_periods)
        results[name] = (mean, std)
        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    for name, (mean, std) in results.items():
        print(f"{name:10s}: {mean:.3f} ± {std:.3f} seconds")


if __name__ == "__main__":
    run_all_benchmarks()
