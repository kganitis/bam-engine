"""Quick benchmarks using pytest-benchmark for local development.

These benchmarks use the pytest-benchmark plugin for fast local iteration.
Unlike ASV benchmarks (which track performance across commits), these are
designed for quick feedback during development.

Run with:
    pytest tests/performance/test_quick_benchmarks.py -v

For detailed statistics:
    pytest tests/performance/test_quick_benchmarks.py -v --benchmark-verbose

To compare against saved baseline:
    pytest tests/performance/test_quick_benchmarks.py --benchmark-compare

To save a baseline:
    pytest tests/performance/test_quick_benchmarks.py --benchmark-save=baseline

Note: These tests require the pytest-benchmark plugin:
    pip install pytest-benchmark
"""

import pytest

from bamengine import Simulation

# Skip entire module if pytest-benchmark is not installed
pytest.importorskip("pytest_benchmark")


@pytest.fixture
def sim():
    """Create a small simulation for benchmarking."""
    return Simulation.init(
        n_firms=50, n_households=250, seed=42, logging={"default_level": "ERROR"}
    )


@pytest.fixture
def sim_steady_state():
    """Create a simulation in steady state for event benchmarks."""
    sim = Simulation.init(
        n_firms=50, n_households=250, seed=42, logging={"default_level": "ERROR"}
    )
    sim.run(10)  # Reach steady state
    return sim


# Core simulation benchmarks
@pytest.mark.benchmark(group="core")
def test_single_step(benchmark, sim):
    """Benchmark single simulation step."""
    benchmark(sim.step)


@pytest.mark.benchmark(group="core")
def test_10_periods(benchmark, sim):
    """Benchmark 10 simulation periods."""
    benchmark(sim.run, n_periods=10)


@pytest.mark.benchmark(group="init")
def test_init_small(benchmark):
    """Benchmark initialization (small config: 50 firms)."""
    benchmark(
        Simulation.init,
        n_firms=50,
        n_households=250,
        seed=42,
        logging={"default_level": "ERROR"},
    )


@pytest.mark.benchmark(group="init")
def test_init_medium(benchmark):
    """Benchmark initialization (medium config: 100 firms)."""
    benchmark(
        Simulation.init,
        n_firms=100,
        n_households=500,
        seed=42,
        logging={"default_level": "ERROR"},
    )


# Critical event benchmarks
@pytest.mark.benchmark(group="events")
def test_consumers_decide_firms_to_visit(benchmark, sim_steady_state):
    """Benchmark goods market firm selection (~48% of runtime)."""
    event = sim_steady_state.get_event("consumers_decide_firms_to_visit")
    benchmark(event.execute, sim_steady_state)


@pytest.mark.benchmark(group="events")
def test_consumers_shop_sequential(benchmark, sim_steady_state):
    """Benchmark sequential shopping (all rounds)."""
    event = sim_steady_state.get_event("consumers_shop_sequential")
    benchmark(event.execute, sim_steady_state)


@pytest.mark.benchmark(group="events")
def test_workers_decide_firms_to_apply(benchmark, sim_steady_state):
    """Benchmark labor market firm selection."""
    event = sim_steady_state.get_event("workers_decide_firms_to_apply")
    benchmark(event.execute, sim_steady_state)


@pytest.mark.benchmark(group="events")
def test_workers_apply_to_firms(benchmark, sim_steady_state):
    """Benchmark cascade labor matching process."""
    event = sim_steady_state.get_event("workers_apply_to_firms")
    benchmark(event.execute, sim_steady_state)
