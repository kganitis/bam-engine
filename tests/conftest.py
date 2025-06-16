# tests/conftest.py
import pytest

from bamengine.simulation import Simulation


@pytest.fixture
def tiny_sched() -> Simulation:
    """A minimal deterministic simulation for fast integration tests."""
    return Simulation.init(
        n_firms=6,
        n_households=15,
        n_banks=3,
        seed=123,
        # keep default‐ish parameters explicit
        h_rho=0.1,
        h_xi=0.05,
        h_phi=0.1,
        h_eta=0.1,
        max_M=4,
        max_H=2,
        max_Z=2,
        theta=8,
        beta=0.87,
        delta=0.15,
    )
