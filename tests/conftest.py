import pytest

from bamengine.scheduler import Scheduler


@pytest.fixture
def tiny_sched() -> Scheduler:
    """A minimal deterministic scheduler for fast integration tests."""
    return Scheduler.init(
        n_firms=4,
        n_households=10,
        n_banks=2,
        seed=123,
        # keep default‐ish parameters explicit
        h_rho=0.1,
        h_xi=0.05,
        max_M=4,
        theta=8,
    )
