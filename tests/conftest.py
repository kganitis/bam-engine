"""Pytest configuration and fixtures for bamengine tests."""

import os

import pytest

import bamengine.events  # noqa: F401 - register all events
from bamengine import logging
from bamengine.core.registry import clear_registry
from bamengine.simulation import Simulation


@pytest.fixture
def clean_registry():
    """
    Save registry state, clear it for the test, then restore it.

    This fixture should be explicitly requested by tests that need isolation
    from real BAM components or from test pollution by other test modules.

    DO NOT use autouse=True, as it would interfere with integration tests
    that rely on real components being registered.
    """
    # noinspection PyProtectedMember
    from bamengine.core.registry import (
        _EVENT_REGISTRY,
        _RELATIONSHIP_REGISTRY,
        _ROLE_REGISTRY,
    )

    # Save current state
    saved_roles = dict(_ROLE_REGISTRY)
    saved_events = dict(_EVENT_REGISTRY)
    saved_relationships = dict(_RELATIONSHIP_REGISTRY)

    # Clear for test
    clear_registry()

    yield

    # Restore original state
    _ROLE_REGISTRY.clear()
    _ROLE_REGISTRY.update(saved_roles)
    _EVENT_REGISTRY.clear()
    _EVENT_REGISTRY.update(saved_events)
    _RELATIONSHIP_REGISTRY.clear()
    _RELATIONSHIP_REGISTRY.update(saved_relationships)


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


@pytest.fixture(autouse=True)
def mute_bamengine_logs(caplog):
    # Optimize log level based on context:
    # - CI coverage run (ubuntu-latest + Python 3.12): DEBUG to execute all logging for accurate coverage
    # - All other CI runs: ERROR for faster execution
    # - Local runs: ERROR for faster tests
    if os.environ.get("COVERAGE_RUN") == "true":
        # Only set for the specific CI run that uploads to Codecov
        level = logging.DEBUG
    else:
        # All other runs (local and non-coverage CI runs)
        level = logging.ERROR

    # Set both caplog level (for capture) and actual logger level
    caplog.set_level(level, logger="bamengine")
    logging.getLogger("bamengine").setLevel(level)
