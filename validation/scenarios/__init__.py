"""Validation scenarios for BAM Engine.

This subpackage provides scenario definitions and visualizations that reproduce
results from Delli Gatti et al. (2011).

Scenarios:
    baseline/: Section 3.9.1 - Standard BAM model behavior
    growth_plus/: Section 3.9.2 - Endogenous productivity growth via R&D
    buffer_stock/: Section 3.9.4 - Buffer-stock consumption extension

Usage:
    # Run scenario with visualization
    from validation.scenarios.baseline import run_scenario
    run_scenario(seed=0, show_plot=True)

    from validation.scenarios.growth_plus import run_scenario
    run_scenario(seed=2, show_plot=True)

    from validation.scenarios.buffer_stock import run_scenario
    run_scenario(seed=0, show_plot=True)

    # Or run as module:
    # python -m validation.scenarios.baseline
    # python -m validation.scenarios.growth_plus
    # python -m validation.scenarios.buffer_stock

    # For extensions, import from extensions package:
    from extensions.rnd import RnD, RND_EVENTS, RND_CONFIG
    from extensions.buffer_stock import BufferStock, BUFFER_STOCK_EVENTS, BUFFER_STOCK_CONFIG
"""

from __future__ import annotations

from validation.scenarios.baseline import SCENARIO as BASELINE_SCENARIO
from validation.scenarios.buffer_stock import SCENARIO as BUFFER_STOCK_SCENARIO
from validation.scenarios.growth_plus import SCENARIO as GROWTH_PLUS_SCENARIO
from validation.types import Scenario

SCENARIO_REGISTRY: dict[str, Scenario] = {
    "baseline": BASELINE_SCENARIO,
    "growth_plus": GROWTH_PLUS_SCENARIO,
    "buffer_stock": BUFFER_STOCK_SCENARIO,
}


def get_scenario(name: str) -> Scenario:
    """Look up a scenario by name.

    Parameters
    ----------
    name : str
        Scenario name (e.g. "baseline", "growth_plus", "buffer_stock").

    Returns
    -------
    Scenario
        The scenario configuration.

    Raises
    ------
    KeyError
        If the scenario name is not found.
    """
    if name not in SCENARIO_REGISTRY:
        valid = ", ".join(sorted(SCENARIO_REGISTRY))
        raise KeyError(f"Unknown scenario: {name!r}. Valid: {valid}")
    return SCENARIO_REGISTRY[name]
