"""Canonical baseline-BAM parameter set for the comparison harness.

This is the single source of truth for the model parameters every framework
runner is held to. The economic parameters are read directly from bamengine's
shipped defaults (``src/bamengine/config/defaults.yml``) so the reference runner
and any re-implementation are compared against identical numbers.

Two things are exposed:

``canonical_params()``
    The scalar *economic* parameters only. Population sizes, run length, the
    RNG seed, and non-economic / environment keys (e.g. ``pipeline_path``,
    ``logging``) are excluded. These are supplied separately per run.

``population_for(n_firms)``
    Population sizes at the canonical BAM ratio of 1 firm : 5 households :
    0.1 banks (i.e. ``n_banks == n_firms // 10``, floored at 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# defaults.yml lives at <repo>/src/bamengine/config/defaults.yml.
# This file is <repo>/comparison/orchestrator/params.py, so parents[2] == <repo>.
_DEFAULTS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "bamengine"
    / "config"
    / "defaults.yml"
)

# Keys that are NOT canonical economic parameters and must never appear in the
# returned dict. Population/run-length/seed are supplied per run. ``pipeline_path``
# is an environment-specific path (value is null and not portable across
# frameworks) and ``logging`` is a verbosity concern, not model behaviour.
_EXCLUDED_KEYS = frozenset(
    {
        "n_firms",
        "n_households",
        "n_banks",
        "n_periods",
        "seed",
        "pipeline_path",
        "logging",
    }
)


def canonical_params() -> dict[str, Any]:
    """Return the canonical scalar economic parameters from ``defaults.yml``.

    Reads bamengine's shipped defaults and returns only scalar parameters,
    excluding population sizes, run length, RNG seed, and non-economic keys
    (``pipeline_path``, ``logging``). Every returned key is accepted by
    :meth:`bamengine.Simulation.init`.

    Returns
    -------
    dict
        Mapping of economic parameter name to value.
    """
    cfg = yaml.safe_load(_DEFAULTS.read_text())
    # defaults.yml is a flat mapping; tolerate a nested "simulation" section.
    flat = cfg.get("simulation", cfg) if isinstance(cfg, dict) else {}
    return {
        k: v
        for k, v in flat.items()
        if k not in _EXCLUDED_KEYS and not isinstance(v, (dict, list))
    }


def population_for(n_firms: int) -> dict[str, int]:
    """Return population sizes at the canonical BAM ratio.

    1 firm : 5 households : 0.1 banks (banks floored at 1).

    Parameters
    ----------
    n_firms : int
        Number of firms (Producer/Employer/Borrower agents).

    Returns
    -------
    dict
        ``{"n_firms": ..., "n_households": ..., "n_banks": ...}``.
    """
    return {
        "n_firms": n_firms,
        "n_households": n_firms * 5,
        "n_banks": max(1, n_firms // 10),
    }
