"""Profit taxation extension for structural experiments.

This extension implements a simple profit tax that removes revenue from the
economy without redistribution. It is used in the entry neutrality experiment
(Section 3.10.2) to increase bankruptcies and test whether the automatic
firm entry mechanism artificially drives recovery.

Usage::

    from extensions.taxation import TAXATION

    sim = bam.Simulation.init(**config)
    sim.use(TAXATION)
    results = sim.run()

Or manually::

    from extensions.taxation import FirmsTaxProfits, TAXATION_EVENTS, TAXATION_CONFIG

    sim = bam.Simulation.init(**config)
    sim.use_events(*TAXATION_EVENTS)
    sim.use_config(TAXATION_CONFIG)
    results = sim.run()
"""

from __future__ import annotations

from bamengine import Extension
from extensions.taxation.events import FirmsTaxProfits

TAXATION_EVENTS = [FirmsTaxProfits]

TAXATION_CONFIG = {
    "profit_tax_rate": 0.0,  # Default no-tax; overridden per experiment
}

TAXATION = Extension(
    roles={},
    events=TAXATION_EVENTS,
    relationships=[],
    config_dict=TAXATION_CONFIG,
)

__all__ = [
    "FirmsTaxProfits",
    "TAXATION",
    "TAXATION_EVENTS",
    "TAXATION_CONFIG",
]
