"""
Distributional-equivalence test for consumers_decide_firms_to_visit.

Guards that the sparse sampler (`sample_k_per_row`) produces a uniform
distribution over firms, identical in distribution to the old dense
priorities approach.
"""

from __future__ import annotations

import numpy as np

import bamengine as bam


def test_consumer_visits_uniform_over_firms_no_loyalty() -> None:
    """With consumer_matching='random', selected firms should be ~uniform over all firms."""
    sim = bam.Simulation.init(
        seed=0,
        n_firms=20,
        n_households=2000,
        n_banks=2,
        consumer_matching="random",
        log_level="ERROR",
    )
    sim.run(n_periods=3)  # warm up so consumers have budget
    con = sim.get_role("Consumer")
    con.income_to_spend[:] = 1.0  # everyone shops
    sim.get_event("consumers_decide_firms_to_visit").execute(sim)
    targets = con.shop_visits_targets[con.shop_visits_targets >= 0]
    counts = np.bincount(targets, minlength=20)
    freq = counts / counts.sum()
    assert np.all(np.abs(freq - 1 / 20) < 0.02)  # roughly uniform
