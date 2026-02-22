"""Unit tests for consumer_matching parameter in goods market events.

Tests that the ``consumer_matching`` config parameter correctly toggles
the preferential attachment (loyalty) mechanism.
"""

from __future__ import annotations

import numpy as np

from bamengine import make_rng
from bamengine.events._internal.goods_market import (
    consumers_calc_propensity,
    consumers_decide_firms_to_visit,
    consumers_decide_income_to_spend,
)
from tests.helpers.factories import mock_consumer, mock_producer


def _setup(*, n_hh: int = 4, n_firms: int = 3, Z: int = 2, seed: int = 0):
    """Return Consumer, Producer, rng, Z with loyalty set on household 0."""
    rng = make_rng(seed)
    con = mock_consumer(
        n=n_hh,
        queue_z=Z,
        income=np.full(n_hh, 3.0),
        savings=np.full(n_hh, 2.0),
    )
    prod = mock_producer(
        n=n_firms,
        price=np.array([1.0, 1.2, 0.9]),
        inventory=np.array([5.0, 5.0, 5.0]),
        production=np.array([5.0, 8.0, 4.0]),
    )
    # Set loyalty: hh-0 is loyal to firm-1 (largest producer)
    con.largest_prod_prev[0] = 1

    consumers_calc_propensity(con, avg_sav=1.0, beta=0.9)
    consumers_decide_income_to_spend(con)
    return con, prod, rng, Z


class TestLoyaltyMode:
    """Tests for consumer_matching='loyalty' (default)."""

    def test_loyalty_firm_included_in_consideration_set(self) -> None:
        """With loyalty mode, the loyal firm should be included."""
        con, prod, rng, Z = _setup()
        consumers_decide_firms_to_visit(
            con, prod, max_Z=Z, rng=rng, consumer_matching="loyalty"
        )
        targets = con.shop_visits_targets[0]
        assert 1 in targets[targets >= 0]

    def test_loyalty_updates_largest_prod_prev(self) -> None:
        """After shopping selection, largest_prod_prev should be updated."""
        con, prod, rng, Z = _setup()
        consumers_decide_firms_to_visit(
            con, prod, max_Z=Z, rng=rng, consumer_matching="loyalty"
        )
        # Active consumers should have their loyalty updated to the
        # largest producer in their consideration set
        active = np.where(con.income_to_spend > 0.0)[0]
        if len(active) > 0:
            # At least some loyalty values should be valid firm IDs
            loyalty_vals = con.largest_prod_prev[active]
            assert np.all((loyalty_vals >= 0) & (loyalty_vals < prod.price.size))

    def test_loyalty_is_default_behavior(self) -> None:
        """Calling without consumer_matching should behave like 'loyalty'."""
        con1, prod1, rng1, Z = _setup(seed=42)
        consumers_decide_firms_to_visit(con1, prod1, max_Z=Z, rng=rng1)

        con2, prod2, rng2, _ = _setup(seed=42)
        consumers_decide_firms_to_visit(
            con2, prod2, max_Z=Z, rng=rng2, consumer_matching="loyalty"
        )

        np.testing.assert_array_equal(
            con1.shop_visits_targets, con2.shop_visits_targets
        )


class TestRandomMode:
    """Tests for consumer_matching='random' (PA off)."""

    def test_random_mode_executes_without_error(self) -> None:
        """Random mode should work without crashing."""
        con, prod, rng, Z = _setup()
        consumers_decide_firms_to_visit(
            con, prod, max_Z=Z, rng=rng, consumer_matching="random"
        )
        # Active consumers should still have valid targets
        active = np.where(con.income_to_spend > 0.0)[0]
        heads = con.shop_visits_head[active]
        assert (heads >= 0).all()

    def test_random_mode_skips_loyalty_update(self) -> None:
        """In random mode, largest_prod_prev should NOT be updated."""
        con, prod, rng, Z = _setup()
        original_loyalty = con.largest_prod_prev.copy()
        consumers_decide_firms_to_visit(
            con, prod, max_Z=Z, rng=rng, consumer_matching="random"
        )
        np.testing.assert_array_equal(con.largest_prod_prev, original_loyalty)

    def test_random_mode_still_selects_firms(self) -> None:
        """Random mode should still select firms for active consumers."""
        con, prod, rng, Z = _setup()
        consumers_decide_firms_to_visit(
            con, prod, max_Z=Z, rng=rng, consumer_matching="random"
        )
        active = np.where(con.income_to_spend > 0.0)[0]
        if len(active) > 0:
            targets = con.shop_visits_targets[0]
            assert np.any(targets >= 0)


class TestConfigIntegration:
    """Tests for consumer_matching config parameter."""

    def test_config_default_is_loyalty(self) -> None:
        """Default config should use 'loyalty' matching."""
        from bamengine import Simulation

        sim = Simulation.init(seed=42)
        assert sim.config.consumer_matching == "loyalty"

    def test_config_accepts_random(self) -> None:
        """Config should accept 'random' as a valid value."""
        from bamengine import Simulation

        sim = Simulation.init(seed=42, consumer_matching="random")
        assert sim.config.consumer_matching == "random"

    def test_config_validation_rejects_invalid(self) -> None:
        """Invalid consumer_matching values should be rejected."""
        import pytest

        from bamengine import Simulation

        with pytest.raises(ValueError, match="consumer_matching"):
            Simulation.init(seed=42, consumer_matching="invalid_value")
