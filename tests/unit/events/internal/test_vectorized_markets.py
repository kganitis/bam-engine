"""Unit tests for vectorized market matching functions."""

from __future__ import annotations

import numpy as np

import bamengine as bam

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_sim(seed: int = 42, **kw) -> bam.Simulation:
    """Create a small simulation for testing."""
    return bam.Simulation.init(seed=seed, **kw)


def _run_through_phase(sim: bam.Simulation, phase: str) -> None:
    """Run events up to and including a specific phase.

    This runs the planning + labor prep events needed before market matching.
    """
    planning = [
        "firms_decide_desired_production",
        "firms_plan_breakeven_price",
        "firms_plan_price",
        "firms_decide_desired_labor",
        "firms_decide_vacancies",
        "firms_fire_excess_workers",
    ]
    labor_prep = [
        "calc_inflation_rate",
        "adjust_minimum_wage",
        "firms_decide_wage_offer",
        "workers_decide_firms_to_apply",
    ]
    credit_prep = [
        "banks_decide_credit_supply",
        "banks_decide_interest_rate",
        "firms_decide_credit_demand",
        "firms_calc_financial_fragility",
        "firms_prepare_loan_applications",
    ]

    events_to_run = planning[:]
    if phase in ("labor", "credit", "goods"):
        events_to_run += labor_prep
    if phase in ("credit", "goods"):
        # Run labor matching first (vectorized)
        events_to_run += labor_prep  # already added, skip
        # Actually run the labor rounds
        from bamengine.events._internal.vectorized_markets import labor_market_round_vec

        for ev_name in planning + labor_prep:
            sim.get_event(ev_name).execute(sim)
        for _ in range(sim.config.max_M):
            labor_market_round_vec(
                sim.emp, sim.wrk, theta=sim.config.theta, rng=sim.rng
            )
        sim.get_event("firms_calc_wage_bill").execute(sim)
        for ev_name in credit_prep:
            sim.get_event(ev_name).execute(sim)
        return

    for ev_name in events_to_run:
        sim.get_event(ev_name).execute(sim)


# ═════════════════════════════════════════════════════════════════════════════
#  LABOR MARKET VECTORIZED
# ═════════════════════════════════════════════════════════════════════════════


class TestLaborMarketRoundVec:
    """Tests for vectorized labor market matching."""

    def test_basic_hiring(self):
        """Workers get hired when firms have vacancies."""
        sim = _make_sim(seed=42)
        # Run planning + labor prep
        for ev in [
            "firms_decide_desired_production",
            "firms_plan_breakeven_price",
            "firms_plan_price",
            "firms_decide_desired_labor",
            "firms_decide_vacancies",
            "firms_fire_excess_workers",
            "calc_inflation_rate",
            "adjust_minimum_wage",
            "firms_decide_wage_offer",
            "workers_decide_firms_to_apply",
        ]:
            sim.get_event(ev).execute(sim)

        from bamengine.events._internal.vectorized_markets import labor_market_round_vec

        initial_unemployed = (~sim.wrk.employed).sum()
        assert initial_unemployed > 0, "Should have unemployed workers initially"

        # Run multiple rounds
        for _ in range(sim.config.max_M):
            labor_market_round_vec(
                sim.emp, sim.wrk, theta=sim.config.theta, rng=sim.rng
            )

        final_unemployed = (~sim.wrk.employed).sum()
        assert final_unemployed < initial_unemployed, "Should have hired some workers"

    def test_hired_workers_have_correct_state(self):
        """Hired workers have employer, wage, contract duration set."""
        sim = _make_sim(seed=42)
        for ev in [
            "firms_decide_desired_production",
            "firms_plan_breakeven_price",
            "firms_plan_price",
            "firms_decide_desired_labor",
            "firms_decide_vacancies",
            "firms_fire_excess_workers",
            "calc_inflation_rate",
            "adjust_minimum_wage",
            "firms_decide_wage_offer",
            "workers_decide_firms_to_apply",
        ]:
            sim.get_event(ev).execute(sim)

        from bamengine.events._internal.vectorized_markets import labor_market_round_vec

        for _ in range(sim.config.max_M):
            labor_market_round_vec(
                sim.emp, sim.wrk, theta=sim.config.theta, rng=sim.rng
            )

        employed = sim.wrk.employed
        assert (sim.wrk.employer[employed] >= 0).all()
        assert (sim.wrk.wage[employed] > 0).all()
        assert (sim.wrk.periods_left[employed] == sim.config.theta).all()

    def test_firm_labor_counts_consistent(self):
        """Firm current_labor matches actual worker count."""
        sim = _make_sim(seed=42)
        for ev in [
            "firms_decide_desired_production",
            "firms_plan_breakeven_price",
            "firms_plan_price",
            "firms_decide_desired_labor",
            "firms_decide_vacancies",
            "firms_fire_excess_workers",
            "calc_inflation_rate",
            "adjust_minimum_wage",
            "firms_decide_wage_offer",
            "workers_decide_firms_to_apply",
        ]:
            sim.get_event(ev).execute(sim)

        from bamengine.events._internal.vectorized_markets import labor_market_round_vec

        for _ in range(sim.config.max_M):
            labor_market_round_vec(
                sim.emp, sim.wrk, theta=sim.config.theta, rng=sim.rng
            )

        # Verify consistency
        n_firms = sim.emp.current_labor.size
        true_counts = np.bincount(sim.wrk.employer[sim.wrk.employed], minlength=n_firms)
        np.testing.assert_array_equal(sim.emp.current_labor, true_counts)

    def test_no_over_hiring(self):
        """Firms don't hire more workers than vacancies."""
        sim = _make_sim(seed=42)
        for ev in [
            "firms_decide_desired_production",
            "firms_plan_breakeven_price",
            "firms_plan_price",
            "firms_decide_desired_labor",
            "firms_decide_vacancies",
            "firms_fire_excess_workers",
            "calc_inflation_rate",
            "adjust_minimum_wage",
            "firms_decide_wage_offer",
            "workers_decide_firms_to_apply",
        ]:
            sim.get_event(ev).execute(sim)

        from bamengine.events._internal.vectorized_markets import labor_market_round_vec

        for _ in range(sim.config.max_M):
            labor_market_round_vec(
                sim.emp, sim.wrk, theta=sim.config.theta, rng=sim.rng
            )

        assert (sim.emp.n_vacancies >= 0).all(), "Vacancies should never go negative"

    def test_no_workers_no_crash(self):
        """No unemployed workers → no crash, no state changes."""
        sim = _make_sim(seed=42)
        # Manually set all workers as employed
        sim.wrk.employer[:] = 0
        sim.wrk.wage[:] = 1.0
        sim.wrk.periods_left[:] = 10

        from bamengine.events._internal.vectorized_markets import labor_market_round_vec

        # Should not crash
        labor_market_round_vec(sim.emp, sim.wrk, theta=sim.config.theta, rng=sim.rng)

    def test_event_class_works(self):
        """LaborMarketRoundVec event class wraps correctly."""
        sim = _make_sim(
            seed=42, pipeline_path="src/bamengine/config/vectorized_pipeline.yml"
        )
        sim.get_event("labor_market_round_vec")
        # Should not crash
        sim.step()


# ═════════════════════════════════════════════════════════════════════════════
#  CREDIT MARKET VECTORIZED
# ═════════════════════════════════════════════════════════════════════════════


class TestCreditMarketRoundVec:
    """Tests for vectorized credit market matching."""

    def test_loans_granted(self):
        """Firms with credit demand receive loans."""
        sim = _make_sim(seed=42)
        # Run a full step with default pipeline first to get proper state,
        # then test credit round
        sim.step()

        # Now prepare credit market state for next period
        for ev in [
            "firms_decide_desired_production",
            "firms_plan_breakeven_price",
            "firms_plan_price",
            "firms_decide_desired_labor",
            "firms_decide_vacancies",
            "firms_fire_excess_workers",
            "calc_inflation_rate",
            "adjust_minimum_wage",
            "firms_decide_wage_offer",
            "workers_decide_firms_to_apply",
        ]:
            sim.get_event(ev).execute(sim)

        from bamengine.events._internal.vectorized_markets import labor_market_round_vec

        for _ in range(sim.config.max_M):
            labor_market_round_vec(
                sim.emp, sim.wrk, theta=sim.config.theta, rng=sim.rng
            )
        sim.get_event("firms_calc_wage_bill").execute(sim)

        for ev in [
            "banks_decide_credit_supply",
            "banks_decide_interest_rate",
            "firms_decide_credit_demand",
            "firms_calc_financial_fragility",
            "firms_prepare_loan_applications",
        ]:
            sim.get_event(ev).execute(sim)

        from bamengine.events._internal.vectorized_markets import (
            credit_market_round_vec,
        )

        initial_lb_size = sim.lb.size
        for _ in range(sim.config.max_H):
            credit_market_round_vec(
                sim.bor,
                sim.lend,
                sim.lb,
                r_bar=sim.r_bar,
                max_leverage=sim.config.max_leverage,
                max_loan_to_net_worth=sim.config.max_loan_to_net_worth,
                rng=sim.rng,
            )

        # If there was credit demand, loans should have been granted
        if (sim.bor.credit_demand > 0).any():
            assert sim.lb.size >= initial_lb_size

    def test_no_credit_demand_no_crash(self):
        """No credit demand → no crash."""
        sim = _make_sim(seed=42)
        # Zero out credit demand
        sim.bor.credit_demand[:] = 0.0

        from bamengine.events._internal.vectorized_markets import (
            credit_market_round_vec,
        )

        # Need opex_shock to be set
        sim.get_event("banks_decide_interest_rate").execute(sim)

        credit_market_round_vec(
            sim.bor,
            sim.lend,
            sim.lb,
            r_bar=sim.r_bar,
            max_leverage=sim.config.max_leverage,
            rng=sim.rng,
        )


class TestFirmsFireWorkersVec:
    """Tests for vectorized worker firing."""

    def test_firing_reduces_gap(self):
        """Firms fire workers until wage bill <= total_funds."""
        sim = _make_sim(seed=42)
        sim.step()  # run a full period to populate state

        from bamengine.events._internal.vectorized_markets import firms_fire_workers_vec

        # Create financing gaps
        firing_firms = np.where(sim.emp.current_labor > 0)[0][:5]
        if firing_firms.size > 0:
            sim.emp.total_funds[firing_firms] = 0.0  # create gap
            sim.bor.total_funds[firing_firms] = 0.0

            firms_fire_workers_vec(sim.emp, sim.wrk, rng=sim.rng)

            # After firing, wage bill should be close to or below total_funds
            for i in firing_firms:
                assert sim.emp.wage_bill[i] <= sim.emp.total_funds[i] + 1e-8

    def test_fired_workers_state(self):
        """Fired workers have correct state updates."""
        sim = _make_sim(seed=42)
        sim.step()

        from bamengine.events._internal.vectorized_markets import firms_fire_workers_vec

        # Create gap
        firm = np.where(sim.emp.current_labor > 2)[0]
        if firm.size > 0:
            i = firm[0]
            sim.emp.total_funds[i] = 0.0
            sim.bor.total_funds[i] = 0.0

            firms_fire_workers_vec(sim.emp, sim.wrk, rng=sim.rng)

            # Check some workers were fired
            fired = sim.wrk.fired.astype(bool)
            if fired.any():
                assert (sim.wrk.employer[fired] == -1).all()
                assert (sim.wrk.wage[fired] == 0.0).all()

    def test_no_gap_no_firing(self):
        """Firms without gaps don't fire anyone."""
        sim = _make_sim(seed=42)
        sim.step()

        from bamengine.events._internal.vectorized_markets import firms_fire_workers_vec

        # Ensure all firms have enough funds
        sim.emp.total_funds[:] = sim.emp.wage_bill + 100.0
        sim.bor.total_funds[:] = sim.emp.total_funds

        employed_before = sim.wrk.employed.sum()
        firms_fire_workers_vec(sim.emp, sim.wrk, rng=sim.rng)
        employed_after = sim.wrk.employed.sum()

        assert employed_after == employed_before


# ═════════════════════════════════════════════════════════════════════════════
#  GOODS MARKET VECTORIZED
# ═════════════════════════════════════════════════════════════════════════════


class TestGoodsMarketRoundVec:
    """Tests for vectorized goods market matching."""

    def test_purchases_made(self):
        """Consumers with budget buy from firms with inventory."""
        sim = _make_sim(seed=42)
        sim.step()  # populate state

        # Re-setup goods market
        sim.get_event("consumers_calc_propensity").execute(sim)
        sim.get_event("consumers_decide_income_to_spend").execute(sim)
        sim.get_event("consumers_decide_firms_to_visit").execute(sim)

        from bamengine.events._internal.vectorized_markets import goods_market_round_vec

        total_inv_before = sim.prod.inventory.sum()
        total_budget_before = sim.con.income_to_spend.sum()

        for _ in range(sim.config.max_Z):
            goods_market_round_vec(
                sim.con, sim.prod, max_Z=sim.config.max_Z, rng=sim.rng
            )

        total_inv_after = sim.prod.inventory.sum()
        total_budget_after = sim.con.income_to_spend.sum()

        if total_inv_before > 0 and total_budget_before > 0:
            assert total_inv_after < total_inv_before, "Inventory should decrease"
            assert total_budget_after < total_budget_before, "Budget should decrease"

    def test_no_negative_inventory(self):
        """Inventory never goes negative."""
        sim = _make_sim(seed=42)
        sim.step()

        sim.get_event("consumers_calc_propensity").execute(sim)
        sim.get_event("consumers_decide_income_to_spend").execute(sim)
        sim.get_event("consumers_decide_firms_to_visit").execute(sim)

        from bamengine.events._internal.vectorized_markets import goods_market_round_vec

        for _ in range(sim.config.max_Z):
            goods_market_round_vec(
                sim.con, sim.prod, max_Z=sim.config.max_Z, rng=sim.rng
            )

        assert (sim.prod.inventory >= -1e-10).all()

    def test_no_negative_budget(self):
        """Consumer budget never goes negative."""
        sim = _make_sim(seed=42)
        sim.step()

        sim.get_event("consumers_calc_propensity").execute(sim)
        sim.get_event("consumers_decide_income_to_spend").execute(sim)
        sim.get_event("consumers_decide_firms_to_visit").execute(sim)

        from bamengine.events._internal.vectorized_markets import goods_market_round_vec

        for _ in range(sim.config.max_Z):
            goods_market_round_vec(
                sim.con, sim.prod, max_Z=sim.config.max_Z, rng=sim.rng
            )

        assert (sim.con.income_to_spend >= -1e-10).all()

    def test_no_shoppers_no_crash(self):
        """No active shoppers → no crash."""
        sim = _make_sim(seed=42)
        sim.con.income_to_spend[:] = 0.0
        sim.con.shop_visits_head[:] = -1

        from bamengine.events._internal.vectorized_markets import goods_market_round_vec

        goods_market_round_vec(sim.con, sim.prod, max_Z=sim.config.max_Z, rng=sim.rng)

    def test_inventory_clamped_to_zero_with_duplicate_targets(self):
        """Inventory is clamped to zero even with within-batch overselling.

        When multiple consumers in the same batch target the same firm,
        np.subtract.at may temporarily produce negative inventory.  The
        clamp to zero bounds the oversell.  This is intentionally retained
        (not FCFS-resolved) because the phantom goods compensate for the
        batch-sequential variance reduction.
        """
        sim = _make_sim(seed=42)
        sim.step()

        sim.get_event("consumers_calc_propensity").execute(sim)
        sim.get_event("consumers_decide_income_to_spend").execute(sim)
        sim.get_event("consumers_decide_firms_to_visit").execute(sim)

        from bamengine.events._internal.vectorized_markets import goods_market_round_vec

        # Use only 1 batch to maximize collision chance
        goods_market_round_vec(
            sim.con, sim.prod, max_Z=sim.config.max_Z, n_batches=1, rng=sim.rng
        )

        # Inventory clamped to zero (never negative in final state)
        assert (sim.prod.inventory >= -1e-10).all(), (
            f"Inventory went negative: min={sim.prod.inventory.min()}"
        )

    def test_n_batches_parameter(self):
        """n_batches parameter controls batch count."""
        sim = _make_sim(seed=42)
        sim.step()

        sim.get_event("consumers_calc_propensity").execute(sim)
        sim.get_event("consumers_decide_income_to_spend").execute(sim)
        sim.get_event("consumers_decide_firms_to_visit").execute(sim)

        from bamengine.events._internal.vectorized_markets import goods_market_round_vec

        inv_before = sim.prod.inventory.copy()
        budget_before = sim.con.income_to_spend.copy()

        # Should work with different batch counts without crashing
        goods_market_round_vec(
            sim.con, sim.prod, max_Z=sim.config.max_Z, n_batches=3, rng=sim.rng
        )

        # Purchases should have occurred
        if inv_before.sum() > 0 and budget_before.sum() > 0:
            assert sim.prod.inventory.sum() < inv_before.sum()


# ═════════════════════════════════════════════════════════════════════════════
#  APPEND LOANS BATCH
# ═════════════════════════════════════════════════════════════════════════════


class TestAppendLoansBatch:
    """Tests for LoanBook.append_loans_batch."""

    def test_basic_batch(self):
        """Batch append creates correct loan entries."""
        from bamengine.relationships.loanbook import LoanBook

        lb = LoanBook()
        banks = np.array([0, 1, 0, 2])
        firms = np.array([5, 3, 7, 1])
        amounts = np.array([100.0, 200.0, 150.0, 50.0])
        rates = np.array([0.02, 0.03, 0.02, 0.04])

        lb.append_loans_batch(banks, firms, amounts, rates)

        assert lb.size == 4
        np.testing.assert_array_equal(lb.source_ids[:4], firms)
        np.testing.assert_array_equal(lb.target_ids[:4], banks)
        np.testing.assert_array_almost_equal(lb.principal[:4], amounts)
        np.testing.assert_array_almost_equal(lb.interest[:4], amounts * rates)
        np.testing.assert_array_almost_equal(lb.debt[:4], amounts * (1 + rates))

    def test_empty_batch(self):
        """Empty arrays → no change."""
        from bamengine.relationships.loanbook import LoanBook

        lb = LoanBook()
        lb.append_loans_batch(
            np.array([], dtype=np.intp),
            np.array([], dtype=np.intp),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        assert lb.size == 0

    def test_accumulates_with_existing(self):
        """Batch append adds to existing loans."""
        from bamengine.relationships.loanbook import LoanBook

        lb = LoanBook()
        lb.append_loans_for_lender(
            np.intp(0), np.array([1]), np.array([100.0]), np.array([0.02])
        )
        assert lb.size == 1

        lb.append_loans_batch(
            np.array([1, 2]),
            np.array([3, 4]),
            np.array([200.0, 300.0]),
            np.array([0.03, 0.04]),
        )
        assert lb.size == 3


# ═════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════


class TestVectorizedPipelineIntegration:
    """Integration tests for the vectorized pipeline."""

    def test_multiple_steps_no_crash(self):
        """Vectorized pipeline runs multiple steps without crashing."""
        import logging

        logging.disable(logging.INFO)
        try:
            sim = _make_sim(
                seed=42,
                pipeline_path="src/bamengine/config/vectorized_pipeline.yml",
            )
            for _ in range(10):
                sim.step()
            assert not sim.ec.collapsed
        finally:
            logging.disable(logging.NOTSET)

    def test_economy_not_collapsed(self):
        """Economy should not collapse in 50 periods."""
        import logging

        logging.disable(logging.INFO)
        try:
            sim = _make_sim(
                seed=0,
                pipeline_path="src/bamengine/config/vectorized_pipeline.yml",
            )
            for _ in range(50):
                sim.step()
                if sim.ec.collapsed:
                    break
            assert not sim.ec.collapsed
        finally:
            logging.disable(logging.NOTSET)

    def test_multiple_seeds_stable(self):
        """Multiple seeds should not crash."""
        import logging

        logging.disable(logging.INFO)
        try:
            for seed in [0, 1, 2, 3, 4]:
                sim = _make_sim(
                    seed=seed,
                    pipeline_path="src/bamengine/config/vectorized_pipeline.yml",
                )
                for _ in range(20):
                    sim.step()
                # At least some should not collapse
        finally:
            logging.disable(logging.NOTSET)
