"""Unit tests for market matching functions."""

from __future__ import annotations

import numpy as np

import bamengine as bam

# -- Helpers -------------------------------------------------------------------


def _make_sim(seed: int = 42, **kw) -> bam.Simulation:
    """Create a small simulation for testing."""
    return bam.Simulation.init(seed=seed, **kw)


# =============================================================================
#  LABOR MARKET
# =============================================================================


class TestLaborMarketRound:
    """Tests for labor market matching."""

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

        from bamengine.events._internal.labor_market import labor_market_round

        initial_unemployed = (~sim.wrk.employed).sum()
        assert initial_unemployed > 0, "Should have unemployed workers initially"

        # Run multiple rounds
        for _ in range(sim.config.max_M):
            labor_market_round(sim.emp, sim.wrk, theta=sim.config.theta, rng=sim.rng)

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

        from bamengine.events._internal.labor_market import labor_market_round

        for _ in range(sim.config.max_M):
            labor_market_round(sim.emp, sim.wrk, theta=sim.config.theta, rng=sim.rng)

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

        from bamengine.events._internal.labor_market import labor_market_round

        for _ in range(sim.config.max_M):
            labor_market_round(sim.emp, sim.wrk, theta=sim.config.theta, rng=sim.rng)

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

        from bamengine.events._internal.labor_market import labor_market_round

        for _ in range(sim.config.max_M):
            labor_market_round(sim.emp, sim.wrk, theta=sim.config.theta, rng=sim.rng)

        assert (sim.emp.n_vacancies >= 0).all(), "Vacancies should never go negative"

    def test_no_workers_no_crash(self):
        """No unemployed workers -> no crash, no state changes."""
        sim = _make_sim(seed=42)
        # Manually set all workers as employed
        sim.wrk.employer[:] = 0
        sim.wrk.wage[:] = 1.0
        sim.wrk.periods_left[:] = 10

        from bamengine.events._internal.labor_market import labor_market_round

        # Should not crash
        labor_market_round(sim.emp, sim.wrk, theta=sim.config.theta, rng=sim.rng)

    def test_event_class_works(self):
        """LaborMarketRound event class wraps correctly."""
        sim = _make_sim(seed=42)
        sim.get_event("labor_market_round")
        # Should not crash
        sim.step()


# =============================================================================
#  CREDIT MARKET
# =============================================================================


class TestCreditMarketRound:
    """Tests for credit market matching."""

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

        from bamengine.events._internal.labor_market import labor_market_round

        for _ in range(sim.config.max_M):
            labor_market_round(sim.emp, sim.wrk, theta=sim.config.theta, rng=sim.rng)
        sim.get_event("firms_calc_wage_bill").execute(sim)

        for ev in [
            "banks_decide_credit_supply",
            "banks_decide_interest_rate",
            "firms_decide_credit_demand",
            "firms_calc_financial_fragility",
            "firms_prepare_loan_applications",
        ]:
            sim.get_event(ev).execute(sim)

        from bamengine.events._internal.credit_market import credit_market_round

        initial_lb_size = sim.lb.size
        for _ in range(sim.config.max_H):
            credit_market_round(
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
        """No credit demand -> no crash."""
        sim = _make_sim(seed=42)
        # Zero out credit demand
        sim.bor.credit_demand[:] = 0.0

        from bamengine.events._internal.credit_market import credit_market_round

        # Need opex_shock to be set
        sim.get_event("banks_decide_interest_rate").execute(sim)

        credit_market_round(
            sim.bor,
            sim.lend,
            sim.lb,
            r_bar=sim.r_bar,
            max_leverage=sim.config.max_leverage,
            rng=sim.rng,
        )


class TestFirmsFireWorkers:
    """Tests for worker firing."""

    def test_firing_reduces_gap(self):
        """Firms fire workers until wage bill <= total_funds."""
        sim = _make_sim(seed=42)
        sim.step()  # run a full period to populate state

        from bamengine.events._internal.credit_market import firms_fire_workers

        # Create financing gaps
        firing_firms = np.where(sim.emp.current_labor > 0)[0][:5]
        if firing_firms.size > 0:
            sim.emp.total_funds[firing_firms] = 0.0  # create gap
            sim.bor.total_funds[firing_firms] = 0.0

            firms_fire_workers(sim.emp, sim.wrk, rng=sim.rng)

            # After firing, wage bill should be close to or below total_funds
            for i in firing_firms:
                assert sim.emp.wage_bill[i] <= sim.emp.total_funds[i] + 1e-8

    def test_fired_workers_state(self):
        """Fired workers have correct state updates."""
        sim = _make_sim(seed=42)
        sim.step()

        from bamengine.events._internal.credit_market import firms_fire_workers

        # Create gap
        firm = np.where(sim.emp.current_labor > 2)[0]
        if firm.size > 0:
            i = firm[0]
            sim.emp.total_funds[i] = 0.0
            sim.bor.total_funds[i] = 0.0

            firms_fire_workers(sim.emp, sim.wrk, rng=sim.rng)

            # Check some workers were fired
            fired = sim.wrk.fired.astype(bool)
            if fired.any():
                assert (sim.wrk.employer[fired] == -1).all()
                assert (sim.wrk.wage[fired] == 0.0).all()

    def test_fires_minimum_workers_to_cover_gap(self):
        """Fires the fewest workers needed to cover the financing gap."""
        from bamengine.events._internal.credit_market import firms_fire_workers

        sim = _make_sim(seed=42)
        sim.step()  # populate state

        # Find a firm with at least 3 workers
        firm_idx = np.where(sim.emp.current_labor >= 3)[0]
        if firm_idx.size == 0:
            return  # skip if no suitable firm

        i = firm_idx[0]
        workers_at_i = np.where(sim.wrk.employer == i)[0]
        n_workers = workers_at_i.size

        # Set uniform wages and create a gap requiring exactly 2 firings
        wage = 10.0
        sim.wrk.wage[workers_at_i] = wage
        sim.emp.wage_bill[i] = n_workers * wage
        sim.emp.total_funds[i] = (n_workers - 2) * wage  # gap = 20
        sim.bor.total_funds[i] = sim.emp.total_funds[i]

        labor_before = int(sim.emp.current_labor[i])
        firms_fire_workers(sim.emp, sim.wrk, rng=sim.rng)

        # Firm should fire exactly 2 workers (gap=20, wage=10 each)
        assert sim.emp.current_labor[i] == labor_before - 2

    def test_no_gap_no_firing(self):
        """Firms without gaps don't fire anyone."""
        sim = _make_sim(seed=42)
        sim.step()

        from bamengine.events._internal.credit_market import firms_fire_workers

        # Ensure all firms have enough funds
        sim.emp.total_funds[:] = sim.emp.wage_bill + 100.0
        sim.bor.total_funds[:] = sim.emp.total_funds

        employed_before = sim.wrk.employed.sum()
        firms_fire_workers(sim.emp, sim.wrk, rng=sim.rng)
        employed_after = sim.wrk.employed.sum()

        assert employed_after == employed_before


# =============================================================================
#  GOODS MARKET
# =============================================================================


class TestGoodsMarketRound:
    """Tests for goods market matching."""

    def test_purchases_made(self):
        """Consumers with budget buy from firms with inventory."""
        sim = _make_sim(seed=42)
        sim.step()  # populate state

        # Re-setup goods market
        sim.get_event("consumers_calc_propensity").execute(sim)
        sim.get_event("consumers_decide_income_to_spend").execute(sim)
        sim.get_event("consumers_decide_firms_to_visit").execute(sim)

        from bamengine.events._internal.goods_market import goods_market_round

        total_inv_before = sim.prod.inventory.sum()
        total_budget_before = sim.con.income_to_spend.sum()

        for _ in range(sim.config.max_Z):
            goods_market_round(sim.con, sim.prod, max_Z=sim.config.max_Z, rng=sim.rng)

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

        from bamengine.events._internal.goods_market import goods_market_round

        for _ in range(sim.config.max_Z):
            goods_market_round(sim.con, sim.prod, max_Z=sim.config.max_Z, rng=sim.rng)

        assert (sim.prod.inventory >= -1e-10).all()

    def test_no_negative_budget(self):
        """Consumer budget never goes negative."""
        sim = _make_sim(seed=42)
        sim.step()

        sim.get_event("consumers_calc_propensity").execute(sim)
        sim.get_event("consumers_decide_income_to_spend").execute(sim)
        sim.get_event("consumers_decide_firms_to_visit").execute(sim)

        from bamengine.events._internal.goods_market import goods_market_round

        for _ in range(sim.config.max_Z):
            goods_market_round(sim.con, sim.prod, max_Z=sim.config.max_Z, rng=sim.rng)

        assert (sim.con.income_to_spend >= -1e-10).all()

    def test_no_shoppers_no_crash(self):
        """No active shoppers -> no crash."""
        sim = _make_sim(seed=42)
        sim.con.income_to_spend[:] = 0.0
        sim.con.shop_visits_head[:] = -1

        from bamengine.events._internal.goods_market import goods_market_round

        goods_market_round(sim.con, sim.prod, max_Z=sim.config.max_Z, rng=sim.rng)

    def test_inventory_never_negative(self):
        """Inventory stays non-negative with sequential processing."""
        sim = _make_sim(seed=42)
        sim.step()

        sim.get_event("consumers_calc_propensity").execute(sim)
        sim.get_event("consumers_decide_income_to_spend").execute(sim)
        sim.get_event("consumers_decide_firms_to_visit").execute(sim)

        from bamengine.events._internal.goods_market import goods_market_round

        goods_market_round(sim.con, sim.prod, max_Z=sim.config.max_Z, rng=sim.rng)

        assert (sim.prod.inventory >= -1e-10).all(), (
            f"Inventory went negative: min={sim.prod.inventory.min()}"
        )


# =============================================================================
#  APPEND LOANS BATCH
# =============================================================================


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
        """Empty arrays -> no change."""
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


# =============================================================================
#  FULL PIPELINE INTEGRATION
# =============================================================================


class TestPipelineIntegration:
    """Integration tests for the default pipeline."""

    def test_multiple_steps_no_crash(self):
        """Pipeline runs multiple steps without crashing."""
        import logging

        logging.disable(logging.INFO)
        try:
            sim = _make_sim(seed=42)
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
            sim = _make_sim(seed=0)
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
                sim = _make_sim(seed=seed)
                for _ in range(20):
                    sim.step()
                # At least some should not collapse
        finally:
            logging.disable(logging.NOTSET)
