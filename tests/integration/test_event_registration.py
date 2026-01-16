"""Integration tests verifying all events are registered.

This test suite ensures that all Event classes are properly registered
in the global event registry via the __init_subclass__ hook.
"""

from bamengine.core.registry import list_events


def test_all_planning_events_registered():
    """All planning events are registered."""
    events = list_events()

    assert "firms_decide_desired_production" in events
    assert "firms_calc_breakeven_price" in events
    assert "firms_adjust_price" in events
    assert "firms_decide_desired_labor" in events
    assert "firms_decide_vacancies" in events


def test_all_labor_market_events_registered():
    """All labor market events are registered."""
    events = list_events()

    assert "calc_annual_inflation_rate" in events
    assert "adjust_minimum_wage" in events
    assert "firms_decide_wage_offer" in events
    assert "workers_decide_firms_to_apply" in events
    assert "workers_send_one_round" in events
    assert "firms_hire_workers" in events
    assert "firms_calc_wage_bill" in events


def test_all_credit_market_events_registered():
    """All credit market events are registered."""
    events = list_events()

    assert "banks_decide_credit_supply" in events
    assert "banks_decide_interest_rate" in events
    assert "firms_decide_credit_demand" in events
    assert "firms_calc_financial_fragility" in events
    assert "firms_prepare_loan_applications" in events
    assert "firms_send_one_loan_app" in events
    assert "banks_provide_loans" in events
    assert "firms_fire_workers" in events


def test_all_production_events_registered():
    """All production events are registered."""
    events = list_events()

    assert "firms_pay_wages" in events
    assert "workers_receive_wage" in events
    assert "firms_run_production" in events
    assert "workers_update_contracts" in events


def test_all_goods_market_events_registered():
    """All goods market events are registered."""
    events = list_events()

    assert "consumers_calc_propensity" in events
    assert "consumers_decide_income_to_spend" in events
    assert "consumers_decide_firms_to_visit" in events
    assert "consumers_shop_one_round" in events
    assert "consumers_finalize_purchases" in events


def test_all_revenue_events_registered():
    """All revenue events are registered."""
    events = list_events()

    assert "firms_collect_revenue" in events
    assert "firms_validate_debt_commitments" in events
    assert "firms_pay_dividends" in events


def test_all_bankruptcy_events_registered():
    """All bankruptcy events are registered."""
    events = list_events()

    assert "firms_update_net_worth" in events
    assert "mark_bankrupt_firms" in events
    assert "mark_bankrupt_banks" in events
    assert "spawn_replacement_firms" in events
    assert "spawn_replacement_banks" in events


def test_all_economy_stats_events_registered():
    """All economy statistics events are registered."""
    events = list_events()

    assert "update_avg_mkt_price" in events
    assert "calc_unemployment_rate" in events


def test_no_duplicate_event_names():
    """Ensure no duplicate event names in registry."""
    events = list_events()

    # list_events returns unique keys, but let's verify logic
    assert len(events) == len(set(events)), "Duplicate event names found"


def test_all_event_names_lowercase_snake_case():
    """All event names follow snake_case convention."""
    events = list_events()

    for event_name in events:
        # Should be lowercase
        assert event_name.islower(), f"Event {event_name} not lowercase"
        # Should use underscores (no hyphens or camelCase)
        assert "-" not in event_name, f"Event {event_name} contains hyphen"
        assert not any(c.isupper() for c in event_name), (
            f"Event {event_name} has uppercase"
        )
