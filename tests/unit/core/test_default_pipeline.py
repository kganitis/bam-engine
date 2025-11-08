"""Tests for default pipeline builder."""

import bamengine.events  # noqa: F401 - needed to register events
from bamengine.core.default_pipeline import create_default_pipeline
from bamengine.core.pipeline import RepeatedEvent


def test_create_default_pipeline():
    """Can create default BAM pipeline."""
    max_M, max_H, max_Z = 5, 3, 2
    pipeline = create_default_pipeline(max_M=max_M, max_H=max_H, max_Z=max_Z)

    # Should have base events (34) + interleaved market rounds
    # Market rounds: 2*max_M + 2*max_H + max_Z
    expected_length = 34 + 2 * max_M + 2 * max_H + max_Z
    assert len(pipeline) == expected_length  # 34 + 2*5 + 2*3 + 2 = 52


def test_default_pipeline_interleaved_market_rounds():
    """Default pipeline has interleaved market rounds."""
    pipeline = create_default_pipeline(max_M=3, max_H=2, max_Z=2)

    # Find all event names
    event_names = [e.name for e in pipeline.events]

    # Check labor market interleaving: send-hire-send-hire-send-hire
    send_indices = [
        i for i, name in enumerate(event_names) if name == "workers_send_one_round"
    ]
    hire_indices = [
        i for i, name in enumerate(event_names) if name == "firms_hire_workers"
    ]

    # Should have 3 of each
    assert len(send_indices) == 3
    assert len(hire_indices) == 3

    # They should alternate: send[i] < hire[i] < send[i+1]
    for i in range(len(send_indices) - 1):
        assert send_indices[i] < hire_indices[i] < send_indices[i + 1]


def test_default_pipeline_order_matches_simulation():
    """Default pipeline matches Simulation.step() order."""
    pipeline = create_default_pipeline(max_M=5, max_H=3, max_Z=2)

    # First event should be planning
    assert pipeline.events[0].name == "firms_decide_desired_production"

    # Last event should be unemployment calc
    assert pipeline.events[-1].name == "calc_unemployment_rate"


def test_default_pipeline_first_event():
    """Default pipeline starts with expected first event."""
    pipeline = create_default_pipeline(max_M=4, max_H=2, max_Z=2)

    # After topological sort, first event should still be from planning
    # (all planning events have no dependencies)
    first_event_name = pipeline.events[0].name
    planning_events = {
        "firms_decide_desired_production",
        "calc_annual_inflation_rate",
        "firms_decide_vacancies",
    }
    # First event should be a planning event (no dependencies)
    assert any(
        first_event_name in name for name in planning_events
    ) or first_event_name.startswith("firms_")


def test_default_pipeline_has_correct_market_round_counts():
    """Market rounds appear correct number of times."""
    max_M, max_H, max_Z = 7, 4, 3
    pipeline = create_default_pipeline(max_M=max_M, max_H=max_H, max_Z=max_Z)

    event_names = [e.name for e in pipeline.events]

    # Count occurrences of each market round event
    assert event_names.count("workers_send_one_round") == max_M
    assert event_names.count("firms_hire_workers") == max_M
    assert event_names.count("firms_send_one_loan_app") == max_H
    assert event_names.count("banks_provide_loans") == max_H
    assert event_names.count("consumers_shop_one_round") == max_Z


def test_default_pipeline_contains_all_phases():
    """Default pipeline contains events from all market phases."""
    pipeline = create_default_pipeline(max_M=4, max_H=2, max_Z=2)

    event_names = {e.name for e in pipeline.events}

    # Check for representative events from each phase
    assert "firms_decide_desired_production" in event_names  # Planning
    assert "workers_send_one_round" in event_names  # Labor
    assert "banks_provide_loans" in event_names  # Credit
    assert "firms_run_production" in event_names  # Production
    assert "consumers_shop_one_round" in event_names  # Goods
    assert "firms_collect_revenue" in event_names  # Revenue
    assert "mark_bankrupt_firms" in event_names  # Bankruptcy
    assert "calc_unemployment_rate" in event_names  # End of period


def test_default_pipeline_executes_without_error():
    """Default pipeline can be executed on a simulation."""
    from bamengine.simulation import Simulation

    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    pipeline = create_default_pipeline(max_M=2, max_H=2, max_Z=2)

    # Should execute without error
    pipeline.execute(sim)

    # Note: t is not incremented by pipeline.execute(), that's Simulation.step()'s job
    # Just verify pipeline executed without errors
