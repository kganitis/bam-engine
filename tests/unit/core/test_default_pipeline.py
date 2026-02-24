"""Tests for default pipeline builder."""

import bamengine.events  # noqa: F401 - needed to register events
from bamengine.core.pipeline import create_default_pipeline


def test_create_default_pipeline():
    """Can create default BAM pipeline."""
    max_M, max_H, max_Z = 5, 3, 2
    pipeline = create_default_pipeline(max_M=max_M, max_H=max_H, max_Z=max_Z)

    # Default is interleaved matching for both labor and credit.
    # Base cascade pipeline = 37 events, then swap:
    #   -2 cascade events + 2*max_M labor rounds + 2*max_H credit rounds
    expected = 37 - 2 + 2 * max_M + 2 * max_H
    assert len(pipeline) == expected


def test_default_pipeline_interleaved_matching_events():
    """Default pipeline uses interleaved matching (repeated rounds)."""
    max_M, max_H = 3, 2
    pipeline = create_default_pipeline(max_M=max_M, max_H=max_H, max_Z=2)

    event_names = [e.name for e in pipeline.events]

    # Interleaved labor: send/hire repeated max_M times
    assert event_names.count("workers_send_one_round") == max_M
    assert event_names.count("firms_hire_workers") == max_M

    # Interleaved credit: app/provide repeated max_H times
    assert event_names.count("firms_send_one_loan_app") == max_H
    assert event_names.count("banks_provide_loans") == max_H

    # Cascade events should NOT appear in default pipeline
    assert event_names.count("workers_apply_to_firms") == 0
    assert event_names.count("firms_apply_for_loans") == 0

    # consumers_shop_sequential is still a single event
    assert event_names.count("consumers_shop_sequential") == 1


def test_default_pipeline_order_matches_simulation():
    """Default pipeline matches Simulation.step() order."""
    pipeline = create_default_pipeline(max_M=5, max_H=3, max_Z=2)

    # First event should be planning
    assert pipeline.events[0].name == "firms_decide_desired_production"

    # Last event should be spawn_replacement_banks (calc_unemployment_rate deprecated)
    assert pipeline.events[-1].name == "spawn_replacement_banks"


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


def test_default_pipeline_contains_all_phases():
    """Default pipeline contains events from all market phases."""
    pipeline = create_default_pipeline(max_M=4, max_H=2, max_Z=2)

    event_names = {e.name for e in pipeline.events}

    # Check for representative events from each phase
    assert "firms_decide_desired_production" in event_names  # Planning
    assert "workers_send_one_round" in event_names  # Labor (interleaved)
    assert "firms_send_one_loan_app" in event_names  # Credit (interleaved)
    assert "firms_run_production" in event_names  # Production
    assert "consumers_shop_sequential" in event_names  # Goods
    assert "firms_collect_revenue" in event_names  # Revenue
    assert "mark_bankrupt_firms" in event_names  # Bankruptcy
    assert "spawn_replacement_banks" in event_names  # Entry (end of period)


def test_default_pipeline_executes_without_error():
    """Default pipeline can be executed on a simulation."""
    from bamengine.simulation import Simulation

    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    pipeline = create_default_pipeline(max_M=2, max_H=2, max_Z=2)

    # Should execute without error
    pipeline.execute(sim)

    # Note: t is not incremented by pipeline.execute(), that's Simulation.step()'s job
    # Just verify pipeline executed without errors


def test_interleaved_labor_event_order():
    """Interleaved labor events appear after workers_decide_firms_to_apply."""
    pipeline = create_default_pipeline(max_M=2, max_H=2, max_Z=2)
    names = [e.name for e in pipeline.events]

    # Find the insertion anchor
    decide_idx = names.index("workers_decide_firms_to_apply")

    # The interleaved events should come right after
    assert names[decide_idx + 1] == "workers_send_one_round"
    assert names[decide_idx + 2] == "firms_hire_workers"
    assert names[decide_idx + 3] == "workers_send_one_round"
    assert names[decide_idx + 4] == "firms_hire_workers"


def test_default_pipeline_length_depends_on_market_params():
    """Default (interleaved) pipeline length scales with max_M and max_H."""
    p1 = create_default_pipeline(max_M=2, max_H=2, max_Z=2)
    p2 = create_default_pipeline(max_M=5, max_H=4, max_Z=2)

    # Different max_M/max_H → different lengths
    assert len(p1) != len(p2)
    assert len(p1) == 37 - 2 + 2 * 2 + 2 * 2  # 43
    assert len(p2) == 37 - 2 + 2 * 5 + 2 * 4  # 53
