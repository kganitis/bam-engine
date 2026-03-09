"""Tests for default pipeline builder."""

import bamengine.events  # noqa: F401 - needed to register events
from bamengine.core.pipeline import create_default_pipeline


def test_create_default_pipeline():
    """Can create default BAM pipeline."""
    max_M, max_H, max_Z = 5, 3, 2
    pipeline = create_default_pipeline(max_M=max_M, max_H=max_H, max_Z=max_Z)

    # Default pipeline = 35 base events + max_M labor rounds + max_H credit rounds
    expected = 35 + max_M + max_H
    assert len(pipeline) == expected


def test_default_pipeline_market_round_counts():
    """Default pipeline has correct number of market round events."""
    max_M, max_H = 3, 2
    pipeline = create_default_pipeline(max_M=max_M, max_H=max_H, max_Z=2)

    event_names = [e.name for e in pipeline.events]

    # Labor rounds repeated max_M times
    assert event_names.count("labor_market_round") == max_M

    # Credit rounds repeated max_H times
    assert event_names.count("credit_market_round") == max_H

    # Goods market round is a single event
    assert event_names.count("goods_market_round") == 1


def test_default_pipeline_order_matches_simulation():
    """Default pipeline matches Simulation.step() order."""
    pipeline = create_default_pipeline(max_M=5, max_H=3, max_Z=2)

    # First event should be planning
    assert pipeline.events[0].name == "firms_decide_desired_production"

    # Last event should be spawn_replacement_banks
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
    assert "labor_market_round" in event_names  # Labor
    assert "credit_market_round" in event_names  # Credit
    assert "firms_run_production" in event_names  # Production
    assert "goods_market_round" in event_names  # Goods
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


def test_default_pipeline_labor_round_order():
    """Labor market rounds appear after workers_decide_firms_to_apply."""
    pipeline = create_default_pipeline(max_M=2, max_H=2, max_Z=2)
    names = [e.name for e in pipeline.events]

    # Find the insertion anchor
    decide_idx = names.index("workers_decide_firms_to_apply")

    # The labor market rounds should come right after
    assert names[decide_idx + 1] == "labor_market_round"
    assert names[decide_idx + 2] == "labor_market_round"


def test_default_pipeline_length_depends_on_market_params():
    """Default pipeline length scales with max_M and max_H."""
    p1 = create_default_pipeline(max_M=2, max_H=2, max_Z=2)
    p2 = create_default_pipeline(max_M=5, max_H=4, max_Z=2)

    # Different max_M/max_H -> different lengths
    assert len(p1) != len(p2)
    assert len(p1) == 35 + 2 + 2  # 39
    assert len(p2) == 35 + 5 + 4  # 44
