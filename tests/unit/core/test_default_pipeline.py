"""Tests for default pipeline builder."""

import bamengine.events  # noqa: F401 - needed to register events
from bamengine.core.pipeline import create_default_pipeline


def test_create_default_pipeline():
    """Can create default BAM pipeline."""
    max_M, max_H, max_Z = 5, 3, 2
    pipeline = create_default_pipeline(max_M=max_M, max_H=max_H, max_Z=max_Z)

    # Fixed pipeline length after switching to cascade matching events:
    # Phase 1: 6, Phase 2: 6, Phase 3: 7, Phase 4: 5, Phase 5: 5,
    # Phase 6: 3, Phase 7: 3, Phase 8: 2 = 37 total
    # (No longer depends on max_M or max_H — cascade events are single events)
    assert len(pipeline) == 37


def test_default_pipeline_cascade_matching_events():
    """Default pipeline uses cascade matching (single events, not interleaved)."""
    pipeline = create_default_pipeline(max_M=3, max_H=2, max_Z=2)

    event_names = [e.name for e in pipeline.events]

    # Cascade labor matching is a single event
    assert event_names.count("workers_apply_to_firms") == 1
    # Cascade credit matching is a single event
    assert event_names.count("firms_apply_for_loans") == 1

    # Legacy interleaved events should NOT appear in default pipeline
    assert event_names.count("workers_send_one_round") == 0
    assert event_names.count("firms_hire_workers") == 0
    assert event_names.count("firms_send_one_loan_app") == 0
    assert event_names.count("banks_provide_loans") == 0

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


def test_default_pipeline_length_independent_of_market_params():
    """Pipeline length is the same regardless of max_M/max_H/max_Z values."""
    p1 = create_default_pipeline(max_M=2, max_H=2, max_Z=2)
    p2 = create_default_pipeline(max_M=7, max_H=4, max_Z=3)
    p3 = create_default_pipeline(max_M=1, max_H=1, max_Z=1)

    assert len(p1) == len(p2) == len(p3) == 37


def test_default_pipeline_contains_all_phases():
    """Default pipeline contains events from all market phases."""
    pipeline = create_default_pipeline(max_M=4, max_H=2, max_Z=2)

    event_names = {e.name for e in pipeline.events}

    # Check for representative events from each phase
    assert "firms_decide_desired_production" in event_names  # Planning
    assert "workers_apply_to_firms" in event_names  # Labor
    assert "firms_apply_for_loans" in event_names  # Credit
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


def test_interleaved_labor_matching_pipeline():
    """Interleaved labor matching replaces cascade event with send/hire rounds."""
    max_M = 3
    pipeline = create_default_pipeline(
        max_M=max_M, max_H=2, max_Z=2, labor_matching="interleaved"
    )

    event_names = [e.name for e in pipeline.events]

    # Cascade event should be gone
    assert "workers_apply_to_firms" not in event_names

    # Legacy interleaved events should appear max_M times each
    assert event_names.count("workers_send_one_round") == max_M
    assert event_names.count("firms_hire_workers") == max_M

    # Credit side should still be cascade
    assert event_names.count("firms_apply_for_loans") == 1
    assert event_names.count("firms_send_one_loan_app") == 0

    # Pipeline should be longer than default (37 - 1 + 2*max_M)
    assert len(pipeline) == 37 - 1 + 2 * max_M


def test_interleaved_credit_matching_pipeline():
    """Interleaved credit matching replaces cascade event with app/provide rounds."""
    max_H = 3
    pipeline = create_default_pipeline(
        max_M=4, max_H=max_H, max_Z=2, credit_matching="interleaved"
    )

    event_names = [e.name for e in pipeline.events]

    # Cascade event should be gone
    assert "firms_apply_for_loans" not in event_names

    # Legacy interleaved events should appear max_H times each
    assert event_names.count("firms_send_one_loan_app") == max_H
    assert event_names.count("banks_provide_loans") == max_H

    # Labor side should still be cascade
    assert event_names.count("workers_apply_to_firms") == 1
    assert event_names.count("workers_send_one_round") == 0

    # Pipeline should be longer than default (37 - 1 + 2*max_H)
    assert len(pipeline) == 37 - 1 + 2 * max_H


def test_both_interleaved_matching_pipeline():
    """Both labor and credit interleaved expands both market phases."""
    max_M, max_H = 4, 2
    pipeline = create_default_pipeline(
        max_M=max_M,
        max_H=max_H,
        max_Z=2,
        labor_matching="interleaved",
        credit_matching="interleaved",
    )

    event_names = [e.name for e in pipeline.events]

    # Both cascade events gone
    assert "workers_apply_to_firms" not in event_names
    assert "firms_apply_for_loans" not in event_names

    # Both sets of legacy events present
    assert event_names.count("workers_send_one_round") == max_M
    assert event_names.count("firms_hire_workers") == max_M
    assert event_names.count("firms_send_one_loan_app") == max_H
    assert event_names.count("banks_provide_loans") == max_H

    # 37 - 2 cascade + 2*max_M labor + 2*max_H credit
    assert len(pipeline) == 37 - 2 + 2 * max_M + 2 * max_H


def test_interleaved_labor_event_order():
    """Interleaved labor events appear after workers_decide_firms_to_apply."""
    pipeline = create_default_pipeline(
        max_M=2, max_H=2, max_Z=2, labor_matching="interleaved"
    )
    names = [e.name for e in pipeline.events]

    # Find the insertion anchor
    decide_idx = names.index("workers_decide_firms_to_apply")

    # The interleaved events should come right after
    assert names[decide_idx + 1] == "workers_send_one_round"
    assert names[decide_idx + 2] == "firms_hire_workers"
    assert names[decide_idx + 3] == "workers_send_one_round"
    assert names[decide_idx + 4] == "firms_hire_workers"


def test_interleaved_pipeline_length_depends_on_market_params():
    """Interleaved pipeline length scales with max_M and max_H."""
    p1 = create_default_pipeline(
        max_M=2,
        max_H=2,
        max_Z=2,
        labor_matching="interleaved",
        credit_matching="interleaved",
    )
    p2 = create_default_pipeline(
        max_M=5,
        max_H=4,
        max_Z=2,
        labor_matching="interleaved",
        credit_matching="interleaved",
    )

    # Different max_M/max_H → different lengths
    assert len(p1) != len(p2)
    assert len(p1) == 37 - 2 + 2 * 2 + 2 * 2  # 43
    assert len(p2) == 37 - 2 + 2 * 5 + 2 * 4  # 53
