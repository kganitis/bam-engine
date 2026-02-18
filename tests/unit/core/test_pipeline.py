"""Unit tests for Pipeline class."""

from typing import cast

import numpy as np
import pytest

from bamengine.core.pipeline import Pipeline
from bamengine.events.planning import FirmsDecideDesiredProduction
from bamengine.events.production import FirmsAdjustPrice, FirmsCalcBreakevenPrice


def test_pipeline_from_event_list_basic():
    """Pipeline can be built from event name list."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_calc_breakeven_price",
            "firms_adjust_price",
        ]
    )

    assert len(pipeline) == 3
    assert pipeline.events[0].name == "firms_decide_desired_production"


def test_pipeline_preserves_order():
    """Pipeline preserves the exact order of events provided."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_adjust_price",  # Out of logical order - but preserved
            "firms_decide_desired_production",
            "firms_calc_breakeven_price",
        ]
    )

    # Order should be preserved exactly as given
    assert pipeline.events[0].name == "firms_adjust_price"
    assert pipeline.events[1].name == "firms_decide_desired_production"
    assert pipeline.events[2].name == "firms_calc_breakeven_price"


def test_pipeline_insert_after():
    """Can insert event after specified event."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_adjust_price",
        ]
    )

    pipeline.insert_after("firms_decide_desired_production", FirmsCalcBreakevenPrice)

    assert len(pipeline) == 3
    assert pipeline.events[1].name == "firms_calc_breakeven_price"


def test_pipeline_insert_after_by_name():
    """Can insert event by name after specified event."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_adjust_price",
        ]
    )

    pipeline.insert_after(
        "firms_decide_desired_production", "firms_calc_breakeven_price"
    )

    assert len(pipeline) == 3
    assert pipeline.events[1].name == "firms_calc_breakeven_price"


def test_pipeline_insert_after_not_found():
    """insert_after raises error if target event not found."""
    pipeline = Pipeline.from_event_list(["firms_decide_desired_production"])

    with pytest.raises(ValueError, match="not found in pipeline"):
        pipeline.insert_after("nonexistent_event", FirmsAdjustPrice)


def test_pipeline_insert_after_list():
    """Can insert multiple events after specified event using a list."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_adjust_price",
        ]
    )

    # Insert a list of events
    pipeline.insert_after(
        "firms_decide_desired_production",
        [
            "firms_calc_breakeven_price",
            "update_avg_mkt_price",
        ],
    )

    assert len(pipeline) == 4
    # Events should be inserted in order
    assert pipeline.events[0].name == "firms_decide_desired_production"
    assert pipeline.events[1].name == "firms_calc_breakeven_price"
    assert pipeline.events[2].name == "update_avg_mkt_price"
    assert pipeline.events[3].name == "firms_adjust_price"


def test_pipeline_insert_after_list_preserves_order():
    """insert_after with list maintains the list order."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
        ]
    )

    # Insert multiple events
    pipeline.insert_after(
        "firms_decide_desired_production",
        [
            "firms_calc_breakeven_price",
            "firms_adjust_price",
            "update_avg_mkt_price",
        ],
    )

    # All events should be in the exact order specified
    assert pipeline.events[1].name == "firms_calc_breakeven_price"
    assert pipeline.events[2].name == "firms_adjust_price"
    assert pipeline.events[3].name == "update_avg_mkt_price"


def test_pipeline_insert_after_empty_list():
    """insert_after with empty list is a no-op."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_adjust_price",
        ]
    )

    # Insert empty list
    pipeline.insert_after("firms_decide_desired_production", [])

    # Pipeline should be unchanged
    assert len(pipeline) == 2
    assert pipeline.events[0].name == "firms_decide_desired_production"
    assert pipeline.events[1].name == "firms_adjust_price"


def test_pipeline_insert_after_single_event_still_works():
    """insert_after with single event (not list) still works (regression)."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_adjust_price",
        ]
    )

    # Single event, not a list
    pipeline.insert_after(
        "firms_decide_desired_production", "firms_calc_breakeven_price"
    )

    assert len(pipeline) == 3
    assert pipeline.events[1].name == "firms_calc_breakeven_price"


def test_pipeline_insert_before():
    """Can insert event before specified event."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_adjust_price",
        ],
    )

    pipeline.insert_before("firms_adjust_price", FirmsCalcBreakevenPrice)

    assert len(pipeline) == 3
    assert pipeline.events[0].name == "firms_decide_desired_production"
    assert pipeline.events[1].name == "firms_calc_breakeven_price"
    assert pipeline.events[2].name == "firms_adjust_price"


def test_pipeline_insert_before_by_name():
    """Can insert event by name before specified event."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_adjust_price",
        ],
    )

    pipeline.insert_before("firms_adjust_price", "firms_calc_breakeven_price")

    assert len(pipeline) == 3
    assert pipeline.events[1].name == "firms_calc_breakeven_price"


def test_pipeline_insert_before_not_found():
    """insert_before raises error if target event not found."""
    pipeline = Pipeline.from_event_list(
        ["firms_decide_desired_production"],
    )

    with pytest.raises(ValueError, match="not found in pipeline"):
        pipeline.insert_before("nonexistent_event", FirmsAdjustPrice)


def test_pipeline_insert_before_list():
    """Can insert multiple events before specified event using a list."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_adjust_price",
        ],
    )

    # Insert a list of events before firms_adjust_price
    pipeline.insert_before(
        "firms_adjust_price",
        [
            "firms_calc_breakeven_price",
            "update_avg_mkt_price",
        ],
    )

    assert len(pipeline) == 4
    # Events should be inserted in order before the target
    assert pipeline.events[0].name == "firms_decide_desired_production"
    assert pipeline.events[1].name == "firms_calc_breakeven_price"
    assert pipeline.events[2].name == "update_avg_mkt_price"
    assert pipeline.events[3].name == "firms_adjust_price"


def test_pipeline_insert_before_at_start():
    """Can insert event before the first event in pipeline."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_adjust_price",
        ],
    )

    pipeline.insert_before(
        "firms_decide_desired_production", "firms_calc_breakeven_price"
    )

    assert len(pipeline) == 3
    assert pipeline.events[0].name == "firms_calc_breakeven_price"
    assert pipeline.events[1].name == "firms_decide_desired_production"
    assert pipeline.events[2].name == "firms_adjust_price"


def test_pipeline_remove():
    """Can remove event from pipeline."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_calc_breakeven_price",
            "firms_adjust_price",
        ]
    )

    pipeline.remove("firms_calc_breakeven_price")

    assert len(pipeline) == 2
    assert "firms_calc_breakeven_price" not in [e.name for e in pipeline.events]


def test_pipeline_remove_not_found():
    """remove raises error if event not found."""
    pipeline = Pipeline.from_event_list(["firms_decide_desired_production"])

    with pytest.raises(ValueError, match="not found in pipeline"):
        pipeline.remove("nonexistent_event")


def test_pipeline_replace():
    """Can replace event in pipeline."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_calc_breakeven_price",
        ]
    )

    pipeline.replace("firms_calc_breakeven_price", FirmsAdjustPrice)

    assert len(pipeline) == 2
    assert pipeline.events[1].name == "firms_adjust_price"


def test_pipeline_replace_by_name():
    """Can replace event by name."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_calc_breakeven_price",
        ]
    )

    pipeline.replace("firms_calc_breakeven_price", "firms_adjust_price")

    assert len(pipeline) == 2
    assert pipeline.events[1].name == "firms_adjust_price"


def test_pipeline_replace_not_found():
    """replace raises error if old event not found."""
    pipeline = Pipeline.from_event_list(["firms_decide_desired_production"])

    with pytest.raises(ValueError, match="not found in pipeline"):
        pipeline.replace("nonexistent_event", FirmsAdjustPrice)


def test_pipeline_execute():
    """Pipeline executes all events in order."""
    from bamengine.simulation import Simulation

    sim = Simulation.init(n_firms=5, n_households=20, seed=42)
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
        ]
    )

    # Store initial state
    initial_desired_production = sim.prod.desired_production.copy()

    # Execute pipeline
    pipeline.execute(sim)

    # Verify event actually ran (desired production should change)
    # Since inventory is 0 and price >= avg, production should increase
    assert not np.all(sim.prod.desired_production == initial_desired_production)


def test_pipeline_repr():
    """Pipeline has informative repr."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_calc_breakeven_price",
        ]
    )

    assert repr(pipeline) == "Pipeline(n_events=2)"


def test_pipeline_len():
    """Pipeline len returns number of events."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_calc_breakeven_price",
            "firms_adjust_price",
        ]
    )

    assert len(pipeline) == 3


def test_pipeline_event_map_updated_on_insert():
    """Event map is updated when events are inserted."""
    pipeline = Pipeline.from_event_list(["firms_decide_desired_production"])

    pipeline.insert_after(
        "firms_decide_desired_production", "firms_calc_breakeven_price"
    )

    assert "firms_calc_breakeven_price" in pipeline._event_map


def test_pipeline_event_map_updated_on_remove():
    """Event map is updated when events are removed."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_calc_breakeven_price",
        ]
    )

    pipeline.remove("firms_calc_breakeven_price")

    assert "firms_calc_breakeven_price" not in pipeline._event_map


def test_pipeline_event_map_updated_on_replace():
    """Event map is updated when events are replaced."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_calc_breakeven_price",
        ]
    )

    pipeline.replace("firms_calc_breakeven_price", "firms_adjust_price")

    assert "firms_calc_breakeven_price" not in pipeline._event_map
    assert "firms_adjust_price" in pipeline._event_map


def test_repeated_event_executes_multiple_times():
    """RepeatedEvent executes underlying event multiple times."""
    from bamengine.core.pipeline import RepeatedEvent
    from bamengine.simulation import Simulation

    sim = Simulation.init(n_firms=5, n_households=20, seed=42)

    # Store initial state
    initial_desired_production = sim.prod.desired_production.copy()

    # Create repeated event
    event = FirmsDecideDesiredProduction()
    repeated = RepeatedEvent(event, n_repeats=3)  # type: ignore[arg-type]

    # Execute
    repeated.execute(sim)

    # Should have executed (production changed)
    assert not np.all(sim.prod.desired_production == initial_desired_production)


def test_repeated_event_name_property():
    """RepeatedEvent.name returns underlying event name."""
    from bamengine.core.pipeline import RepeatedEvent

    event = FirmsDecideDesiredProduction()
    repeated = RepeatedEvent(event, n_repeats=5)  # type: ignore[arg-type]

    assert repeated.name == "firms_decide_desired_production"


def test_pipeline_with_repeats():
    """Pipeline handles repeated events."""
    from bamengine.core.pipeline import RepeatedEvent

    pipeline = Pipeline.from_event_list(
        [
            "workers_send_one_round",
            "firms_hire_workers",
        ],
        repeats={
            "workers_send_one_round": 5,
            "firms_hire_workers": 5,
        },
    )

    event1 = pipeline.events[0]
    event2 = pipeline.events[1]

    # Both events should be wrapped in RepeatedEvent
    assert isinstance(event1, RepeatedEvent)
    assert isinstance(event2, RepeatedEvent)
    # Cast to RepeatedEvent to check n_repeats
    cast(RepeatedEvent, event1)
    cast(RepeatedEvent, event2)
    # Both should have n_repeats of 5
    assert event1.n_repeats == 5
    assert event2.n_repeats == 5


def test_pipeline_repeated_events_preserve_order():
    """Pipeline preserves order of repeated events."""
    from bamengine.core.pipeline import RepeatedEvent

    # These events are in reverse order - will be preserved
    pipeline = Pipeline.from_event_list(
        [
            "firms_hire_workers",  # Listed first
            "workers_send_one_round",  # Listed second
        ],
        repeats={
            "workers_send_one_round": 3,
            "firms_hire_workers": 3,
        },
    )

    # Order should be preserved (even though it's logically wrong)
    assert pipeline.events[0].name == "firms_hire_workers"
    assert pipeline.events[1].name == "workers_send_one_round"
    # Both should be repeated
    assert isinstance(pipeline.events[0], RepeatedEvent)
    assert isinstance(pipeline.events[1], RepeatedEvent)


# ============================================================================
# After-Event Callback Tests
# ============================================================================


def test_pipeline_register_after_event_callback():
    """Can register callback to run after specific event."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
            "firms_calc_breakeven_price",
        ]
    )

    callback_calls = []

    def my_callback(sim):
        callback_calls.append("called")

    pipeline.register_after_event("firms_decide_desired_production", my_callback)

    # Verify callback is registered
    assert "firms_decide_desired_production" in pipeline._after_event_callbacks
    assert len(pipeline._after_event_callbacks["firms_decide_desired_production"]) == 1


def test_pipeline_clear_callbacks():
    """Can clear all registered callbacks."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
        ]
    )

    def my_callback(sim):
        pass

    pipeline.register_after_event("firms_decide_desired_production", my_callback)
    assert len(pipeline._after_event_callbacks) > 0

    pipeline.clear_callbacks()
    assert len(pipeline._after_event_callbacks) == 0


def test_pipeline_multiple_callbacks_same_event():
    """Can register multiple callbacks for same event."""
    pipeline = Pipeline.from_event_list(
        [
            "firms_decide_desired_production",
        ]
    )

    def callback1(sim):
        pass

    def callback2(sim):
        pass

    pipeline.register_after_event("firms_decide_desired_production", callback1)
    pipeline.register_after_event("firms_decide_desired_production", callback2)

    callbacks = pipeline._after_event_callbacks["firms_decide_desired_production"]
    assert len(callbacks) == 2


def test_pipeline_callbacks_fire_during_execute():
    """Callbacks fire during pipeline.execute()."""
    from bamengine.simulation import Simulation

    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    callback_results = []

    def capture_production(s):
        prod = s.get_role("Producer")
        callback_results.append(prod.production.sum())

    sim.pipeline.register_after_event("firms_run_production", capture_production)

    # Execute one step
    sim.step()

    # Callback should have been called
    assert len(callback_results) == 1
    assert callback_results[0] > 0

    # Clean up
    sim.pipeline.clear_callbacks()
