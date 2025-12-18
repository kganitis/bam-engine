"""Tests for YAML-based pipeline loading."""

import tempfile
from pathlib import Path

import pytest

from bamengine.core.pipeline import Pipeline


def test_parse_event_spec_single_event():
    """Parse single event specification."""
    result = Pipeline._parse_event_spec("firms_decide_desired_production")
    assert result == ["firms_decide_desired_production"]


def test_parse_event_spec_repeated_event():
    """Parse repeated event specification (event x N)."""
    result = Pipeline._parse_event_spec("consumers_shop_one_round x 3")
    assert result == [
        "consumers_shop_one_round",
        "consumers_shop_one_round",
        "consumers_shop_one_round",
    ]


def test_parse_event_spec_interleaved_events():
    """Parse interleaved event specification (event1 <-> event2 x N)."""
    result = Pipeline._parse_event_spec(
        "workers_send_one_round <-> firms_hire_workers x 2"
    )
    assert result == [
        "workers_send_one_round",
        "firms_hire_workers",
        "workers_send_one_round",
        "firms_hire_workers",
    ]


def test_parse_event_spec_whitespace_handling():
    """Parse handles various whitespace patterns."""
    # Extra spaces around operators
    result = Pipeline._parse_event_spec("event1  <->  event2  x  3")
    assert result == ["event1", "event2", "event1", "event2", "event1", "event2"]

    # Tabs and spaces
    result = Pipeline._parse_event_spec("event_name\tx\t5")
    assert len(result) == 5
    assert all(e == "event_name" for e in result)


def test_from_yaml_basic():
    """Load pipeline from simple YAML file."""
    import bamengine.events  # noqa: F401 - register events

    yaml_content = """
events:
  - firms_decide_desired_production
  - firms_calc_breakeven_price
  - firms_adjust_price
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        pipeline = Pipeline.from_yaml(yaml_path)
        assert len(pipeline) == 3
        assert pipeline.events[0].name == "firms_decide_desired_production"
        assert pipeline.events[1].name == "firms_calc_breakeven_price"
        assert pipeline.events[2].name == "firms_adjust_price"
    finally:
        Path(yaml_path).unlink()


def test_from_yaml_with_parameters():
    """Load pipeline with parameter substitution."""
    import bamengine.events  # noqa: F401

    yaml_content = """
events:
  - firms_decide_desired_production
  - workers_send_one_round x {max_M}
  - consumers_shop_one_round x {max_Z}
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        pipeline = Pipeline.from_yaml(yaml_path, max_M=3, max_Z=2)
        event_names = [e.name for e in pipeline.events]

        # Should have 1 + 3 + 2 = 6 events
        assert len(pipeline) == 6
        assert event_names.count("workers_send_one_round") == 3
        assert event_names.count("consumers_shop_one_round") == 2
    finally:
        Path(yaml_path).unlink()


def test_from_yaml_with_interleaved_events():
    """Load pipeline with interleaved event pattern."""
    import bamengine.events  # noqa: F401

    yaml_content = """
events:
  - firms_decide_wage_offer
  - workers_send_one_round <-> firms_hire_workers x {max_M}
  - firms_calc_wage_bill
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        pipeline = Pipeline.from_yaml(yaml_path, max_M=4)
        event_names = [e.name for e in pipeline.events]

        # Should have 1 + (2*4) + 1 = 10 events
        assert len(pipeline) == 10

        # Check interleaving pattern
        assert event_names[0] == "firms_decide_wage_offer"
        assert event_names[1] == "workers_send_one_round"
        assert event_names[2] == "firms_hire_workers"
        assert event_names[3] == "workers_send_one_round"
        assert event_names[4] == "firms_hire_workers"
        assert event_names[-1] == "firms_calc_wage_bill"
    finally:
        Path(yaml_path).unlink()


def test_from_yaml_missing_events_key():
    """YAML without 'events' key raises error."""
    yaml_content = """
pipeline:
  - some_event
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        with pytest.raises(ValueError, match="must have 'events' key"):
            Pipeline.from_yaml(yaml_path)
    finally:
        Path(yaml_path).unlink()


def test_default_pipeline_loads_from_yaml():
    """Default pipeline successfully loads from YAML."""
    from bamengine.core.pipeline import create_default_pipeline

    pipeline = create_default_pipeline(max_M=4, max_H=2, max_Z=2)

    # Should create a valid pipeline
    assert len(pipeline) > 0
    assert pipeline.events[0].name == "firms_decide_desired_production"
    # calc_unemployment_rate deprecated, last event is spawn_replacement_banks
    assert pipeline.events[-1].name == "spawn_replacement_banks"


def test_user_custom_pipeline_example():
    """Example of user creating custom pipeline from YAML."""
    import bamengine.events  # noqa: F401

    custom_yaml = """
events:
  # Custom simplified pipeline (for testing/research)
  - firms_decide_desired_production
  - firms_adjust_price
  - workers_send_one_round <-> firms_hire_workers x 2
  - firms_run_production
  - consumers_shop_one_round x 2
  - firms_collect_revenue
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(custom_yaml)
        yaml_path = f.name

    try:
        pipeline = Pipeline.from_yaml(yaml_path)

        # Should create simplified pipeline
        # 1 + 1 + (2*2) + 1 + 2 + 1 = 10 events
        assert len(pipeline) == 10
        event_names = [e.name for e in pipeline.events]
        assert event_names[0] == "firms_decide_desired_production"
        assert event_names[-1] == "firms_collect_revenue"
    finally:
        Path(yaml_path).unlink()
