"""Tests for configuration loading and precedence."""

import logging
import tempfile
from pathlib import Path

import pytest

import bamengine.events  # noqa: F401 - register events
from bamengine.simulation import Simulation


def test_defaults_yml_loads():
    """Package config/defaults.yml can be loaded."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    # Check population sizes (stored on Simulation, not Config)
    assert sim.n_firms == 10  # Overridden by kwargs
    assert sim.n_households == 50  # Overridden by kwargs
    assert sim.n_banks == 10  # From config/defaults.yml

    # Check hyperparameters (stored in Config)
    assert sim.config.h_rho == 0.10  # From config/defaults.yml
    assert sim.config.max_M == 4  # From config/defaults.yml


def test_user_yaml_overrides_defaults():
    """User YAML file overrides defaults.yml."""
    yaml_content = """
        n_firms: 200
        n_households: 1000
        h_rho: 0.15
        max_M: 5
        """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        sim = Simulation.init(config=yaml_path, seed=42)

        # User YAML overrides defaults
        assert sim.n_firms == 200
        assert sim.n_households == 1000
        assert sim.config.h_rho == 0.15
        assert sim.config.max_M == 5

        # Defaults still apply for unspecified values
        assert sim.n_banks == 10  # From config/defaults.yml
        assert sim.config.theta == 8  # From config/defaults.yml
    finally:
        Path(yaml_path).unlink()


def test_kwargs_override_yaml():
    """Keyword arguments have highest precedence."""
    yaml_content = """
        n_firms: 200
        n_households: 1000
        h_rho: 0.15
        """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        sim = Simulation.init(
            config=yaml_path,
            n_firms=50,  # Override YAML
            h_rho=0.20,  # Override YAML
            seed=42,
        )

        # Kwargs override YAML
        assert sim.n_firms == 50
        assert sim.config.h_rho == 0.20

        # YAML still applies where not overridden
        assert sim.n_households == 1000
    finally:
        Path(yaml_path).unlink()


def test_config_precedence_chain():
    """Full precedence chain: defaults < YAML < kwargs."""
    yaml_content = """
n_firms: 200
max_M: 5
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        sim = Simulation.init(
            config=yaml_path,
            n_firms=75,  # Highest precedence
            seed=42,
        )

        # kwargs > YAML > defaults
        assert sim.n_firms == 75  # From kwargs
        assert sim.config.max_M == 5  # From YAML
        assert sim.n_banks == 10  # From defaults
    finally:
        Path(yaml_path).unlink()


def test_custom_pipeline_path():
    """Custom pipeline path can be specified."""
    # Create a minimal custom pipeline
    pipeline_yaml = """
events:
  - firms_decide_desired_production
  - firms_calc_breakeven_price
  - firms_adjust_price
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(pipeline_yaml)
        pipeline_path = f.name

    try:
        sim = Simulation.init(
            n_firms=10,
            n_households=50,
            pipeline_path=pipeline_path,
            seed=42,
        )

        # Should have custom pipeline with 3 events
        assert len(sim.pipeline) == 3
        assert sim.pipeline.events[0].name == "firms_decide_desired_production"
        assert sim.pipeline.events[1].name == "firms_calc_breakeven_price"
        assert sim.pipeline.events[2].name == "firms_adjust_price"
    finally:
        Path(pipeline_path).unlink()


def test_default_pipeline_when_no_custom_path():
    """Default pipeline is used when pipeline_path is None."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    # Interleaved matching (default): pipeline length depends on max_M and max_H
    # 37 base events + (max_M-1)*2 extra labor rounds + (max_H-1)*2 extra credit rounds
    assert len(sim.pipeline) == 47

    # Check first and last events
    assert sim.pipeline.events[0].name == "firms_decide_desired_production"
    assert sim.pipeline.events[-1].name == "spawn_replacement_banks"


def test_logging_default_level():
    """Default log level can be configured."""
    log_config = {
        "default_level": "DEBUG",
        "events": {},
    }

    Simulation.init(
        n_firms=10,
        n_households=50,
        logging=log_config,
        seed=42,
    )

    # Check that bamengine logger has DEBUG level
    logger = logging.getLogger("bamengine")
    assert logger.level == logging.DEBUG


def test_logging_per_event_overrides():
    """Per-event log level overrides work."""
    log_config = {
        "default_level": "INFO",
        "events": {
            "workers_send_one_round": "DEBUG",
            "firms_hire_workers": "WARNING",
        },
    }

    Simulation.init(
        n_firms=10,
        n_households=50,
        logging=log_config,
        seed=42,
    )

    # Check default level
    logger = logging.getLogger("bamengine")
    assert logger.level == logging.INFO

    # Check per-event overrides
    worker_logger = logging.getLogger("bamengine.events.workers_send_one_round")
    assert worker_logger.level == logging.DEBUG

    hire_logger = logging.getLogger("bamengine.events.firms_hire_workers")
    assert hire_logger.level == logging.WARNING


def test_logging_from_yaml():
    """Logging configuration can be specified in YAML."""
    yaml_content = """
n_firms: 10
n_households: 50
logging:
  default_level: DEBUG
  events:
    consumers_shop_one_round: WARNING
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        Simulation.init(config=yaml_path, seed=42)

        # Check logging configuration was applied
        logger = logging.getLogger("bamengine")
        assert logger.level == logging.DEBUG

        shop_logger = logging.getLogger("bamengine.events.consumers_shop_one_round")
        assert shop_logger.level == logging.WARNING
    finally:
        Path(yaml_path).unlink()


def test_pipeline_path_from_yaml():
    """Pipeline path can be specified in YAML."""
    # Create custom pipeline
    pipeline_yaml = """
events:
  - firms_decide_desired_production
  - firms_run_production
"""

    # Create temp files - close them before deleting (Windows compatibility)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as pf:
        pf.write(pipeline_yaml)
        pf.flush()
        pipeline_path = pf.name
    # File is now closed

    # Create config YAML referencing custom pipeline (must use absolute path)
    # Use forward slashes for YAML compatibility (works on Windows too)
    pipeline_path_posix = Path(pipeline_path).as_posix()
    config_yaml = f"""
n_firms: 10
n_households: 50
pipeline_path: "{pipeline_path_posix}"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as cf:
        cf.write(config_yaml)
        cf.flush()
        config_path = cf.name
    # File is now closed

    try:
        sim = Simulation.init(config=config_path, seed=42)

        # Should have custom pipeline with 2 events
        assert len(sim.pipeline) == 2
        assert sim.pipeline.events[0].name == "firms_decide_desired_production"
        assert sim.pipeline.events[1].name == "firms_run_production"
    finally:
        # Clean up both temp files (both are closed now - Windows safe)
        Path(config_path).unlink(missing_ok=True)
        Path(pipeline_path).unlink(missing_ok=True)


def test_kwargs_override_logging():
    """Logging config from kwargs overrides YAML."""
    yaml_content = """
logging:
  default_level: WARNING
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        log_config = {
            "default_level": "DEBUG",
            "events": {},
        }

        Simulation.init(
            config=yaml_path,
            n_firms=10,
            n_households=50,
            logging=log_config,  # Override YAML
            seed=42,
        )

        # Kwargs override YAML
        logger = logging.getLogger("bamengine")
        assert logger.level == logging.DEBUG
    finally:
        Path(yaml_path).unlink()


def test_config_dict_passed_directly():
    """Config can be passed as dict directly (not just YAML path)."""
    config_dict = {
        "n_firms": 75,
        "n_households": 400,
        "h_rho": 0.15,
    }

    sim = Simulation.init(config=config_dict, seed=42)

    # Dict config applied
    assert sim.n_firms == 75
    assert sim.n_households == 400
    assert sim.config.h_rho == 0.15


def test_pricing_phase_conflicts_with_pipeline_path():
    """Setting pricing_phase != 'planning' with pipeline_path should raise."""

    pipeline_yaml = """
events:
  - firms_decide_desired_production
  - firms_calc_breakeven_price
  - firms_adjust_price
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(pipeline_yaml)
        pipeline_path = f.name

    try:
        with pytest.raises(
            ValueError, match="cannot be used with a custom pipeline_path"
        ):
            Simulation.init(
                n_firms=10,
                n_households=50,
                pipeline_path=pipeline_path,
                pricing_phase="production",
                seed=42,
            )
    finally:
        Path(pipeline_path).unlink()


def test_pricing_phase_planning_with_pipeline_path_is_ok():
    """pricing_phase='planning' (default) with pipeline_path should be fine."""

    pipeline_yaml = """
events:
  - firms_decide_desired_production
  - firms_calc_breakeven_price
  - firms_adjust_price
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(pipeline_yaml)
        pipeline_path = f.name

    try:
        sim = Simulation.init(
            n_firms=10,
            n_households=50,
            pipeline_path=pipeline_path,
            pricing_phase="planning",
            seed=42,
        )
        assert len(sim.pipeline) == 3
    finally:
        Path(pipeline_path).unlink()


def test_labor_matching_conflicts_with_pipeline_path():
    """Setting labor_matching != default with pipeline_path should raise."""

    pipeline_yaml = """
events:
  - firms_decide_desired_production
  - firms_calc_breakeven_price
  - firms_adjust_price
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(pipeline_yaml)
        pipeline_path = f.name

    try:
        with pytest.raises(
            ValueError, match="cannot be used with a custom pipeline_path"
        ):
            Simulation.init(
                n_firms=10,
                n_households=50,
                pipeline_path=pipeline_path,
                labor_matching="cascade",
                seed=42,
            )
    finally:
        Path(pipeline_path).unlink()


def test_credit_matching_conflicts_with_pipeline_path():
    """Setting credit_matching != default with pipeline_path should raise."""

    pipeline_yaml = """
events:
  - firms_decide_desired_production
  - firms_calc_breakeven_price
  - firms_adjust_price
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(pipeline_yaml)
        pipeline_path = f.name

    try:
        with pytest.raises(
            ValueError, match="cannot be used with a custom pipeline_path"
        ):
            Simulation.init(
                n_firms=10,
                n_households=50,
                pipeline_path=pipeline_path,
                credit_matching="cascade",
                seed=42,
            )
    finally:
        Path(pipeline_path).unlink()


def test_min_wage_ratchet_with_pipeline_path_is_ok():
    """min_wage_ratchet with pipeline_path should be fine (doesn't affect pipeline)."""

    pipeline_yaml = """
events:
  - firms_decide_desired_production
  - firms_calc_breakeven_price
  - firms_adjust_price
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(pipeline_yaml)
        pipeline_path = f.name

    try:
        sim = Simulation.init(
            n_firms=10,
            n_households=50,
            pipeline_path=pipeline_path,
            min_wage_ratchet=True,
            seed=42,
        )
        assert sim.config.min_wage_ratchet is True
    finally:
        Path(pipeline_path).unlink()


def test_multiple_pipeline_params_conflict_with_pipeline_path():
    """Multiple non-default pipeline params with pipeline_path reports all."""

    pipeline_yaml = """
events:
  - firms_decide_desired_production
  - firms_calc_breakeven_price
  - firms_adjust_price
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(pipeline_yaml)
        pipeline_path = f.name

    try:
        with pytest.raises(ValueError, match="labor_matching.*credit_matching"):
            Simulation.init(
                n_firms=10,
                n_households=50,
                pipeline_path=pipeline_path,
                labor_matching="cascade",
                credit_matching="cascade",
                seed=42,
            )
    finally:
        Path(pipeline_path).unlink()


def test_implementation_variant_config_defaults():
    """New implementation variant params have correct defaults."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    assert sim.config.labor_matching == "interleaved"
    assert sim.config.credit_matching == "interleaved"
    assert sim.config.min_wage_ratchet is False


def test_implementation_variant_config_override():
    """Implementation variant params can be overridden via kwargs."""
    sim = Simulation.init(
        n_firms=10,
        n_households=50,
        labor_matching="cascade",
        credit_matching="cascade",
        min_wage_ratchet=True,
        seed=42,
    )
    assert sim.config.labor_matching == "cascade"
    assert sim.config.credit_matching == "cascade"
    assert sim.config.min_wage_ratchet is True


def test_invalid_labor_matching_value():
    """Invalid labor_matching value should raise ValueError."""
    with pytest.raises(ValueError, match="labor_matching"):
        Simulation.init(n_firms=10, n_households=50, labor_matching="invalid", seed=42)


def test_invalid_credit_matching_value():
    """Invalid credit_matching value should raise ValueError."""
    with pytest.raises(ValueError, match="credit_matching"):
        Simulation.init(n_firms=10, n_households=50, credit_matching="invalid", seed=42)


def test_config_yaml_non_mapping_root():
    """Reject YAML file with non-mapping root."""
    yaml_content = """
- item1
- item2
- item3
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        with pytest.raises(TypeError, match="config root must be mapping"):
            Simulation.init(config=yaml_path, seed=42)
    finally:
        Path(yaml_path).unlink()
