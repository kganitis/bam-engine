"""Tests for configuration loading and precedence."""

import logging
import tempfile
from pathlib import Path

import bamengine.events  # noqa: F401 - register events
from bamengine.simulation import Simulation


def test_defaults_yml_loads():
    """Package defaults.yml can be loaded."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    # Check population sizes (stored on Simulation, not Config)
    assert sim.n_firms == 10  # Overridden by kwargs
    assert sim.n_households == 50  # Overridden by kwargs
    assert sim.n_banks == 10  # From defaults.yml

    # Check hyperparameters (stored in Config)
    assert sim.config.h_rho == 0.10  # From defaults.yml
    assert sim.config.max_M == 4  # From defaults.yml


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
        assert sim.n_banks == 10  # From defaults.yml
        assert sim.config.theta == 8  # From defaults.yml
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

    # Should have default pipeline (52 events with default max_M=4, max_H=2, max_Z=2)
    expected_length = 34 + 2 * 4 + 2 * 2 + 2
    assert len(sim.pipeline) == expected_length

    # Check first and last events
    assert sim.pipeline.events[0].name == "firms_decide_desired_production"
    assert sim.pipeline.events[-1].name == "calc_unemployment_rate"


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

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as pf:
        pf.write(pipeline_yaml)
        pf.flush()
        pipeline_path = pf.name

        # Create config YAML referencing custom pipeline (must use absolute path)
        config_yaml = f"""
n_firms: 10
n_households: 50
pipeline_path: "{pipeline_path}"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as cf:
            cf.write(config_yaml)
            cf.flush()
            config_path = cf.name

            try:
                sim = Simulation.init(config=config_path, seed=42)

                # Should have custom pipeline with 2 events
                assert len(sim.pipeline) == 2
                assert sim.pipeline.events[0].name == "firms_decide_desired_production"
                assert sim.pipeline.events[1].name == "firms_run_production"
            finally:
                Path(config_path).unlink()

    Path(pipeline_path).unlink()


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
