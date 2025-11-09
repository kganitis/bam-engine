"""Tests for logging configuration and behavior."""

import logging
import tempfile
from pathlib import Path

import pytest

from bamengine import Simulation
from bamengine._logging_ext import DEEP_DEBUG, BamLogger, getLogger


class TestBamLogger:
    """Test custom BamLogger functionality."""

    def test_bamlogger_has_deep_method(self):
        """BamLogger should have a deep() method."""
        logger = getLogger("test")
        assert hasattr(logger, "deep")
        assert callable(logger.deep)

    def test_deep_level_exists(self):
        """DEEP_DEBUG level should be registered."""
        assert DEEP_DEBUG == 5
        assert logging.getLevelName(DEEP_DEBUG) == "DEEP"

    def test_deep_logging_when_enabled(self, caplog):
        """deep() should log when level is DEEP_DEBUG."""
        logger = getLogger("test.deep")
        logger.setLevel(DEEP_DEBUG)

        with caplog.at_level(DEEP_DEBUG, logger="test.deep"):
            logger.deep("Deep debug message")

        assert "Deep debug message" in caplog.text

    def test_deep_logging_when_disabled(self, caplog):
        """deep() should not log when level is INFO."""
        logger = getLogger("test.deep_disabled")
        logger.setLevel(logging.INFO)

        with caplog.at_level(logging.INFO, logger="test.deep_disabled"):
            logger.deep("Should not appear")

        assert "Should not appear" not in caplog.text

    def test_is_enabled_for_deep(self):
        """isEnabledFor(DEEP_DEBUG) should work correctly."""
        logger = getLogger("test.enabled")

        # Should be disabled at INFO level
        logger.setLevel(logging.INFO)
        assert not logger.isEnabledFor(DEEP_DEBUG)

        # Should be enabled at DEEP_DEBUG level
        logger.setLevel(DEEP_DEBUG)
        assert logger.isEnabledFor(DEEP_DEBUG)


class TestLoggingConfiguration:
    """Test logging configuration via Simulation.init()."""

    def test_default_log_level(self):
        """Default log level should be INFO."""
        sim = Simulation.init(n_firms=10, n_households=10, n_banks=2, seed=42)

        # Check default level
        logger = logging.getLogger("bamengine")
        assert logger.level == logging.INFO

    def test_set_default_level_debug(self):
        """Can set default level to DEBUG via kwargs."""
        log_config = {"default_level": "DEBUG", "events": {}}
        sim = Simulation.init(
            n_firms=10, n_households=10, n_banks=2, logging=log_config, seed=42
        )

        logger = logging.getLogger("bamengine")
        assert logger.level == logging.DEBUG

    def test_set_default_level_warning(self):
        """Can set default level to WARNING via kwargs."""
        log_config = {"default_level": "WARNING", "events": {}}
        sim = Simulation.init(
            n_firms=10, n_households=10, n_banks=2, logging=log_config, seed=42
        )

        logger = logging.getLogger("bamengine")
        assert logger.level == logging.WARNING

    def test_set_default_level_deep_debug(self):
        """Can set default level to DEEP_DEBUG via kwargs."""
        log_config = {"default_level": "DEEP_DEBUG", "events": {}}
        sim = Simulation.init(
            n_firms=10, n_households=10, n_banks=2, logging=log_config, seed=42
        )

        logger = logging.getLogger("bamengine")
        assert logger.level == DEEP_DEBUG

    def test_per_event_log_level_override(self):
        """Can override log level for specific events."""
        log_config = {
            "default_level": "INFO",
            "events": {
                "firms_adjust_price": "DEBUG",
                "workers_send_one_round": "WARNING",
            },
        }
        sim = Simulation.init(
            n_firms=10, n_households=10, n_banks=2, logging=log_config, seed=42
        )

        # Check event-specific loggers
        pricing_logger = logging.getLogger("bamengine.events.firms_adjust_price")
        assert pricing_logger.level == logging.DEBUG

        worker_logger = logging.getLogger("bamengine.events.workers_send_one_round")
        assert worker_logger.level == logging.WARNING

    def test_per_event_deep_debug_level(self):
        """Can set DEEP_DEBUG level for specific events."""
        log_config = {
            "default_level": "INFO",
            "events": {
                "workers_send_one_round": "DEEP_DEBUG",
                "firms_hire_workers": "DEEP_DEBUG",
            },
        }
        sim = Simulation.init(
            n_firms=10, n_households=10, n_banks=2, logging=log_config, seed=42
        )

        # Check event-specific loggers have DEEP_DEBUG level
        worker_logger = logging.getLogger("bamengine.events.workers_send_one_round")
        assert worker_logger.level == DEEP_DEBUG

        hiring_logger = logging.getLogger("bamengine.events.firms_hire_workers")
        assert hiring_logger.level == DEEP_DEBUG

    def test_logging_from_yaml_config(self):
        """Can configure logging via YAML file."""
        yaml_content = """
n_firms: 10
n_households: 10
n_banks: 2
logging:
  default_level: DEBUG
  events:
    firms_adjust_price: WARNING
    workers_send_one_round: DEBUG
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            yaml_path = f.name

        try:
            sim = Simulation.init(config=yaml_path, seed=42)

            # Check default level
            logger = logging.getLogger("bamengine")
            assert logger.level == logging.DEBUG

            # Check event-specific levels
            pricing_logger = logging.getLogger("bamengine.events.firms_adjust_price")
            assert pricing_logger.level == logging.WARNING

            worker_logger = logging.getLogger("bamengine.events.workers_send_one_round")
            assert worker_logger.level == logging.DEBUG
        finally:
            Path(yaml_path).unlink()

    def test_kwargs_override_yaml_logging(self):
        """Logging config in kwargs should override YAML."""
        yaml_content = """
n_firms: 10
n_households: 10
n_banks: 2
logging:
  default_level: INFO
  events:
    firms_adjust_price: INFO
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            yaml_path = f.name

        try:
            # Override with kwargs
            log_config = {
                "default_level": "DEBUG",
                "events": {"firms_adjust_price": "WARNING"},
            }
            sim = Simulation.init(config=yaml_path, logging=log_config, seed=42)

            # Kwargs should win
            logger = logging.getLogger("bamengine")
            assert logger.level == logging.DEBUG

            pricing_logger = logging.getLogger("bamengine.events.firms_adjust_price")
            assert pricing_logger.level == logging.WARNING
        finally:
            Path(yaml_path).unlink()


class TestEventGetLogger:
    """Test Event.get_logger() method."""

    def test_event_get_logger_returns_correct_logger(self):
        """Event.get_logger() should return logger with correct name."""
        sim = Simulation.init(n_firms=10, n_households=10, n_banks=2, seed=42)

        # Get an event from the pipeline
        event = sim.get_event("firms_adjust_price")
        logger = event.get_logger()

        # Check logger name
        assert logger.name == "bamengine.events.firms_adjust_price"

    def test_event_logger_respects_per_event_level(self):
        """Event logger should have the configured per-event level."""
        log_config = {
            "default_level": "INFO",
            "events": {"firms_adjust_price": "DEBUG"},
        }
        sim = Simulation.init(
            n_firms=10, n_households=10, n_banks=2, logging=log_config, seed=42
        )

        event = sim.get_event("firms_adjust_price")
        logger = event.get_logger()

        # Should have DEBUG level
        assert logger.level == logging.DEBUG

    def test_multiple_events_different_levels(self):
        """Different events can have different log levels."""
        log_config = {
            "default_level": "INFO",
            "events": {
                "firms_adjust_price": "DEBUG",
                "workers_send_one_round": "WARNING",
                "firms_hire_workers": "ERROR",
            },
        }
        sim = Simulation.init(
            n_firms=10, n_households=10, n_banks=2, logging=log_config, seed=42
        )

        # Check each event has correct level
        pricing_event = sim.get_event("firms_adjust_price")
        assert pricing_event.get_logger().level == logging.DEBUG

        worker_event = sim.get_event("workers_send_one_round")
        assert worker_event.get_logger().level == logging.WARNING

        hiring_event = sim.get_event("firms_hire_workers")
        assert hiring_event.get_logger().level == logging.ERROR


class TestIsEnabledForGuards:
    """Test that isEnabledFor() guards work correctly."""

    def test_is_enabled_for_with_info_level(self):
        """isEnabledFor() should return False for DEBUG when level is INFO."""
        logger = getLogger("test.guards")
        logger.setLevel(logging.INFO)

        assert logger.isEnabledFor(logging.INFO)
        assert logger.isEnabledFor(logging.WARNING)
        assert not logger.isEnabledFor(logging.DEBUG)
        assert not logger.isEnabledFor(DEEP_DEBUG)

    def test_is_enabled_for_with_debug_level(self):
        """isEnabledFor() should return True for DEBUG when level is DEBUG."""
        logger = getLogger("test.guards_debug")
        logger.setLevel(logging.DEBUG)

        assert logger.isEnabledFor(logging.DEBUG)
        assert logger.isEnabledFor(logging.INFO)
        assert not logger.isEnabledFor(DEEP_DEBUG)

    def test_is_enabled_for_with_deep_level(self):
        """isEnabledFor() should return True for all levels when level is DEEP."""
        logger = getLogger("test.guards_deep")
        logger.setLevel(DEEP_DEBUG)

        assert logger.isEnabledFor(DEEP_DEBUG)
        assert logger.isEnabledFor(logging.DEBUG)
        assert logger.isEnabledFor(logging.INFO)

    def test_expensive_debug_only_called_when_enabled(self, caplog):
        """Expensive debug formatting should only happen when enabled."""
        logger = getLogger("test.expensive")
        logger.setLevel(logging.INFO)

        call_count = 0

        def expensive_operation():
            nonlocal call_count
            call_count += 1
            return "expensive result"

        # This should NOT call expensive_operation
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Debug: %s", expensive_operation())

        assert call_count == 0

        # Enable DEBUG
        logger.setLevel(logging.DEBUG)

        # Now it SHOULD call expensive_operation
        with caplog.at_level(logging.DEBUG, logger="test.expensive"):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Debug: %s", expensive_operation())

        assert call_count == 1
        assert "expensive result" in caplog.text


class TestStructuredLogging:
    """Test structured logging with extra dict."""

    def test_logging_with_extra_dict(self, caplog):
        """Can log with extra dict for structured logging."""
        logger = getLogger("test.structured")
        logger.setLevel(logging.INFO)

        with caplog.at_level(logging.INFO, logger="test.structured"):
            logger.info(
                "Event executed",
                extra={"event_name": "test_event", "duration": 0.5, "step": 10},
            )

        # Check message appears
        assert "Event executed" in caplog.text

    def test_extra_dict_available_in_record(self, caplog):
        """Extra dict values should be available in log record."""
        logger = getLogger("test.structured_record")
        logger.setLevel(logging.DEBUG)

        with caplog.at_level(logging.DEBUG, logger="test.structured_record"):
            logger.debug(
                "Processing agents", extra={"n_agents": 100, "market": "labor"}
            )

        # Check that extra fields were captured
        assert len(caplog.records) > 0
        record = caplog.records[0]
        assert hasattr(record, "n_agents")
        assert hasattr(record, "market")
        assert record.n_agents == 100
        assert record.market == "labor"

    def test_deep_logging_with_extra(self, caplog):
        """DEEP level logging should work with extra dict."""
        logger = getLogger("test.deep_structured")
        logger.setLevel(DEEP_DEBUG)

        with caplog.at_level(DEEP_DEBUG, logger="test.deep_structured"):
            logger.deep("Inner loop iteration", extra={"round": 5, "agent_id": 42})

        assert "Inner loop iteration" in caplog.text
        assert len(caplog.records) > 0
        record = caplog.records[0]
        assert record.round == 5
        assert record.agent_id == 42
