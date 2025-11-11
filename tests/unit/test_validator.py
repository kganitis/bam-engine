"""Tests for configuration validation."""

import tempfile
import warnings
from pathlib import Path

import pytest

from bamengine.config import ConfigValidator
from bamengine.simulation import Simulation


class TestTypeValidation:
    """Test type checking for configuration parameters."""

    def test_integer_params_accept_int(self):
        """Integer parameters should accept int values."""
        cfg = {"n_firms": 100, "n_households": 500, "max_M": 4}
        # Should not raise
        ConfigValidator._validate_types(cfg)

    def test_integer_params_reject_float(self):
        """Integer parameters should reject float values."""
        cfg = {"n_firms": 100.5}
        with pytest.raises(ValueError, match="must be int"):
            ConfigValidator._validate_types(cfg)

    def test_integer_params_reject_string(self):
        """Integer parameters should reject string values."""
        cfg = {"n_firms": "100"}
        with pytest.raises(ValueError, match="must be int"):
            ConfigValidator._validate_types(cfg)

    def test_float_params_accept_float(self):
        """Float parameters should accept float values."""
        cfg = {"h_rho": 0.1, "beta": 2.5, "delta": 0.4}
        # Should not raise
        ConfigValidator._validate_types(cfg)

    def test_float_params_accept_int(self):
        """Float parameters should accept int values (coercion)."""
        cfg = {"h_rho": 1, "beta": 2}
        # Should not raise
        ConfigValidator._validate_types(cfg)

    def test_float_params_reject_string(self):
        """Float parameters should reject string values."""
        cfg = {"h_rho": "0.1"}
        with pytest.raises(ValueError, match="must be float"):
            ConfigValidator._validate_types(cfg)

    def test_pipeline_path_accepts_string(self):
        """pipeline_path should accept string values."""
        cfg = {"pipeline_path": "/path/to/pipeline.yml"}
        # Should not raise
        ConfigValidator._validate_types(cfg)

    def test_pipeline_path_accepts_none(self):
        """pipeline_path should accept None."""
        cfg = {"pipeline_path": None}
        # Should not raise
        ConfigValidator._validate_types(cfg)

    def test_pipeline_path_rejects_int(self):
        """pipeline_path should reject int values."""
        cfg = {"pipeline_path": 123}
        with pytest.raises(ValueError, match="must be str or None"):
            ConfigValidator._validate_types(cfg)

    def test_optional_params_accept_none(self):
        """Optional parameters like cap_factor should accept None."""
        cfg = {"cap_factor": None}
        # Should not raise
        ConfigValidator._validate_types(cfg)


class TestRangeValidation:
    """Test range validation for configuration parameters."""

    def test_population_sizes_must_be_positive(self):
        """Population sizes must be at least 1."""
        with pytest.raises(ValueError, match="must be >= 1"):
            ConfigValidator._validate_ranges({"n_firms": 0})

        with pytest.raises(ValueError, match="must be >= 1"):
            ConfigValidator._validate_ranges({"n_households": -1})

    def test_shock_params_in_range(self):
        """Shock parameters must be between 0 and 1."""
        # Valid values
        ConfigValidator._validate_ranges({"h_rho": 0.0, "h_xi": 1.0})

        # Too small
        with pytest.raises(ValueError, match="must be >= 0.0"):
            ConfigValidator._validate_ranges({"h_rho": -0.1})

        # Too large
        with pytest.raises(ValueError, match="must be <= 1.0"):
            ConfigValidator._validate_ranges({"h_eta": 1.5})

    def test_theta_positive(self):
        """Theta (job contract length) must be at least 1."""
        # Valid
        ConfigValidator._validate_ranges({"theta": 5})

        # Too small
        with pytest.raises(ValueError, match="must be >= 1"):
            ConfigValidator._validate_ranges({"theta": 0})

    def test_beta_positive(self):
        """Beta must be positive."""
        # Valid
        ConfigValidator._validate_ranges({"beta": 2.5})

        # Too small
        with pytest.raises(ValueError, match="must be >= 0.0"):
            ConfigValidator._validate_ranges({"beta": -1.0})

    def test_delta_in_range(self):
        """Delta (dividend payout ratio) must be between 0 and 1."""
        # Valid
        ConfigValidator._validate_ranges({"delta": 0.4})

        # Too large
        with pytest.raises(ValueError, match="must be <= 1.0"):
            ConfigValidator._validate_ranges({"delta": 1.5})

    def test_search_frictions_positive(self):
        """max_M, max_H, max_Z must be at least 1."""
        with pytest.raises(ValueError, match="must be >= 1"):
            ConfigValidator._validate_ranges({"max_M": 0})

        with pytest.raises(ValueError, match="must be >= 1"):
            ConfigValidator._validate_ranges({"max_Z": -1})

    def test_initial_values_non_negative(self):
        """Initial values must be non-negative."""
        # Valid
        ConfigValidator._validate_ranges({"price_init": 1.0, "net_worth_init": 12.0})

        # Negative
        with pytest.raises(ValueError, match="must be >= 0.0"):
            ConfigValidator._validate_ranges({"price_init": -1.0})

    def test_none_values_skipped(self):
        """None values should be skipped in range validation."""
        # Should not raise even though None is not in range
        ConfigValidator._validate_ranges({"cap_factor": None})


class TestRelationshipValidation:
    """Test cross-parameter constraint validation."""

    def test_warns_if_more_firms_than_households(self):
        """Should warn if n_households < n_firms."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConfigValidator._validate_relationships(
                {"n_firms": 100, "n_households": 50}
            )

            assert len(w) == 1
            assert "unemployment" in str(w[0].message).lower()

    def test_no_warning_if_more_households(self):
        """Should not warn if n_households >= n_firms."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConfigValidator._validate_relationships(
                {"n_firms": 100, "n_households": 500}
            )

            # No warnings related to unemployment
            unemployment_warnings = [
                x for x in w if "unemployment" in str(x.message).lower()
            ]
            assert len(unemployment_warnings) == 0

    def test_warns_if_min_wage_too_high(self):
        """Should warn if min_wage >= wage_offer_init."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConfigValidator._validate_relationships(
                {"min_wage": 1.0, "wage_offer_init": 0.9}
            )

            assert len(w) == 1
            assert "hire" in str(w[0].message).lower()

    def test_no_warning_if_min_wage_reasonable(self):
        """Should not warn if min_wage < wage_offer_init."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConfigValidator._validate_relationships(
                {"min_wage": 0.6, "wage_offer_init": 0.9}
            )

            # No warnings related to hiring
            hire_warnings = [x for x in w if "hire" in str(x.message).lower()]
            assert len(hire_warnings) == 0


class TestLoggingValidation:
    """Test logging configuration validation."""

    def test_valid_default_level(self):
        """Valid log levels should be accepted."""
        log_config = {"default_level": "DEBUG", "events": {}}
        # Should not raise
        ConfigValidator._validate_logging(log_config)

    def test_invalid_default_level(self):
        """Invalid log levels should be rejected."""
        log_config = {"default_level": "INVALID"}
        with pytest.raises(ValueError, match="Invalid log level"):
            ConfigValidator._validate_logging(log_config)

    def test_case_insensitive_level(self):
        """Log levels should be case-insensitive."""
        log_config = {"default_level": "debug"}
        # Should not raise (converted to uppercase)
        ConfigValidator._validate_logging(log_config)

    def test_valid_event_overrides(self):
        """Valid event log level overrides should be accepted."""
        log_config = {
            "default_level": "INFO",
            "events": {
                "workers_send_one_round": "DEBUG",
                "firms_hire_workers": "WARNING",
            },
        }
        # Should not raise
        ConfigValidator._validate_logging(log_config)

    def test_invalid_event_level(self):
        """Invalid event log levels should be rejected."""
        log_config = {"events": {"workers_send_one_round": "INVALID"}}
        with pytest.raises(ValueError, match="Invalid log level"):
            ConfigValidator._validate_logging(log_config)

    def test_events_must_be_dict(self):
        """Events must be a dictionary."""
        log_config = {"events": ["workers_send_one_round"]}
        with pytest.raises(ValueError, match="must be dict"):
            ConfigValidator._validate_logging(log_config)

    def test_default_level_must_be_string(self):
        """default_level must be a string."""
        log_config = {"default_level": 123}
        with pytest.raises(ValueError, match="must be str"):
            ConfigValidator._validate_logging(log_config)


class TestPipelinePathValidation:
    """Test pipeline path validation."""

    def test_valid_path(self):
        """Valid pipeline path should be accepted."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("events:\n  - firms_decide_desired_production\n")
            path = f.name

        try:
            # Should not raise
            ConfigValidator.validate_pipeline_path(path)
        finally:
            Path(path).unlink()

    def test_nonexistent_path(self):
        """Nonexistent path should raise ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            ConfigValidator.validate_pipeline_path("/nonexistent/path.yml")

    def test_directory_not_file(self):
        """Directory instead of file should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="is not a file"):
                ConfigValidator.validate_pipeline_path(tmpdir)

    def test_warns_non_yaml_extension(self):
        """Non-YAML extension should warn."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("content")
            path = f.name

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ConfigValidator.validate_pipeline_path(path)

                assert len(w) == 1
                assert "yml" in str(w[0].message).lower()
        finally:
            Path(path).unlink()


class TestPipelineYamlValidation:
    """Test pipeline YAML structure validation."""

    def test_valid_pipeline_yaml(self):
        """Valid pipeline YAML should be accepted."""
        yaml_content = """
events:
  - firms_decide_desired_production
  - firms_calc_breakeven_price
  - firms_adjust_price
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            # Should not raise (need to import events first to populate registry)
            import bamengine.events  # noqa: F401

            ConfigValidator.validate_pipeline_yaml(path)
        finally:
            Path(path).unlink()

    def test_missing_events_key(self):
        """YAML without 'events' key should raise ValueError."""
        yaml_content = """
config:
  - something
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            with pytest.raises(ValueError, match="must have 'events' key"):
                ConfigValidator.validate_pipeline_yaml(path)
        finally:
            Path(path).unlink()

    def test_events_not_list(self):
        """Events key must be a list."""
        yaml_content = """
events:
  event1: value
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            with pytest.raises(ValueError, match="must be a list"):
                ConfigValidator.validate_pipeline_yaml(path)
        finally:
            Path(path).unlink()

    def test_unknown_event_name(self):
        """Unknown event names should raise ValueError."""
        yaml_content = """
events:
  - nonexistent_event
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            import bamengine.events  # noqa: F401

            with pytest.raises(ValueError, match="not found in registry"):
                ConfigValidator.validate_pipeline_yaml(path)
        finally:
            Path(path).unlink()

    def test_parameter_substitution(self):
        """Parameter placeholders should be substituted and validated."""
        yaml_content = """
events:
  - firms_decide_desired_production
  - consumers_shop_one_round x {max_Z}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            import bamengine.events  # noqa: F401

            # Should substitute {max_Z} with 2
            ConfigValidator.validate_pipeline_yaml(path, params={"max_Z": 2})
        finally:
            Path(path).unlink()

    def test_unsubstituted_placeholder(self):
        """Unsubstituted placeholders should raise ValueError."""
        yaml_content = """
events:
  - workers_send_one_round_{i}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            # No params provided, so {i} won't be substituted
            with pytest.raises(ValueError, match="unsubstituted placeholders"):
                ConfigValidator.validate_pipeline_yaml(path)
        finally:
            Path(path).unlink()

    def test_repeated_event_syntax(self):
        """Repeated event syntax should be validated."""
        yaml_content = """
events:
  - firms_decide_desired_production x 3
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            import bamengine.events  # noqa: F401

            ConfigValidator.validate_pipeline_yaml(path)
        finally:
            Path(path).unlink()

    def test_interleaved_event_syntax(self):
        """Interleaved event syntax should be validated."""
        yaml_content = """
events:
  - workers_send_one_round <-> firms_hire_workers x 2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            import bamengine.events  # noqa: F401

            ConfigValidator.validate_pipeline_yaml(path)
        finally:
            Path(path).unlink()


class TestFullConfigValidation:
    """Test full config validation via validate_config()."""

    def test_valid_config(self):
        """Valid configuration should pass all validations."""
        cfg = {
            "n_firms": 100,
            "n_households": 500,
            "n_banks": 10,
            "h_rho": 0.1,
            "beta": 2.5,
            "delta": 0.4,
            "max_M": 4,
        }
        # Should not raise
        ConfigValidator.validate_config(cfg)

    def test_invalid_type_caught(self):
        """Invalid types should be caught."""
        cfg = {"n_firms": "100"}
        with pytest.raises(ValueError, match="must be int"):
            ConfigValidator.validate_config(cfg)

    def test_invalid_range_caught(self):
        """Invalid ranges should be caught."""
        cfg = {"n_firms": -1}
        with pytest.raises(ValueError, match="must be >= 1"):
            ConfigValidator.validate_config(cfg)

    def test_logging_config_validated(self):
        """Logging configuration should be validated."""
        cfg = {"logging": {"default_level": "INVALID"}}
        with pytest.raises(ValueError, match="Invalid log level"):
            ConfigValidator.validate_config(cfg)


class TestIntegrationWithSimulation:
    """Test that validation is integrated into Simulation.init()."""

    def test_invalid_config_rejected_at_init(self):
        """Invalid configuration should be rejected at Simulation.init()."""
        with pytest.raises(ValueError, match="must be >= 1"):
            Simulation.init(n_firms=-1)

    def test_invalid_type_rejected_at_init(self):
        """Invalid types should be rejected at Simulation.init()."""
        with pytest.raises(ValueError, match="must be int"):
            Simulation.init(n_firms="100")

    def test_valid_config_accepted_at_init(self):
        """Valid configuration should be accepted."""
        # Should not raise
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        assert sim.n_firms == 10

    def test_invalid_pipeline_path_rejected(self):
        """Invalid pipeline path should be rejected."""
        with pytest.raises(ValueError, match="does not exist"):
            Simulation.init(pipeline_path="/nonexistent/path.yml")

    def test_valid_custom_pipeline_accepted(self):
        """Valid custom pipeline should be accepted."""
        import bamengine.events  # noqa: F401

        yaml_content = """
events:
  - firms_decide_desired_production
  - firms_calc_breakeven_price
  - firms_adjust_price
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            sim = Simulation.init(
                n_firms=10, n_households=50, pipeline_path=path, seed=42
            )
            assert len(sim.pipeline) == 3
        finally:
            Path(path).unlink()
