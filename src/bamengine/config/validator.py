"""Centralized configuration validation for BAM Engine."""

from __future__ import annotations

import logging
import warnings
from typing import Any


class ConfigValidator:
    """
    Centralized validation for simulation configuration.

    All validation happens once at Simulation.init() to ensure:
    - Type correctness
    - Valid parameter ranges
    - Relationship constraints between parameters
    - Clear error messages with actionable feedback
    """

    # Valid log levels for logging configuration
    VALID_LOG_LEVELS = {"DEEP_DEBUG", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    @staticmethod
    def validate_config(cfg: dict[str, Any]) -> None:
        """
        Validate all configuration parameters.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary to validate.

        Raises
        ------
        ValueError
            If any validation check fails.
        """
        # Type checking
        ConfigValidator._validate_types(cfg)

        # Range validation
        ConfigValidator._validate_ranges(cfg)

        # Relationship constraints
        ConfigValidator._validate_relationships(cfg)

        # Logging configuration
        if "logging" in cfg:
            ConfigValidator._validate_logging(cfg["logging"])

    @staticmethod
    def _validate_types(cfg: dict[str, Any]) -> None:
        """
        Ensure correct types for configuration parameters.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary to validate.

        Raises
        ------
        ValueError
            If any parameter has incorrect type.
        """
        # Integer parameters
        int_params = [
            "n_firms",
            "n_households",
            "n_banks",
            "n_periods",
            "max_M",
            "max_H",
            "max_Z",
            "theta",
            "min_wage_rev_period",
            "seed",
            "cap_factor",
        ]

        # Float parameters
        float_params = [
            "h_rho",
            "h_xi",
            "h_phi",
            "h_eta",
            "beta",
            "delta",
            "v",
            "r_bar",
            "min_wage",
            "net_worth_init",
            "production_init",
            "price_init",
            "savings_init",
            "wage_offer_init",
            "equity_base_init",
        ]

        # Check integers
        for key in int_params:
            if key not in cfg:
                continue
            val = cfg[key]
            if val is not None and not isinstance(val, int):
                raise ValueError(
                    f"Config parameter '{key}' must be int, got {type(val).__name__}"
                )

        # Check floats (accept int or float)
        for key in float_params:
            if key not in cfg:
                continue
            val = cfg[key]
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"Config parameter '{key}' must be float, got {type(val).__name__}"
                )

        # Check pipeline_path (str or None)
        if "pipeline_path" in cfg:
            val = cfg["pipeline_path"]
            if val is not None and not isinstance(val, str):
                raise ValueError(
                    f"Config parameter 'pipeline_path' must be str or None, "
                    f"got {type(val).__name__}"
                )

    @staticmethod
    def _validate_ranges(cfg: dict[str, Any]) -> None:
        """
        Ensure parameters are in valid ranges.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary to validate.

        Raises
        ------
        ValueError
            If any parameter is out of valid range.
        """
        # Define constraints as (min_val, max_val) tuples
        # None means unbounded
        constraints = {
            # Population sizes (must be positive)
            "n_firms": (1, None),
            "n_households": (1, None),
            "n_banks": (1, None),
            "n_periods": (1, None),
            # Shock parameters (0 to 1)
            "h_rho": (0.0, 1.0),
            "h_xi": (0.0, 1.0),
            "h_phi": (0.0, 1.0),
            "h_eta": (0.0, 1.0),
            # Search frictions (positive integers)
            "max_M": (1, None),
            "max_H": (1, None),
            "max_Z": (1, None),
            # Contract length (positive)
            "theta": (1, None),
            # Consumption propensity exponent (positive, typically < 10)
            "beta": (0.0, 10.0),
            # Dividend payout ratio (0 to 1)
            "delta": (0.0, 1.0),
            # Bank capital requirement (positive, typically < 1)
            "v": (0.0, 1.0),
            # Interest rate (typically small positive)
            "r_bar": (0.0, 1.0),
            # Minimum wage (positive)
            "min_wage": (0.0, None),
            # Minimum wage revision period (positive)
            "min_wage_rev_period": (1, None),
            # Initial values (must be positive)
            "net_worth_init": (0.0, None),
            "production_init": (0.0, None),
            "price_init": (0.0, None),
            "savings_init": (0.0, None),
            "wage_offer_init": (0.0, None),
            "equity_base_init": (0.0, None),
        }

        for key, (min_val, max_val) in constraints.items():
            if key not in cfg:
                continue

            val = cfg[key]

            # Skip None values for optional parameters
            if val is None:
                continue

            # Check minimum
            if min_val is not None and val < min_val:
                raise ValueError(
                    f"Config parameter '{key}' must be >= {min_val}, got {val}"
                )

            # Check maximum
            if max_val is not None and val > max_val:
                raise ValueError(
                    f"Config parameter '{key}' must be <= {max_val}, got {val}"
                )

    @staticmethod
    def _validate_relationships(cfg: dict[str, Any]) -> None:
        """
        Validate cross-parameter constraints.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary to validate.
        """
        # Warn if more firms than households (unusual configuration)
        n_firms = cfg.get("n_firms", 0)
        n_households = cfg.get("n_households", 0)

        if n_firms > 0 and n_households > 0 and n_households < n_firms:
            warnings.warn(
                f"n_households ({n_households}) < n_firms ({n_firms}). "
                "This may lead to high unemployment and labor shortages.",
                UserWarning,
                stacklevel=3,
            )

        # Warn if min_wage >= wage_offer_init
        min_wage = cfg.get("min_wage", 0.0)
        wage_offer_init = cfg.get("wage_offer_init", float("inf"))

        if min_wage >= wage_offer_init:
            warnings.warn(
                f"min_wage ({min_wage}) >= wage_offer_init ({wage_offer_init}). "
                "Firms may not be able to hire workers at initialization.",
                UserWarning,
                stacklevel=3,
            )

    @staticmethod
    def _validate_logging(log_config: dict[str, Any]) -> None:
        """
        Validate logging configuration.

        Parameters
        ----------
        log_config : dict
            Logging configuration dictionary with keys:
            - default_level: str (e.g., 'INFO', 'DEBUG')
            - events: dict[str, str] (per-event overrides)

        Raises
        ------
        ValueError
            If logging configuration is invalid.
        """
        # Check default_level
        if "default_level" in log_config:
            level = log_config["default_level"]
            if not isinstance(level, str):
                raise ValueError(
                    f"Logging default_level must be str, got {type(level).__name__}"
                )

            level_upper = level.upper()
            if level_upper not in ConfigValidator.VALID_LOG_LEVELS:
                raise ValueError(
                    f"Invalid log level '{level}'. "
                    f"Must be one of {ConfigValidator.VALID_LOG_LEVELS}"
                )

        # Check events dictionary
        if "events" in log_config:
            events = log_config["events"]
            if not isinstance(events, dict):
                raise ValueError(
                    f"Logging events must be dict, got {type(events).__name__}"
                )

            for event_name, level in events.items():
                if not isinstance(event_name, str):
                    raise ValueError(
                        f"Event name must be str, got {type(event_name).__name__}"
                    )

                if not isinstance(level, str):
                    raise ValueError(
                        f"Log level for event '{event_name}' must be str, "
                        f"got {type(level).__name__}"
                    )

                level_upper = level.upper()
                if level_upper not in ConfigValidator.VALID_LOG_LEVELS:
                    raise ValueError(
                        f"Invalid log level '{level}' for event '{event_name}'. "
                        f"Must be one of {ConfigValidator.VALID_LOG_LEVELS}"
                    )

    @staticmethod
    def validate_pipeline_path(pipeline_path: str) -> None:
        """
        Validate pipeline path exists and is readable.

        Parameters
        ----------
        pipeline_path : str
            Path to pipeline YAML file.

        Raises
        ------
        ValueError
            If path does not exist or is not readable.
        """
        from pathlib import Path

        path = Path(pipeline_path)

        if not path.exists():
            raise ValueError(f"Pipeline path '{pipeline_path}' does not exist")

        if not path.is_file():
            raise ValueError(f"Pipeline path '{pipeline_path}' is not a file")

        if not path.suffix in [".yml", ".yaml"]:
            warnings.warn(
                f"Pipeline path '{pipeline_path}' does not have .yml/.yaml extension",
                UserWarning,
                stacklevel=2,
            )

    @staticmethod
    def validate_pipeline_yaml(
        yaml_path: str, params: dict[str, int] | None = None
    ) -> None:
        """
        Validate pipeline YAML file structure and event references.

        Parameters
        ----------
        yaml_path : str
            Path to pipeline YAML file.
        params : dict[str, int], optional
            Parameters available for substitution (e.g., {"max_M": 4}).

        Raises
        ------
        ValueError
            If YAML structure is invalid or references unknown events.
        """
        import yaml
        from pathlib import Path
        from bamengine.core.registry import list_events

        params = params or {}

        # Read YAML
        path = Path(yaml_path)
        with open(path) as f:
            config = yaml.safe_load(f)

        # Check for 'events' key
        if not isinstance(config, dict):
            raise ValueError(
                f"Pipeline YAML must be a dictionary, got {type(config).__name__}"
            )

        if "events" not in config:
            raise ValueError(f"Pipeline YAML must have 'events' key: {yaml_path}")

        event_specs = config["events"]

        if not isinstance(event_specs, list):
            raise ValueError(
                f"Pipeline 'events' must be a list, got {type(event_specs).__name__}"
            )

        # Get all registered event names
        registered_events = set(list_events())

        # Parse and validate each event spec
        for i, spec in enumerate(event_specs):
            if not isinstance(spec, str):
                raise ValueError(
                    f"Event spec at index {i} must be str, got {type(spec).__name__}"
                )

            # Substitute parameters
            substituted_spec = spec
            for param_name, param_value in params.items():
                substituted_spec = substituted_spec.replace(
                    f"{{{param_name}}}", str(param_value)
                )

            # Check for unsubstituted placeholders
            if "{" in substituted_spec or "}" in substituted_spec:
                raise ValueError(
                    f"Event spec '{spec}' contains unsubstituted placeholders. "
                    f"Available params: {list(params.keys())}"
                )

            # Parse spec to extract event names
            event_names = ConfigValidator._parse_event_spec_for_validation(
                substituted_spec
            )

            # Validate each event name exists in registry
            for name in event_names:
                if name not in registered_events:
                    raise ValueError(
                        f"Event '{name}' (from spec '{spec}') not found in registry. "
                        f"Available events: {sorted(registered_events)}"
                    )

    @staticmethod
    def _parse_event_spec_for_validation(spec: str) -> list[str]:
        """
        Parse event spec to extract event names for validation.

        This is a simplified parser that extracts event names without
        expanding repeats (we just need to check the names exist).

        Parameters
        ----------
        spec : str
            Event specification string.

        Returns
        -------
        list[str]
            List of unique event names referenced in spec.
        """
        import re

        spec = spec.strip()

        # Pattern 1: Interleaved events (event1 <-> event2 x N)
        interleaved_pattern = r"^(.+?)\s*<->\s*(.+?)\s+x\s+(\d+)$"
        match = re.match(interleaved_pattern, spec)
        if match:
            event1 = match.group(1).strip()
            event2 = match.group(2).strip()
            return [event1, event2]

        # Pattern 2: Repeated event (event_name x N)
        repeated_pattern = r"^(.+?)\s+x\s+(\d+)$"
        match = re.match(repeated_pattern, spec)
        if match:
            event_name = match.group(1).strip()
            return [event_name]

        # Pattern 3: Single event (event_name)
        return [spec]
