"""
Centralized configuration validation for BAM Engine.

This module provides the ConfigValidator class, which performs all validation
once at Simulation.init() to ensure fail-fast behavior with clear error messages.

Validation Layers
-----------------
1. **Type Checking**: Ensures all parameters have correct types
   (int, float, str, etc.)
2. **Range Validation**: Ensures parameters are within valid ranges
   (e.g., 0 <= h_rho <= 1)
3. **Boolean and Enum Validation**: Ensures implementation variant
   parameters have valid values
4. **Relationship Constraints**: Validates cross-parameter dependencies
   (e.g., warns if n_households < n_firms)
5. **Pipeline Validation**: Validates custom pipeline YAML files
   (structure, event names, parameter substitution)
6. **Logging Validation**: Validates log levels and event names

Benefits
--------
- **Fail-fast**: Invalid configurations rejected immediately
  with clear error messages
- **Single location**: All validation logic in one place
  (no scattered checks)
- **No runtime overhead**: Validation happens once at initialization,
  not during simulation
- **Clear errors**: Error messages include parameter name, expected range,
  and actual value

Examples
--------
Type errors are caught immediately:

>>> from bamengine.config import ConfigValidator
>>> cfg = {"n_firms": "100"}
>>> ConfigValidator._validate_types(cfg)  # doctest: +SKIP
ValueError: Config parameter 'n_firms' must be int, got str

Range errors provide clear feedback:

>>> cfg = {"h_rho": 1.5}
>>> ConfigValidator._validate_ranges(cfg)  # doctest: +SKIP
ValueError: Config parameter 'h_rho' must be <= 1.0, got 1.5

Cross-parameter warnings help avoid common mistakes:

>>> cfg = {"n_firms": 100, "n_households": 50}
>>> ConfigValidator._validate_relationships(cfg)  # doctest: +SKIP
UserWarning: n_households (50) < n_firms (100). This may lead to high unemployment...

Pipeline validation catches unknown events:

>>> ConfigValidator.validate_pipeline_yaml("custom.yml")  # doctest: +SKIP
ValueError: Event 'nonexistent_event' not found in registry...

See Also
--------
Config : Immutable configuration dataclass
bamengine.simulation.Simulation.init : Uses ConfigValidator before initialization
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


class ConfigValidator:
    """
    Centralized validation for simulation configuration.

    All validation happens once at Simulation.init() to ensure:

    - Type correctness
    - Valid parameter ranges
    - Relationship constraints between parameters
    - Clear error messages with actionable feedback

    This class is stateless and uses only static methods. All validation
    logic is centralized here rather than scattered across the codebase.

    Examples
    --------
    Validate a complete configuration:

    >>> from bamengine.config import ConfigValidator
    >>> cfg = {
    ...     "n_firms": 100,
    ...     "n_households": 500,
    ...     "h_rho": 0.1,
    ...     "beta": 2.5,
    ... }
    >>> ConfigValidator.validate_config(cfg)  # No exception = valid

    Validate only types:

    >>> cfg = {"n_firms": 100, "h_rho": 0.1}
    >>> ConfigValidator._validate_types(cfg)  # Valid

    >>> cfg = {"n_firms": "100"}
    >>> ConfigValidator._validate_types(cfg)  # doctest: +SKIP
    ValueError: Config parameter 'n_firms' must be int, got str

    Validate only ranges:

    >>> cfg = {"h_rho": 0.5}
    >>> ConfigValidator._validate_ranges(cfg)  # Valid

    >>> cfg = {"h_rho": 1.5}
    >>> ConfigValidator._validate_ranges(cfg)  # doctest: +SKIP
    ValueError: Config parameter 'h_rho' must be <= 1.0, got 1.5

    Notes
    -----
    The ConfigValidator ensures validation happens once, upfront, rather than
    during simulation execution. This provides better error messages and
    avoids runtime overhead.

    See Also
    --------
    Config : Configuration dataclass validated by this class
    bamengine.simulation.Simulation.init : Entry point that triggers validation
    """

    # Valid log levels for logging configuration
    VALID_LOG_LEVELS = {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    @staticmethod
    def validate_config(cfg: dict[str, Any]) -> None:
        """
        Validate all configuration parameters.

        This is the main entry point for configuration validation. It runs all
        validation checks in sequence: types, ranges, relationships, and logging.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary to validate. Should contain simulation
            parameters like n_firms, n_households, h_rho, etc.

        Raises
        ------
        ValueError
            If any validation check fails (type error, out of range, invalid
            pipeline/logging config).

        Examples
        --------
        Valid configuration passes silently:

        >>> from bamengine.config import ConfigValidator
        >>> cfg = {"n_firms": 100, "h_rho": 0.1, "beta": 2.5}
        >>> ConfigValidator.validate_config(cfg)  # No exception

        Invalid type raises ValueError:

        >>> cfg = {"n_firms": "100"}
        >>> ConfigValidator.validate_config(cfg)  # doctest: +SKIP
        ValueError: Config parameter 'n_firms' must be int, got str

        Out of range raises ValueError:

        >>> cfg = {"h_rho": 1.5}
        >>> ConfigValidator.validate_config(cfg)  # doctest: +SKIP
        ValueError: Config parameter 'h_rho' must be <= 1.0, got 1.5

        Notes
        -----
        This method is called by Simulation.init() before any initialization
        occurs, ensuring fail-fast behavior with clear error messages.

        See Also
        --------
        _validate_types : Type checking for configuration parameters
        _validate_ranges : Range validation for configuration parameters
        _validate_relationships : Cross-parameter constraint validation
        _validate_logging : Logging configuration validation
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

        Validates that all present configuration parameters have the correct
        types: int for counts, float for rates/shocks, str for paths, etc.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary to validate.

        Raises
        ------
        ValueError
            If any parameter has incorrect type. Error message includes
            parameter name, expected type, and actual type.

        Examples
        --------
        Valid types pass silently:

        >>> from bamengine.config import ConfigValidator
        >>> cfg = {"n_firms": 100, "h_rho": 0.1}
        >>> ConfigValidator._validate_types(cfg)

        Invalid integer type raises ValueError:

        >>> cfg = {"n_firms": "100"}
        >>> ConfigValidator._validate_types(cfg)  # doctest: +SKIP
        ValueError: Config parameter 'n_firms' must be int, got str

        Float parameters accept int or float:

        >>> cfg = {"h_rho": 1, "beta": 2.5}
        >>> ConfigValidator._validate_types(cfg)

        Notes
        -----
        Missing parameters are allowed (validation only checks present keys).
        Default values are handled by Simulation.init() before validation.

        See Also
        --------
        validate_config : Main validation entry point
        """
        # Integer parameters (seed handled separately)
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
            "contract_poisson_mean",
        ]

        # Float parameters (scalars only)
        float_params = [
            "h_rho",
            "h_xi",
            "h_phi",
            "h_eta",
            "labor_productivity",
            "beta",
            "delta",
            "v",
            "r_bar",
            "min_wage_ratio",
            "net_worth_ratio",
            "max_loan_to_net_worth",
            "max_leverage",
            "new_firm_size_factor",
            "new_firm_production_factor",
            "new_firm_wage_factor",
            "new_firm_price_markup",
        ]

        # Vector parameters (can be scalar or 1D array)
        vector_params = [
            "price_init",
            "savings_init",
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

        # Check vector parameters (accept int, float, or array-like)
        for key in vector_params:
            if key not in cfg:
                continue
            val = cfg[key]
            # Accept scalars or anything np.asarray can handle
            if not isinstance(val, (int, float)):
                try:
                    np.asarray(val)  # Verify it's array-like
                except (ValueError, TypeError) as err:
                    raise ValueError(
                        f"Config parameter '{key}' must be float or array-like, "
                        f"got {type(val).__name__}"
                    ) from err

        # Check optional float `cap_factor`
        if "cap_factor" in cfg:
            val = cfg["cap_factor"]
            if val is not None and not isinstance(val, (int, float)):
                raise ValueError(
                    f"Config parameter 'cap_factor' must be float or None, "
                    f"got {type(val).__name__}"
                )

        # Check boolean implementation variant parameters
        bool_params = [
            "price_cut_allow_increase",
        ]
        for key in bool_params:
            if key not in cfg:
                continue
            val = cfg[key]
            if not isinstance(val, bool):
                raise ValueError(
                    f"Config parameter '{key}' must be bool, got {type(val).__name__}"
                )

        # Check string enum implementation variant parameters
        str_enum_params = [
            "loan_priority_method",
            "firing_method",
            "matching_method",
            "job_search_method",
        ]
        for key in str_enum_params:
            if key not in cfg:
                continue
            val = cfg[key]
            if not isinstance(val, str):
                raise ValueError(
                    f"Config parameter '{key}' must be str, got {type(val).__name__}"
                )

        # Check pipeline_path (str or None)
        if "pipeline_path" in cfg:
            val = cfg["pipeline_path"]
            if val is not None and not isinstance(val, str):
                raise ValueError(
                    f"Config parameter 'pipeline_path' must be str or None, "
                    f"got {type(val).__name__}"
                )

        # Check seed (int or np.random.Generator)
        if "seed" in cfg:
            val = cfg["seed"]
            if val is not None and not isinstance(val, (int, np.random.Generator)):
                raise ValueError(
                    f"Config parameter 'seed' must be int or np.random.Generator, "
                    f"got {type(val).__name__}"
                )

    @staticmethod
    def _validate_ranges(cfg: dict[str, Any]) -> None:
        """
        Ensure parameters are in valid ranges.

        Validates that all present configuration parameters fall within
        their valid ranges (e.g., 0 <= h_rho <= 1, n_firms >= 1).

        Parameters
        ----------
        cfg : dict
            Configuration dictionary to validate.

        Raises
        ------
        ValueError
            If any parameter is out of valid range. Error message includes
            parameter name, valid range, and actual value.

        Examples
        --------
        Valid ranges pass silently:

        >>> from bamengine.config import ConfigValidator
        >>> cfg = {"n_firms": 100, "h_rho": 0.5}
        >>> ConfigValidator._validate_ranges(cfg)

        Value below minimum raises ValueError:

        >>> cfg = {"n_firms": 0}
        >>> ConfigValidator._validate_ranges(cfg)  # doctest: +SKIP
        ValueError: Config parameter 'n_firms' must be >= 1, got 0

        Value above maximum raises ValueError:

        >>> cfg = {"h_rho": 1.5}
        >>> ConfigValidator._validate_ranges(cfg)  # doctest: +SKIP
        ValueError: Config parameter 'h_rho' must be <= 1.0, got 1.5

        Notes
        -----
        Constraints are defined internally as (min_val, max_val) tuples.
        None means unbounded in that direction.

        See Also
        --------
        validate_config : Main validation entry point
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
            # Labor productivity (positive)
            "labor_productivity": (0.0, None),
            # Contract length (positive)
            "theta": (1, None),
            # Consumption propensity exponent (positive)
            "beta": (0.0, None),
            # Dividend payout ratio (0 to 1)
            "delta": (0.0, 1.0),
            # Bank capital requirement (positive, typically < 1)
            "v": (0.0, 1.0),
            # Interest rate (typically small positive)
            "r_bar": (0.0, 1.0),
            # Minimum wage revision period (positive)
            "min_wage_rev_period": (1, None),
            # Initial values (must be positive)
            "price_init": (0.0, None),
            "savings_init": (0.0, None),
            "equity_base_init": (0.0, None),
            "min_wage_ratio": (0.0, 1.0),
            "net_worth_ratio": (0.0, None),
            # Implementation variant parameters
            "contract_poisson_mean": (0, None),
            "max_loan_to_net_worth": (0.0, None),
            "max_leverage": (1.0, None),
            "cap_factor": (1.0, None),
            # New firm parameters (non-negative)
            "new_firm_size_factor": (0.0, None),
            "new_firm_production_factor": (0.0, None),
            "new_firm_wage_factor": (0.0, None),
            "new_firm_price_markup": (0.0, None),
        }

        # Valid values for string enum parameters
        valid_enums = {
            "loan_priority_method": {"by_net_worth", "by_leverage", "by_appearance"},
            "firing_method": {"random", "expensive"},
            "matching_method": {"sequential", "simultaneous"},
            "job_search_method": {"vacancies_only", "all_firms"},
        }

        # Vector parameters (validated separately by _validate_float1d)
        vector_params = {
            "price_init",
            "savings_init",
            "equity_base_init",
        }

        for key, (min_val, max_val) in constraints.items():
            if key not in cfg:
                continue

            val = cfg[key]

            # Skip None values for optional parameters
            if val is None:
                continue

            # Skip vector parameters (arrays) - validated later by _validate_float1d
            if key in vector_params and not isinstance(val, (int, float)):
                continue

            # Check minimum (all current constraints have min_val, but keeping check for future)
            if min_val is not None and val < min_val:  # type: ignore[redundant-expr]
                raise ValueError(
                    f"Config parameter '{key}' must be >= {min_val}, got {val}"
                )

            # Check maximum
            if max_val is not None and val > max_val:
                raise ValueError(
                    f"Config parameter '{key}' must be <= {max_val}, got {val}"
                )

        # Validate string enum parameters
        for key, valid_values in valid_enums.items():
            if key not in cfg:
                continue
            val = cfg[key]
            if val not in valid_values:
                raise ValueError(
                    f"Config parameter '{key}' must be one of {valid_values}, got '{val}'"
                )

    @staticmethod
    def _validate_relationships(cfg: dict[str, Any]) -> None:
        """
        Validate cross-parameter constraints.

        Checks for unusual parameter combinations that may lead to issues,
        issuing warnings (not errors) to help users avoid common mistakes.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary to validate.

        Warns
        -----
        UserWarning
            If n_households < n_firms (may lead to high unemployment).
            If min_wage >= wage_offer_init (firms may not be able to hire).

        Examples
        --------
        Unusual configurations trigger warnings:

        >>> from bamengine.config import ConfigValidator
        >>> import warnings
        >>> cfg = {"n_firms": 100, "n_households": 50}
        >>> with warnings.catch_warnings(record=True) as w:
        ...     warnings.simplefilter("always")
        ...     ConfigValidator._validate_relationships(cfg)
        ...     assert len(w) == 1
        ...     assert "high unemployment" in str(w[0].message)

        >>> cfg = {"min_wage": 10.0, "wage_offer_init": 5.0}
        >>> with warnings.catch_warnings(record=True) as w:
        ...     warnings.simplefilter("always")
        ...     ConfigValidator._validate_relationships(cfg)
        ...     assert len(w) == 1
        ...     assert "may not be able to hire" in str(w[0].message)

        Notes
        -----
        These checks issue warnings, not errors, because the configurations
        may be intentional for specific experiments or edge case testing.

        See Also
        --------
        validate_config : Main validation entry point
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

        Checks that log levels are valid and event names exist in the registry.

        Parameters
        ----------
        log_config : dict
            Logging configuration dictionary with keys:

            - default_level : str, optional
                Default log level (e.g., 'INFO', 'DEBUG')
            - log_file : str or None, optional
                Path to log file for saving output (None = console only)
            - events : dict[str, str], optional
                Per-event log level overrides (event_name -> level)

        Raises
        ------
        ValueError
            If default_level or any event level is invalid.
            If any parameter has wrong type (not str or dict).

        Examples
        --------
        Valid logging config passes silently:

        >>> from bamengine.config import ConfigValidator
        >>> log_cfg = {"default_level": "INFO"}
        >>> ConfigValidator._validate_logging(log_cfg)

        >>> log_cfg = {
        ...     "default_level": "DEBUG",
        ...     "log_file": "simulation.log",
        ...     "events": {"workers_send_one_round": "WARNING"},
        ... }
        >>> ConfigValidator._validate_logging(log_cfg)

        Invalid log level raises ValueError:

        >>> log_cfg = {"default_level": "INVALID"}
        >>> ConfigValidator._validate_logging(log_cfg)  # doctest: +SKIP
        ValueError: Invalid log level 'INVALID'. Must be one of ...

        Notes
        -----
        Valid log levels: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL.
        Case-insensitive comparison (INFO == info == Info).

        See Also
        --------
        validate_config : Main validation entry point
        bamengine.logging : Custom logging with TRACE level
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

        # Check log_file (optional string path or None)
        if "log_file" in log_config:
            log_file = log_config["log_file"]
            if log_file is not None and not isinstance(log_file, str):
                raise ValueError(
                    f"Logging log_file must be str or None, got {type(log_file).__name__}"
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

        Checks that the given path points to an existing, readable file.
        Issues warning if file doesn't have .yml or .yaml extension.

        Parameters
        ----------
        pipeline_path : str
            Path to pipeline YAML file.

        Raises
        ------
        ValueError
            If path does not exist or is not a file.

        Warns
        -----
        UserWarning
            If file doesn't have .yml or .yaml extension.

        Examples
        --------
        Valid path passes silently:

        >>> from bamengine.config import ConfigValidator
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as f:
        ...     f.write(b"events: []")
        ...     path = f.name
        >>> ConfigValidator.validate_pipeline_path(path)
        >>> Path(path).unlink()  # cleanup

        Non-existent path raises ValueError:

        >>> ConfigValidator.validate_pipeline_path("/nonexistent.yml")  # doctest: +SKIP
        ValueError: Pipeline path '/nonexistent.yml' does not exist

        Notes
        -----
        This method only checks path validity, not YAML structure or event
        names. Use validate_pipeline_yaml() for full validation.

        See Also
        --------
        validate_pipeline_yaml : Validate pipeline YAML structure and events
        """
        from pathlib import Path

        path = Path(pipeline_path)

        if not path.exists():
            raise ValueError(f"Pipeline path '{pipeline_path}' does not exist")

        if not path.is_file():
            raise ValueError(f"Pipeline path '{pipeline_path}' is not a file")

        if path.suffix not in [".yml", ".yaml"]:
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

        Checks that:

        1. YAML is valid and has 'events' key
        2. All event names exist in the registry
        3. All parameter placeholders can be substituted
        4. Event spec syntax is valid (repeat, interleave)

        Parameters
        ----------
        yaml_path : str
            Path to pipeline YAML file.
        params : dict[str, int], optional
            Parameters available for substitution (e.g., {"max_M": 4}).
            If None, no parameter substitution is performed.

        Raises
        ------
        ValueError
            If YAML structure is invalid, references unknown events,
            or contains unsubstituted placeholders.

        Examples
        --------
        Valid pipeline YAML passes silently:

        >>> from bamengine.config import ConfigValidator
        >>> import tempfile
        >>> from pathlib import Path
        >>> yaml_content = '''
        ... events:
        ...   - firms_decide_desired_production
        ...   - workers_send_one_round x 4
        ... '''
        >>> with tempfile.NamedTemporaryFile(
        ...     mode="w", suffix=".yml", delete=False
        ... ) as f:
        ...     f.write(yaml_content)
        ...     path = f.name
        >>> ConfigValidator.validate_pipeline_yaml(path)
        >>> Path(path).unlink()  # cleanup

        Unknown event raises ValueError:

        >>> yaml_content = '''
        ... events:
        ...   - nonexistent_event
        ... '''
        >>> with tempfile.NamedTemporaryFile(
        ...     mode="w", suffix=".yml", delete=False
        ... ) as f:
        ...     f.write(yaml_content)
        ...     path = f.name
        >>> ConfigValidator.validate_pipeline_yaml(path)  # doctest: +SKIP
        ValueError: Event 'nonexistent_event' not found in registry...
        >>> Path(path).unlink()  # cleanup

        Notes
        -----
        This validation is called by Pipeline.from_yaml() after path validation.
        It ensures all event names are registered before attempting to load
        the pipeline.

        See Also
        --------
        validate_pipeline_path : Validate path exists
        bamengine.core.pipeline.Pipeline.from_yaml : Loads validated pipeline
        """
        from pathlib import Path

        import yaml

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
        Handles three spec formats:

        1. Single event: "event_name"
        2. Repeated event: "event_name x N"
        3. Interleaved events: "event1 <-> event2 x N"

        Parameters
        ----------
        spec : str
            Event specification string from pipeline YAML.

        Returns
        -------
        list[str]
            List of unique event names referenced in spec.

        Examples
        --------
        Single event:

        >>> from bamengine.config import ConfigValidator
        >>> ConfigValidator._parse_event_spec_for_validation("my_event")
        ['my_event']

        Repeated event:

        >>> ConfigValidator._parse_event_spec_for_validation("my_event x 4")
        ['my_event']

        Interleaved events:

        >>> ConfigValidator._parse_event_spec_for_validation("event1 <-> event2 x 3")
        ['event1', 'event2']

        Notes
        -----
        This parser only extracts names for validation. The actual pipeline
        expansion (repeating/interleaving) is done by Pipeline.from_yaml().

        See Also
        --------
        validate_pipeline_yaml : Uses this parser to validate event names
        bamengine.core.pipeline.Pipeline.from_yaml : Expands specs into full pipeline
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
