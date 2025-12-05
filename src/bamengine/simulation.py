"""
Main simulation facade for BAM Engine.

This module provides the Simulation class, the primary interface for running
BAM (Bottom-Up Adaptive Macroeconomics) simulations. The Simulation class
manages the economy state, agent roles, event pipeline, and provides methods
for stepping through periods.

Key Features
------------
- Three-tier configuration precedence (defaults → user config → kwargs)
- Deterministic random number generation with seed control
- Event pipeline with explicit ordering and YAML configuration
- Getter methods for roles, events, and relationships (case-insensitive)
- In-place state mutation for memory efficiency
- Built-in logging configuration at global and per-event levels

Classes
-------
Simulation
    Main simulation facade for initializing and running BAM simulations.

Functions
---------
_read_yaml
    Load configuration from YAML file, dict, or None.
_package_defaults
    Load default configuration from bamengine/config/defaults.yml.
_validate_float1d
    Validate 1D float array or scalar for initialization.

See Also
--------
:mod:`bamengine.config` : Configuration dataclass and validation
:mod:`bamengine.core` : Event and Pipeline infrastructure
:mod:`bamengine.roles` : Agent role components (Producer, Worker, etc.)
:mod:`bamengine.events` : Event classes wrapping system functions

Examples
--------
Basic simulation with default configuration:

>>> import bamengine as bam
>>> sim = bam.Simulation.init(seed=42)
>>> sim.run(n_periods=100)
>>> unemployment = sim.ec.unemp_rate_history[-1]

Custom configuration via YAML file:

>>> sim = bam.Simulation.init(config="my_config.yml", seed=42)
>>> sim.run(n_periods=100)

Override specific parameters via kwargs:

>>> sim = bam.Simulation.init(n_firms=200, n_households=1000, seed=42)
>>> sim.run(n_periods=100)

Step-by-step execution with intermediate analysis:

>>> sim = bam.Simulation.init(seed=42)
>>> for period in range(100):
...     sim.step()
...     if period % 10 == 0:
...         print(
...             f"Period {period}: Unemployment = {sim.ec.unemp_rate_history[-1]:.2%}"
...         )
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:  # pragma: no cover
    from bamengine.results import SimulationResults, _DataCollector

import bamengine.events  # noqa: F401 - needed to register events
from bamengine import Rng, logging, make_rng
from bamengine.config import Config
from bamengine.core.pipeline import Pipeline, create_default_pipeline
from bamengine.economy import Economy

# Import roles BEFORE relationships (LoanBook needs roles to be registered first)
from bamengine.roles import Borrower, Consumer, Employer, Lender, Producer, Worker

# isort: off
from bamengine.relationships import LoanBook  # Must import after roles

# isort: on
from bamengine.typing import Float1D

__all__ = ["Simulation"]

log = logging.getLogger(__name__)


# helpers
# ---------------------------------------------------------------------------
def _read_yaml(obj: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    """
    Load configuration from YAML file, dict, or None.

    Parameters
    ----------
    obj : str, Path, Mapping, or None
        Configuration source (file path, dict, or None).

    Returns
    -------
    dict
        Configuration dictionary (empty dict if obj is None).

    Raises
    ------
    TypeError
        If YAML root is not a mapping.
    """
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    p = Path(obj)
    with p.open("rt", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"config root must be mapping, got {type(data)!r}")
    return dict(data)


def _package_defaults() -> dict[str, Any]:
    """
    Load default configuration from bamengine/config/defaults.yml.

    Returns
    -------
    dict
        Default configuration parameters.
    """
    txt = resources.files("bamengine").joinpath("config/defaults.yml").read_text()
    return yaml.safe_load(txt) or {}


def _validate_float1d(
    name: str,
    arr: float | Float1D,
    expected_len: int,
) -> float | Float1D:
    """
    Validate 1D float array or scalar for initialization.

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    arr : float or Float1D
        Scalar or 1D array to validate.
    expected_len : int
        Required array length (ignored for scalars).

    Returns
    -------
    float or Float1D
        Validated scalar or array.

    Raises
    ------
    ValueError
        If array has wrong shape or length.
    """
    if np.isscalar(arr):
        return float(arr)  # type: ignore[arg-type]
    arr = np.asarray(arr)
    if arr.ndim != 1 or arr.shape[0] != expected_len:
        raise ValueError(
            f"{name!s} must be length-{expected_len} 1-D array (got shape={arr.shape})"
        )
    return arr


# Simulation
# ---------------------------------------------------------------------
@dataclass(slots=True)
class Simulation:
    """
    Main simulation facade for BAM Engine.

    The Simulation class is the primary interface for running BAM (Bottom-Up Adaptive
    Macroeconomics) simulations. It manages the economy state, agent roles, event pipeline,
    and provides methods for stepping through periods.

    Attributes
    ----------
    ec : :class:`~bamengine.Economy`
        Global economy state (prices, wages, histories).
    config : :class:`~bamengine.config.Config`
        Configuration parameters for the simulation.
    rng : :class:`numpy.random.Generator`
        Random number generator for deterministic simulations.
    n_firms : int
        Number of firms in the economy.
    n_households : int
        Number of households in the economy.
    n_banks : int
        Number of banks in the economy.
    prod : :class:`~bamengine.roles.Producer`
        Producer role (firm production state).
    wrk : :class:`~bamengine.roles.Worker`
        Worker role (household employment state).
    emp : :class:`~bamengine.roles.Employer`
        Employer role (firm hiring state).
    bor : :class:`~bamengine.roles.Borrower`
        Borrower role (firm financial state).
    lend : :class:`~bamengine.roles.Lender`
        Lender role (bank state).
    con : :class:`~bamengine.roles.Consumer`
        Consumer role (household consumption state).
    pipeline : :class:`~bamengine.core.Pipeline`
        Event pipeline controlling simulation execution order.
    lb : :class:`~bamengine.relationships.LoanBook`
        Loan relationship between borrowers and lenders.
    n_periods : int
        Default run length for run() method.
    t : int
        Current period (starts at 0).

    Examples
    --------
    Basic usage with default configuration:

    >>> import bamengine as bam
    >>> sim = bam.Simulation.init(seed=42)
    >>> sim.run(n_periods=100)
    >>> unemployment = sim.ec.unemp_rate_history[-1]
    >>> print(f"Final unemployment: {unemployment:.2%}")
    Final unemployment: 0.04%

    Override configuration parameters:

    >>> sim = bam.Simulation.init(n_firms=200, n_households=1000, n_banks=15, seed=42)
    >>> sim.step()  # Single period
    >>> sim.t
    1

    Use custom configuration file:

    >>> sim = bam.Simulation.init(config="my_config.yml", seed=42)
    >>> sim.run(n_periods=50)

    Access roles and inspect state:

    >>> sim = bam.Simulation.init(seed=42)
    >>> sim.step()
    >>> prod = sim.get_role("Producer")
    >>> avg_price = prod.price.mean()
    >>> print(f"Average price: {avg_price:.2f}")
    Average price: 1.50

    Custom pipeline:

    >>> sim = bam.Simulation.init(pipeline_path="custom_pipeline.yml", seed=42)
    >>> sim.run(n_periods=100)

    Notes
    -----
    - All simulations are deterministic when seed is specified
    - State is mutated in-place during step() and run()
    - Agent roles share NumPy arrays for memory efficiency
    - Pipeline execution order can be customized via YAML files

    See Also
    --------
    init : Class method to create Simulation instances
    step : Execute one simulation period
    run : Execute multiple periods
    get_role : Access role instances
    get_event : Access event instances
    :class:`~bamengine.core.Pipeline` : Event pipeline configuration
    """

    # Economy instance
    ec: Economy

    # configuration
    config: Config
    rng: Rng

    # population sizes
    n_firms: int
    n_households: int
    n_banks: int

    # roles
    prod: Producer
    wrk: Worker
    emp: Employer
    bor: Borrower
    lend: Lender
    con: Consumer

    # event pipeline
    pipeline: Pipeline

    # relationships
    lb: LoanBook

    # periods
    n_periods: int  # run length
    t: int  # current period

    # Backward-compatible properties (delegate to config)
    @property
    def h_rho(self) -> float:
        """Max production-growth shock."""
        return self.config.h_rho

    @property
    def h_xi(self) -> float:
        """Max wage-growth shock."""
        return self.config.h_xi

    @property
    def h_phi(self) -> float:
        """Max bank operational costs shock."""
        return self.config.h_phi

    @property
    def h_eta(self) -> float:
        """Max price-growth shock."""
        return (
            self.config.h_eta
        )  # pragma: no cover - convenience accessor, tested via config

    @property
    def max_M(self) -> int:
        """Max job applications per unemployed worker."""
        return self.config.max_M

    @property
    def max_H(self) -> int:
        """Max loan applications per firm."""
        return self.config.max_H

    @property
    def max_Z(self) -> int:
        """Max firm visits per consumer."""
        return self.config.max_Z

    @property
    def theta(self) -> int:
        """Job contract length θ."""
        return self.config.theta

    @property
    def beta(self) -> float:
        """Propensity to consume exponent β."""
        return self.config.beta

    @property
    def delta(self) -> float:
        """Dividend payout ratio δ (DPR)."""
        return (
            self.config.delta
        )  # pragma: no cover - convenience accessor, tested via config

    @property
    def r_bar(self) -> float:
        """Baseline interest rate r̄."""
        return self.config.r_bar

    @property
    def v(self) -> float:
        """Bank capital requirement coefficient v."""
        return self.config.v

    @property
    def cap_factor(self) -> float | None:
        """Breakeven price cap factor."""
        return self.config.cap_factor

    # Constructor
    # ---------------------------------------------------------------------
    @classmethod
    def init(
        cls,
        config: str | Path | Mapping[str, Any] | None = None,
        **overrides: Any,  # anything here wins last
    ) -> Simulation:
        """
        Create a new Simulation instance with validated configuration.

        Configuration parameters are merged from three sources (later overrides earlier):
        1. Package defaults (bamengine/config/defaults.yml)
        2. User config (YAML file path, dict, or None)
        3. Keyword arguments (highest priority)

        Parameters
        ----------
        config : str, Path, Mapping, or None, optional
            Configuration source:
            - str/Path: Path to YAML configuration file
            - Mapping: Dictionary of configuration parameters
            - None: Use package defaults only
        **overrides : Any
            Configuration parameters to override (highest precedence).
            Common parameters:
            - n_firms : int (default: 100)
            - n_households : int (default: 500)
            - n_banks : int (default: 10)
            - seed : int or None (default: None)
            - pipeline_path : str or None (default: None)
            - logging : dict (default: {"default_level": "INFO"})
            See config/defaults.yml for all parameters.

        Returns
        -------
        Simulation
            Initialized simulation ready to run.

        Raises
        ------
        ValueError
            If configuration validation fails (invalid ranges, types, etc.).
        FileNotFoundError
            If config file path does not exist.

        Examples
        --------
        Use default configuration:

        >>> import bamengine as bam
        >>> sim = bam.Simulation.init(seed=42)
        >>> sim.n_firms, sim.n_households, sim.n_banks
        (100, 500, 10)

        Override population sizes:

        >>> sim = bam.Simulation.init(
        ...     n_firms=200, n_households=1000, n_banks=15, seed=42
        ... )
        >>> sim.n_firms
        200

        Load configuration from file:

        >>> sim = bam.Simulation.init(config="my_config.yml")  # doctest: +SKIP

        Combine file config with overrides:

        >>> sim = bam.Simulation.init(  # doctest: +SKIP
        ...     config="base_config.yml", seed=42, n_firms=150
        ... )

        Custom pipeline:

        >>> sim = bam.Simulation.init(
        ...     pipeline_path="custom_pipeline.yml", seed=42
        ... )  # doctest: +SKIP

        Configure logging:

        >>> log_config = {
        ...     "default_level": "DEBUG",
        ...     "events": {
        ...         "firms_adjust_price": "INFO",
        ...         "workers_send_one_round": "WARNING",
        ...     },
        ... }
        >>> sim = bam.Simulation.init(logging=log_config, seed=42)

        Notes
        -----
        - All configuration is validated before initialization
        - Invalid parameters raise ValueError with clear error messages
        - Vector parameters (price_init, net_worth_init, etc.) accept scalars
          (broadcast to all agents) or 1D arrays of appropriate length
        - Random seed ensures reproducible simulations
        - Default pipeline includes 39 events across 8 economic phases

        See Also
        --------
        :class:`bamengine.config.Config` : Configuration dataclass
        :class:`bamengine.config.ConfigValidator` : Centralized validation logic
        :class:`bamengine.core.Pipeline` : Event pipeline configuration
        """
        # 1 + 2 + 3 → one merged dict
        cfg_dict: dict[str, Any] = _package_defaults()
        cfg_dict.update(_read_yaml(config))
        cfg_dict.update(overrides)

        # Validate configuration (centralized validation)
        from bamengine.config import ConfigValidator

        ConfigValidator.validate_config(cfg_dict)

        # Validate pipeline path if specified
        pipeline_path = cfg_dict.get("pipeline_path")
        if pipeline_path is not None:
            ConfigValidator.validate_pipeline_path(pipeline_path)
            # Validate pipeline YAML with available parameters
            ConfigValidator.validate_pipeline_yaml(
                pipeline_path,
                params={
                    "max_M": cfg_dict.get("max_M", 4),
                    "max_H": cfg_dict.get("max_H", 2),
                    "max_Z": cfg_dict.get("max_Z", 2),
                },
            )

        # pull required scalars
        n_firms = int(cfg_dict.pop("n_firms"))
        n_households = int(cfg_dict.pop("n_households"))
        n_banks = int(cfg_dict.pop("n_banks"))

        # Random-seed handling
        seed_val = cfg_dict.pop("seed", None)
        rng: Rng = seed_val if isinstance(seed_val, Rng) else make_rng(seed_val)

        # vector params (validate size)
        cfg_dict["net_worth_init"] = _validate_float1d(
            "net_worth_init", cfg_dict.get("net_worth_init", 10.0), n_firms
        )
        cfg_dict["production_init"] = _validate_float1d(
            "production_init", cfg_dict.get("production_init", 1.0), n_firms
        )
        cfg_dict["price_init"] = _validate_float1d(
            "price_init", cfg_dict.get("price_init", 1.5), n_firms
        )
        cfg_dict["wage_offer_init"] = _validate_float1d(
            "wage_offer_init", cfg_dict.get("wage_offer_init", 1.0), n_firms
        )
        cfg_dict["savings_init"] = _validate_float1d(
            "savings_init", cfg_dict.get("savings_init", 1.0), n_households
        )
        cfg_dict["equity_base_init"] = _validate_float1d(
            "equity_base_init", cfg_dict.get("equity_base_init", 10_000.0), n_banks
        )

        # delegate to private constructor
        return cls._from_params(
            rng=rng,
            n_firms=n_firms,
            n_households=n_households,
            n_banks=n_banks,
            **cfg_dict,  # all remaining, size-checked parameters
        )

    @staticmethod
    def _configure_logging(log_config: dict[str, Any]) -> None:
        """
        Configure logging levels for bamengine loggers.

        Parameters
        ----------
        log_config : dict
            Logging configuration with keys:
            - default_level: str (e.g., 'INFO', 'DEBUG', 'TRACE')
            - events: dict[str, str] (per-event overrides)

        Notes
        -----
        Supports standard Python logging levels (DEBUG, INFO, WARNING, ERROR,
        CRITICAL) plus custom TRACE level (5) for fine-grained debugging.
        """

        # Map level names to numeric values
        # Include standard levels + custom TRACE
        level_map = {
            "TRACE": logging.TRACE,
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        # Set default level for bamengine logger
        default_level = log_config.get("default_level", "INFO")
        level_value = level_map.get(default_level, logging.INFO)
        logging.getLogger("bamengine").setLevel(level_value)

        # Set per-event log level overrides
        event_levels = log_config.get("events", {})
        for event_name, level in event_levels.items():
            logger_name = f"bamengine.events.{event_name}"
            level_value = level_map.get(level, logging.INFO)
            logging.getLogger(logger_name).setLevel(level_value)

    @classmethod
    def _from_params(cls, *, rng: Rng, **p: Any) -> Simulation:
        """
        Internal factory method to construct Simulation from validated config dict.

        This is an internal method called by `init()` after configuration validation.
        Users should use `Simulation.init()` instead.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator for deterministic simulations.
        **p : Any
            Validated configuration parameters (keys from Config dataclass).

        Returns
        -------
        Simulation
            Initialized simulation instance.

        Notes
        -----
        - Called internally by `init()` after config validation
        - Initializes all role arrays (Producer, Worker, etc.) from config
        - Creates default pipeline from YAML if `pipeline_path` not specified
        - Not intended for direct use by users

        See Also
        --------
        init : Public factory method for creating Simulation instances
        """

        # Vector initialization

        # finance
        net_worth = np.full(p["n_firms"], fill_value=p["net_worth_init"])
        total_funds = net_worth.copy()
        rnd_intensity = np.ones(p["n_firms"])
        gross_profit = np.zeros_like(net_worth)
        net_profit = np.zeros_like(net_worth)
        retained_profit = np.zeros_like(net_worth)

        # producer
        price = np.full(p["n_firms"], fill_value=p["price_init"])
        production = np.full(p["n_firms"], fill_value=p["production_init"])
        inventory = np.zeros_like(production)
        expected_demand = np.ones_like(production)
        desired_production = np.zeros_like(production)
        labor_productivity = np.ones(p["n_firms"])
        breakeven_price = price.copy()

        # employer
        current_labor = np.zeros(p["n_firms"], dtype=np.int64)
        desired_labor = np.zeros_like(current_labor)
        wage_offer = np.full(p["n_firms"], fill_value=p["wage_offer_init"])
        wage_bill = np.zeros_like(wage_offer)
        n_vacancies = np.zeros_like(desired_labor)
        recv_job_apps_head = np.full(p["n_firms"], -1, dtype=np.int64)
        recv_job_apps = np.full((p["n_firms"], p["n_households"]), -1, dtype=np.int64)

        # worker
        employer = np.full(p["n_households"], -1, dtype=np.int64)
        employer_prev = np.full_like(employer, -1)
        periods_left = np.zeros(p["n_households"], dtype=np.int64)
        contract_expired = np.zeros(p["n_households"], dtype=np.bool_)
        fired = np.zeros(p["n_households"], dtype=np.bool_)
        wage = np.zeros(p["n_households"])
        job_apps_head = np.full(p["n_households"], -1, dtype=np.int64)
        job_apps_targets = np.full((p["n_households"], p["max_M"]), -1, dtype=np.int64)

        # borrower
        credit_demand = np.zeros_like(net_worth)
        projected_fragility = np.zeros(p["n_firms"])
        loan_apps_head = np.full(p["n_firms"], -1, dtype=np.int64)
        loan_apps_targets = np.full((p["n_firms"], p["max_H"]), -1, dtype=np.int64)

        # lender
        equity_base = np.full(p["n_banks"], fill_value=p["equity_base_init"])
        # noinspection DuplicatedCode
        credit_supply = np.zeros_like(equity_base)
        interest_rate = np.zeros(p["n_banks"])
        recv_loan_apps_head = np.full(p["n_banks"], -1, dtype=np.int64)
        recv_loan_apps = np.full((p["n_banks"], p["n_firms"]), -1, dtype=np.int64)

        # consumer
        income = np.zeros_like(wage)
        savings = np.full_like(income, fill_value=p["savings_init"])
        income_to_spend = np.zeros_like(income)
        propensity = np.zeros(p["n_households"])
        largest_prod_prev = np.full(p["n_households"], -1, dtype=np.int64)
        shop_visits_head = np.full(p["n_households"], -1, dtype=np.int64)
        shop_visits_targets = np.full(
            (p["n_households"], p["max_Z"]), -1, dtype=np.int64
        )

        # economy level scalars & time-series
        avg_mkt_price = price.mean()
        avg_mkt_price_history = np.array([avg_mkt_price])
        unemp_rate_history = np.array([1.0])
        inflation_history = np.array([0.0])

        # Wrap into components
        # -----------------------------------------------------------------
        ec = Economy(
            avg_mkt_price=avg_mkt_price,
            min_wage=p["min_wage"],
            min_wage_rev_period=p["min_wage_rev_period"],
            avg_mkt_price_history=avg_mkt_price_history,
            unemp_rate_history=unemp_rate_history,
            inflation_history=inflation_history,
        )
        prod = Producer(
            price=price,
            production=production,
            inventory=inventory,
            expected_demand=expected_demand,
            desired_production=desired_production,
            labor_productivity=labor_productivity,
            breakeven_price=breakeven_price,
        )
        wrk = Worker(
            employer=employer,
            employer_prev=employer_prev,
            wage=wage,
            periods_left=periods_left,
            contract_expired=contract_expired,
            fired=fired,
            job_apps_head=job_apps_head,
            job_apps_targets=job_apps_targets,
        )
        emp = Employer(
            desired_labor=desired_labor,
            current_labor=current_labor,
            wage_offer=wage_offer,
            wage_bill=wage_bill,
            n_vacancies=n_vacancies,
            total_funds=total_funds,
            recv_job_apps_head=recv_job_apps_head,
            recv_job_apps=recv_job_apps,
        )
        bor = Borrower(
            net_worth=net_worth,
            total_funds=total_funds,
            wage_bill=wage_bill,
            credit_demand=credit_demand,
            rnd_intensity=rnd_intensity,
            gross_profit=gross_profit,
            net_profit=net_profit,
            retained_profit=retained_profit,
            projected_fragility=projected_fragility,
            loan_apps_head=loan_apps_head,
            loan_apps_targets=loan_apps_targets,
        )
        lend = Lender(
            equity_base=equity_base,
            credit_supply=credit_supply,
            interest_rate=interest_rate,
            recv_loan_apps_head=recv_loan_apps_head,
            recv_loan_apps=recv_loan_apps,
        )
        con = Consumer(
            income=income,
            savings=savings,
            income_to_spend=income_to_spend,
            propensity=propensity,
            largest_prod_prev=largest_prod_prev,
            shop_visits_head=shop_visits_head,
            shop_visits_targets=shop_visits_targets,
        )

        # Create config object
        cfg = Config(
            h_rho=p["h_rho"],
            h_xi=p["h_xi"],
            h_phi=p["h_phi"],
            h_eta=p["h_eta"],
            max_M=p["max_M"],
            max_H=p["max_H"],
            max_Z=p["max_Z"],
            theta=p["theta"],
            beta=p["beta"],
            delta=p["delta"],
            r_bar=p["r_bar"],
            v=p["v"],
            cap_factor=p.get("cap_factor"),
        )

        # Create event pipeline (default or custom)
        pipeline_path = p.get("pipeline_path")
        if pipeline_path is not None:
            from bamengine.core.pipeline import Pipeline

            pipeline = Pipeline.from_yaml(
                pipeline_path,
                max_M=p["max_M"],
                max_H=p["max_H"],
                max_Z=p["max_Z"],
            )
        else:
            pipeline = create_default_pipeline(
                max_M=p["max_M"], max_H=p["max_H"], max_Z=p["max_Z"]
            )

        # Configure logging (if specified)
        if "logging" in p:
            cls._configure_logging(p["logging"])

        return cls(
            ec=ec,
            prod=prod,
            wrk=wrk,
            emp=emp,
            bor=bor,
            lend=lend,
            lb=LoanBook(),
            con=con,
            config=cfg,
            pipeline=pipeline,
            n_firms=p["n_firms"],
            n_households=p["n_households"],
            n_banks=p["n_banks"],
            n_periods=p["n_periods"],
            t=0,
            rng=rng,
        )

    # public API
    # ---------------------------------------------------------------------
    def run(
        self,
        n_periods: int | None = None,
        collect: bool | dict[str, Any] = False,
    ) -> SimulationResults | None:
        """
        Run the simulation for multiple periods.

        Executes the event pipeline for a specified number of periods, advancing
        the economy state incrementally. If no argument is provided, uses the
        n_periods value from initialization.

        Parameters
        ----------
        n_periods : int, optional
            Number of periods to simulate. If None (default), uses the n_periods
            value passed at initialization via `Simulation.init()`.
        collect : bool or dict, default=False
            Whether to collect and return simulation results.

            - False: No collection, returns None (default)
            - True: Collect all roles (aggregated means) + economy metrics
            - dict: Custom collection config with keys:
                - 'roles': list of role names or None for all
                - 'variables': dict mapping role -> list of variables
                - 'aggregate': 'mean', 'median', 'sum', 'std', or None for full
                - 'economy': bool, include economy metrics (default True)

        Returns
        -------
        SimulationResults or None
            Results object if collect is truthy, None otherwise.

        Examples
        --------
        Run simulation for 100 periods using default configuration:

        >>> import bamengine as be
        >>> sim = be.Simulation.init(seed=42)
        >>> sim.run(n_periods=100)
        >>> unemployment = sim.ec.unemp_rate_history[-1]
        >>> print(f"Final unemployment rate: {unemployment:.2%}")
        Final unemployment rate: 4.32%

        Use n_periods from initialization:

        >>> sim = be.Simulation.init(n_periods=50, seed=42)
        >>> sim.run()  # Runs for 50 periods

        Collect results with default settings:

        >>> sim = be.Simulation.init(seed=42)
        >>> results = sim.run(n_periods=100, collect=True)
        >>> df = results.to_dataframe()
        >>> results.economy_metrics.plot()

        Custom data collection:

        >>> results = sim.run(
        ...     collect={
        ...         "roles": ["Producer"],
        ...         "variables": {"Producer": ["price", "inventory"]},
        ...         "aggregate": None,  # full per-agent data
        ...     }
        ... )

        Notes
        -----
        - Each period corresponds to one execution of the full event pipeline
        - State is mutated in-place regardless of collect parameter
        - Simulation halts early if economy is destroyed (all firms/banks bankrupt)
        - For step-by-step execution with custom logic, use `step()` instead

        See Also
        --------
        step : Execute a single simulation period
        init : Initialize simulation with configuration
        :class:`~bamengine.SimulationResults` : Container for collected simulation data
        """
        n = n_periods if n_periods is not None else self.n_periods

        # Set up collector if requested
        collector = None
        if collect:
            collector = self._create_collector(collect)

        # Run simulation
        start_time = time.time()
        for _ in range(int(n)):
            self.step()
            if collector:
                collector.capture(self)

        # Return results if collecting
        if collector:
            elapsed = time.time() - start_time
            metadata = {
                "n_periods": int(n),
                "n_firms": self.n_firms,
                "n_households": self.n_households,
                "n_banks": self.n_banks,
                "seed": self.config.seed if hasattr(self.config, "seed") else None,
                "runtime_seconds": elapsed,
            }
            config_dict = {
                "h_rho": self.config.h_rho,
                "h_xi": self.config.h_xi,
                "h_phi": self.config.h_phi,
                "h_eta": self.config.h_eta,
                "max_M": self.config.max_M,
                "max_H": self.config.max_H,
                "max_Z": self.config.max_Z,
                "theta": self.config.theta,
                "beta": self.config.beta,
                "delta": self.config.delta,
                "r_bar": self.config.r_bar,
                "v": self.config.v,
            }
            return collector.finalize(config_dict, metadata)

        return None

    def _create_collector(self, collect: bool | dict[str, Any]) -> _DataCollector:
        """
        Create data collector from collect parameter.

        Parameters
        ----------
        collect : bool or dict
            Collection configuration (True for defaults, dict for custom).

        Returns
        -------
        _DataCollector
            Configured data collector instance.
        """
        from bamengine.results import _DataCollector

        # Get all role names
        all_roles = ["Producer", "Worker", "Employer", "Borrower", "Lender", "Consumer"]

        if collect is True:
            # Default: all roles, aggregated means, include economy
            return _DataCollector(
                roles=all_roles,
                variables=None,
                include_economy=True,
                aggregate="mean",
            )
        else:
            # Custom configuration (collect must be a dict at this point)
            assert isinstance(collect, dict)
            return _DataCollector(
                roles=collect.get("roles") or all_roles,
                variables=collect.get("variables"),
                include_economy=collect.get("economy", True),
                aggregate=collect.get("aggregate", "mean"),
            )

    def step(self) -> None:
        """
        Execute one simulation period through the event pipeline.

        Advances the economy by exactly one period, executing all events in the
        pipeline in the order specified. This is the core execution method called
        by `run()`. Users can call this directly for fine-grained control between
        periods.

        Returns
        -------
        None
            State is mutated in-place. The period counter (`sim.t`) is incremented
            by 1. If the economy is destroyed (all firms/banks bankrupt), execution
            halts immediately.

        Examples
        --------
        Step through simulation manually with intermediate analysis:

        >>> import bamengine as be
        >>> sim = be.Simulation.init(seed=42)
        >>> for period in range(100):
        ...     sim.step()
        ...     if period % 10 == 0:
        ...         unemployment = sim.ec.unemp_rate_history[-1]
        ...         print(f"Period {period}: Unemployment = {unemployment:.2%}")
        Period 0: Unemployment = 8.40%
        Period 10: Unemployment = 5.20%
        Period 20: Unemployment = 4.80%
        ...

        Modify pipeline before stepping:

        >>> sim = be.Simulation.init(seed=42)
        >>> # Remove a specific event from the pipeline
        >>> sim.pipeline.remove("firms_pay_dividends")
        >>> sim.step()  # Executes modified pipeline

        Conditional execution based on economy state:

        >>> sim = be.Simulation.init(seed=42)
        >>> while sim.t < 100 and not sim.ec.destroyed:
        ...     sim.step()
        ...     avg_price = sim.ec.avg_mkt_price
        ...     if avg_price > 2.0:
        ...         print(f"High prices detected at period {sim.t}")
        ...         break

        Notes
        -----
        - Executes all events in `sim.pipeline` in order
        - Increments period counter (`sim.t`) before pipeline execution
        - Halts immediately if `sim.ec.destroyed` is True (economy collapse)
        - For bulk execution over many periods, use `run()` instead
        - Pipeline can be modified between calls to `step()`

        See Also
        --------
        run : Execute multiple periods
        pipeline : Event pipeline (can be modified before stepping)
        """
        if self.ec.destroyed:
            return

        self.t += 1

        # Execute pipeline
        self.pipeline.execute(self)

        if self.ec.destroyed:  # May be set during pipeline.execute()
            log.info("SIMULATION TERMINATED")  # type: ignore[unreachable]

    def get_role(self, name: str) -> Any:
        """
        Get role instance by name.

        Parameters
        ----------
        name : str
            Role name (case-insensitive): 'Producer', 'Worker', 'Employer',
            'Borrower', 'Lender', 'Consumer'.

        Returns
        -------
        :class:`~bamengine.core.Role`
            Role instance from simulation.

        Raises
        ------
        KeyError
            If role name not found.

        Examples
        --------
        >>> sim = Simulation.init()
        >>> prod = sim.get_role("Producer")
        >>> assert prod is sim.prod
        """
        from bamengine.core.registry import get_role as get_role_class
        from bamengine.core.registry import list_roles

        # Try to get the role class from registry (case-sensitive first)
        try:
            # First try exact match
            role_cls = get_role_class(name)
        except KeyError:
            # Try case-insensitive match
            name_title = name.title()  # Convert to TitleCase
            try:
                role_cls = get_role_class(name_title)
            except KeyError:
                # Provide helpful error message
                available = list_roles()
                raise KeyError(
                    f"Role '{name}' not found. Available roles: {available}"
                ) from None

        # Map role class to instance
        instance_map: dict[type, Any] = {
            Producer: self.prod,
            Worker: self.wrk,
            Employer: self.emp,
            Borrower: self.bor,
            Lender: self.lend,
            Consumer: self.con,
        }

        if role_cls not in instance_map:
            # This shouldn't happen unless new roles are added without updating Simulation
            raise KeyError(
                f"Role '{role_cls.__name__}' is registered but not available in simulation"
            )

        return instance_map[role_cls]

    def get_event(self, name: str) -> Any:
        """
        Get event instance from pipeline by name.

        Parameters
        ----------
        name : str
            Event name (e.g., 'firms_adjust_price').

        Returns
        -------
        :class:`~bamengine.core.Event`
            Event instance from current pipeline.

        Raises
        ------
        KeyError
            If event not found in pipeline.

        Examples
        --------
        >>> sim = Simulation.init()
        >>> pricing_event = sim.get_event("firms_adjust_price")
        """
        from bamengine.core.registry import get_event as get_event_class
        from bamengine.core.registry import list_events

        # First try to find by exact name match in pipeline
        for event in self.pipeline.events:
            if event.name == name:
                return event

        # If not found, check if it's a valid event in registry
        # This provides better error messages
        try:
            get_event_class(name)  # Just to check if it exists
            # Event exists in registry but not in current pipeline
            pipeline_events = [e.name for e in self.pipeline.events[:5]]  # TODO cover
            raise KeyError(
                f"Event '{name}' is registered but not in current pipeline. "
                f"Pipeline events (first 5): {pipeline_events}..."
            )
        except KeyError:
            # Event doesn't exist in registry at all
            available = list_events()
            raise KeyError(
                f"Event '{name}' not found. Available events in registry: {available[:10]}..."
            ) from None

    def get_relationship(self, name: str) -> Any:
        """
        Get relationship instance by name.

        Parameters
        ----------
        name : str
            Relationship name (case-insensitive): 'LoanBook'.

        Returns
        -------
        :class:`~bamengine.core.Relationship`
            Relationship instance from simulation.

        Raises
        ------
        KeyError
            If relationship name not found.

        Examples
        --------
        >>> sim = Simulation.init()
        >>> lb = sim.get_relationship("LoanBook")
        >>> assert lb is sim.lb
        """
        from bamengine.core.registry import get_relationship as get_relationship_class
        from bamengine.core.registry import list_relationships

        # Try to get the relationship class from registry (case-sensitive first)
        try:
            # First try exact match
            rel_cls = get_relationship_class(name)
        except KeyError:
            # Try case-insensitive match - check each registered name
            for registered_name in list_relationships():
                if registered_name.lower() == name.lower():
                    rel_cls = get_relationship_class(registered_name)
                    break
            else:
                # Provide helpful error message
                available = list_relationships()
                raise KeyError(
                    f"Relationship '{name}' not found. Available relationships: {available}"
                ) from None

        # Map relationship class to instance
        instance_map: dict[type, Any] = {
            LoanBook: self.lb,
        }

        if rel_cls not in instance_map:
            # This shouldn't happen unless new relationships are added without updating Simulation
            raise KeyError(
                f"Relationship '{rel_cls.__name__}' is registered but not available in simulation"
            )

        return instance_map[rel_cls]

    def get(self, name: str) -> Any:
        """
        Get role, event or relationship by name.

        Parameters
        ----------
        name : str
            Role, event or relationship name.

        Returns
        -------
        :class:`~bamengine.core.Role` | :class:`~bamengine.core.Event` | :class:`~bamengine.core.Relationship`
            Role, event or relationship instance from simulation.

        Raises
        ------
        ValueError
            If name not found in simulation.

        Note
        ----
        Searches roles first, then events, then relationships.

        Examples
        --------
        >>> sim = Simulation.init()
        >>> prod = sim.get("Producer")
        >>> event = sim.get("firms_adjust_price")
        """
        try:
            return self.get_role(name)
        except KeyError:
            pass

        try:
            return self.get_event(name)
        except KeyError:
            pass

        raise ValueError(f"'{name}' not found in simulation.")
