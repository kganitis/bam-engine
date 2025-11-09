# src/bamengine/simulation.py
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Mapping

# noinspection PyPackageRequirements
import numpy as np

# noinspection PyPackageRequirements
import yaml

# noinspection PyPackageRequirements
from numpy.random import Generator, default_rng

import bamengine.events  # noqa: F401 - needed to register events
from bamengine._logging_ext import getLogger
from bamengine.config import Config
from bamengine.core.default_pipeline import create_default_pipeline
from bamengine.core.pipeline import Pipeline
from bamengine.roles import (
    Borrower,
    Consumer,
    Economy,
    Employer,
    Lender,
    LoanBook,
    Producer,
    Worker,
)
from bamengine.systems.bankruptcy import (
    firms_update_net_worth,
    mark_bankrupt_banks,
    mark_bankrupt_firms,
    spawn_replacement_banks,
    spawn_replacement_firms,
)
from bamengine.systems.credit_market import (
    banks_decide_credit_supply,
    banks_decide_interest_rate,
    banks_provide_loans,
    firms_calc_credit_metrics,
    firms_decide_credit_demand,
    firms_fire_workers,
    firms_prepare_loan_applications,
    firms_send_one_loan_app,
)
from bamengine.systems.goods_market import (
    consumers_calc_propensity,
    consumers_decide_firms_to_visit,
    consumers_decide_income_to_spend,
    consumers_finalize_purchases,
    consumers_shop_one_round,
)
from bamengine.systems.labor_market import (
    adjust_minimum_wage,
    calc_annual_inflation_rate,
    firms_calc_wage_bill,
    firms_decide_wage_offer,
    firms_hire_workers,
    workers_decide_firms_to_apply,
    workers_send_one_round,
)
from bamengine.systems.planning import (
    firms_adjust_price,
    firms_calc_breakeven_price,
    firms_decide_desired_labor,
    firms_decide_desired_production,
    firms_decide_vacancies,
)
from bamengine.systems.production import (
    calc_unemployment_rate,
    firms_pay_wages,
    firms_run_production,
    update_avg_mkt_price,
    workers_receive_wage,
    workers_update_contracts,
)
from bamengine.systems.revenue import (
    firms_collect_revenue,
    firms_pay_dividends,
    firms_validate_debt_commitments,
)
from bamengine.typing import Float1D

__all__ = ["Simulation"]

log = getLogger(__name__)


# helpers
# ---------------------------------------------------------------------------
def _read_yaml(obj: str | Path | Mapping[str, Any] | None) -> Dict[str, Any]:
    """Return a plain dict – {} if *obj* is None."""
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


def _package_defaults() -> Dict[str, Any]:
    """Load bamengine/defaults.yml"""
    txt = resources.files("bamengine").joinpath("defaults.yml").read_text()
    return yaml.safe_load(txt) or {}


def _validate_float1d(
    name: str,
    arr: float | Float1D,
    expected_len: int,
) -> float | Float1D:
    """Ensure Float1D has the right length; scalars are accepted verbatim."""
    if np.isscalar(arr):
        return float(arr)  # type: ignore[arg-type]
    arr = np.asarray(arr)
    if arr.ndim != 1 or arr.shape[0] != expected_len:
        raise ValueError(
            f"{name!s} must be length-{expected_len} 1-D array "
            f"(got shape={arr.shape})"
        )
    return arr


# Simulation
# ---------------------------------------------------------------------
@dataclass(slots=True)
class Simulation:
    """
    Facade that drives one Economy instance through *n* consecutive periods.

    One call to `run` → *n* calls to `step`.
    """

    # core state
    rng: Generator
    ec: Economy
    prod: Producer
    wrk: Worker
    emp: Employer
    bor: Borrower
    lend: Lender
    con: Consumer
    lb: LoanBook

    # configuration
    config: Config

    # event pipeline
    pipeline: Pipeline

    # population sizes
    n_firms: int
    n_households: int
    n_banks: int

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
        return self.config.h_eta

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
        return self.config.delta

    # Constructor
    # ---------------------------------------------------------------------
    @classmethod
    def init(
        cls,
        config: str | Path | Mapping[str, Any] | None = None,
        **overrides: Any,  # anything here wins last
    ) -> "Simulation":
        """
        Build a Simulation.

        Order of precedence (later overrides earlier):

            1. package defaults  (bamengine/defaults.yml)
            2. *config*  (Path / str / Mapping / None)
            3. explicit keyword arguments (**overrides)
        """
        # 1 + 2 + 3 → one merged dict
        cfg_dict: Dict[str, Any] = _package_defaults()
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
        rng: Generator = (
            seed_val if isinstance(seed_val, Generator) else default_rng(seed_val)
        )

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
    def _configure_logging(log_config: Dict[str, Any]) -> None:
        """
        Configure logging levels for bamengine loggers.

        Parameters
        ----------
        log_config : dict
            Logging configuration with keys:
            - default_level: str (e.g., 'INFO', 'DEBUG')
            - events: dict[str, str] (per-event overrides)
        """
        import logging

        # Set default level for bamengine logger
        default_level = log_config.get("default_level", "INFO")
        logging.getLogger("bamengine").setLevel(getattr(logging, default_level))

        # Set per-event log level overrides
        event_levels = log_config.get("events", {})
        for event_name, level in event_levels.items():
            logger_name = f"bamengine.events.{event_name}"
            logging.getLogger(logger_name).setLevel(getattr(logging, level))

    @classmethod
    def _from_params(cls, *, rng: Generator, **p: Any) -> "Simulation":  # noqa: C901

        # Vector initilization

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
        employed = np.zeros(p["n_households"], dtype=np.bool_)
        employer = np.full(p["n_households"], -1, dtype=np.int64)
        employer_prev = np.full_like(employer, -1)
        periods_left = np.zeros(p["n_households"], dtype=np.int64)
        contract_expired = np.zeros_like(employed)
        # noinspection DuplicatedCode
        fired = np.zeros_like(employed)
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
            # TODO move theta, beta and delta in here
            avg_mkt_price=avg_mkt_price,
            min_wage=p["min_wage"],
            min_wage_rev_period=p["min_wage_rev_period"],
            r_bar=p["r_bar"],
            v=p["v"],
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
            employed=employed,
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
    def run(self, n_periods: int | None = None) -> None:
        """
        Advance the simulation *n_periods* steps
        (defaults to the ``n_periods`` passed at construction).

        Returns
        -------
        None   (state is mutated in-place)
        """
        n = n_periods if n_periods is not None else self.n_periods
        for _ in range(int(n)):
            self.step()

    def step(self) -> None:
        """
        Advance the economy by exactly one period using the event pipeline.

        This method executes all events in the pipeline in dependency-resolved
        order. The pipeline can be customized by users before calling step().

        See Also
        --------
        _step_legacy : Legacy implementation for backward compatibility
        pipeline : Event pipeline attribute that can be modified
        """
        if self.ec.destroyed:
            return

        self.t += 1

        # Execute pipeline
        self.pipeline.execute(self)

        if self.ec.destroyed:
            log.info("SIMULATION TERMINATED")

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
        Role
            Role instance from simulation.

        Raises
        ------
        ValueError
            If role name not found.

        Examples
        --------
        >>> sim = Simulation.init()
        >>> prod = sim.get_role("Producer")
        >>> assert prod is sim.prod
        """
        role_map = {
            "producer": self.prod,
            "worker": self.wrk,
            "employer": self.emp,
            "borrower": self.bor,
            "lender": self.lend,
            "consumer": self.con,
        }

        name_lower = name.lower()
        if name_lower not in role_map:
            available = list(role_map.keys())
            raise ValueError(f"Role '{name}' not found. Available roles: {available}")

        return role_map[name_lower]

    def get_event(self, name: str) -> Any:
        """
        Get event instance from pipeline by name.

        Parameters
        ----------
        name : str
            Event name (e.g., 'firms_adjust_price').

        Returns
        -------
        Event
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
        for event in self.pipeline.events:
            if event.name == name:
                return event

        available = [e.name for e in self.pipeline.events[:5]]
        raise KeyError(
            f"Event '{name}' not found in pipeline. "
            f"Available (first 5): {available}..."
        )

    def get(self, name: str) -> Any:
        """
        Get role or event by name (searches roles first, then events).

        Parameters
        ----------
        name : str
            Role or event name.

        Returns
        -------
        Role | Event
            Role or event instance from simulation.

        Raises
        ------
        ValueError
            If name not found in roles or events.

        Examples
        --------
        >>> sim = Simulation.init()
        >>> prod = sim.get("Producer")
        >>> event = sim.get("firms_adjust_price")
        """
        try:
            return self.get_role(name)
        except ValueError:
            pass

        try:
            return self.get_event(name)
        except KeyError:
            pass

        raise ValueError(
            f"'{name}' not found in roles or events. "
            "Use get_role() or get_event() for more specific error messages."
        )

    def _step_legacy(self) -> None:
        """
        Legacy step implementation (pre-pipeline).

        This method is kept for backward compatibility and golden master testing.
        It will be removed in a future version once pipeline implementation
        is fully validated.
        """

        # TODO
        #  - Wrap for-loops into their own systems
        #  - Break systems into simpler systems

        if self.ec.destroyed:
            return

        self.t += 1

        # *************** event 1 – planning ***************

        firms_decide_desired_production(
            self.prod, p_avg=self.ec.avg_mkt_price, h_rho=self.h_rho, rng=self.rng
        )
        firms_calc_breakeven_price(self.prod, self.emp, self.lb)
        firms_adjust_price(
            self.prod,
            p_avg=self.ec.avg_mkt_price,
            h_eta=self.h_eta,
            rng=self.rng,
        )
        update_avg_mkt_price(self.ec, self.prod)
        calc_annual_inflation_rate(self.ec)
        firms_decide_desired_labor(self.prod, self.emp)
        firms_decide_vacancies(self.emp)

        # *************** event 2 – labor-market ***************

        adjust_minimum_wage(self.ec)
        firms_decide_wage_offer(
            self.emp, w_min=self.ec.min_wage, h_xi=self.h_xi, rng=self.rng
        )
        workers_decide_firms_to_apply(
            self.wrk, self.emp, max_M=self.max_M, rng=self.rng
        )
        for _ in range(self.max_M):
            workers_send_one_round(self.wrk, self.emp, rng=self.rng)
            firms_hire_workers(self.wrk, self.emp, theta=self.theta, rng=self.rng)
        firms_calc_wage_bill(self.emp, self.wrk)

        # *************** event 3 – credit-market ***************

        banks_decide_credit_supply(self.lend, v=self.ec.v)
        banks_decide_interest_rate(
            self.lend, r_bar=self.ec.r_bar, h_phi=self.h_phi, rng=self.rng
        )
        firms_decide_credit_demand(self.bor)
        firms_calc_credit_metrics(self.bor)
        firms_prepare_loan_applications(
            self.bor, self.lend, max_H=self.max_H, rng=self.rng
        )
        for _ in range(self.max_H):
            firms_send_one_loan_app(self.bor, self.lend, rng=self.rng)
            banks_provide_loans(self.bor, self.lb, self.lend, r_bar=0.02, h_phi=0.10)
        firms_fire_workers(self.emp, self.wrk, rng=self.rng)

        # *************** event 4 – production ***************

        firms_pay_wages(self.emp)
        workers_receive_wage(self.con, self.wrk)
        firms_run_production(self.prod, self.emp)
        workers_update_contracts(self.wrk, self.emp)

        # *************** event 5 – goods-market ***************

        _avg_sav = float(self.con.savings.mean())
        consumers_calc_propensity(self.con, avg_sav=_avg_sav, beta=self.beta)
        consumers_decide_income_to_spend(self.con)
        consumers_decide_firms_to_visit(
            self.con, self.prod, max_Z=self.max_Z, rng=self.rng
        )
        for _ in range(self.max_Z):
            consumers_shop_one_round(self.con, self.prod, rng=self.rng)
        consumers_finalize_purchases(self.con)

        # *************** event 6 – revenue ***************

        firms_collect_revenue(self.prod, self.bor)
        firms_validate_debt_commitments(self.bor, self.lend, self.lb)
        firms_pay_dividends(self.bor, delta=self.delta)

        # *************** event 7 – bankruptcy ***************

        firms_update_net_worth(self.bor)
        mark_bankrupt_firms(self.ec, self.emp, self.bor, self.prod, self.wrk, self.lb)
        mark_bankrupt_banks(self.ec, self.lend, self.lb)

        # *************** event 8 – entry ***************

        spawn_replacement_firms(self.ec, self.prod, self.emp, self.bor, rng=self.rng)
        spawn_replacement_banks(self.ec, self.lend, rng=self.rng)

        # *************** end of period ***************

        calc_unemployment_rate(self.ec, self.wrk)

        if self.ec.destroyed:
            log.info("SIMULATION TERMINATED")
