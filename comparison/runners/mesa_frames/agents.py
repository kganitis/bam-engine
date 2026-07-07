"""Agent classes for the mesa-frames (Polars backend) implementation of the baseline BAM model.

Each class is an AgentSetPolars subclass whose columns mirror the attributes of
the corresponding Mesa agent class in comparison/runners/mesa/agents.py one-to-one.
Pointer/object attributes from the Mesa port (employer, loan_apps, etc.) are omitted
here; they are handled via separate relationship tables on the model.
"""

from __future__ import annotations

import mesa_frames as mf
import polars as pl


def _alloc_ids(model: mf.ModelDF, n: int) -> list[int]:
    """Allocate n sequential unique_ids from model.current_id and advance the counter."""
    start = model.current_id
    model.current_id += n
    return list(range(start, start + n))


class Firms(mf.AgentSetPolars):
    """Firm agents (Producer + Employer + Borrower roles).

    Columns mirror Firm.__init__ in comparison/runners/mesa/agents.py.
    Object-valued attributes (loan_apps, loans, employees) are excluded and
    handled via relationship tables on the model.
    """

    def __init__(
        self,
        model: mf.ModelDF,
        n: int,
        params: dict,
        rng,  # numpy Generator - not used during construction (no randomness)
    ) -> None:
        super().__init__(model)

        # Derived initial values (mirrors BamModel.__init__ in mesa/model.py)
        lp: float = float(params["labor_productivity"])
        price_init: float = float(params["price_init"])
        n_households: int = model.n_households
        production_init: float = n_households * lp / n
        wage_offer_init: float = price_init / 3.0
        net_worth_init: float = (
            production_init * price_init * float(params["net_worth_ratio"])
        )
        max_H: int = int(params["max_H"])

        ids = _alloc_ids(model, n)

        data = {
            # Mesa: unique_id is assigned by super().__init__(model) sequentially
            "unique_id": ids,
            # Producer columns
            "production": [0.0] * n,
            "production_prev": [production_init] * n,
            "inventory": [0.0] * n,
            "expected_demand": [1.0] * n,
            "desired_production": [0.0] * n,
            "labor_productivity": [lp] * n,
            "breakeven_price": [price_init] * n,
            "price": [price_init] * n,
            # Employer columns
            "desired_labor": [0] * n,
            "current_labor": [0] * n,
            "wage_offer": [wage_offer_init] * n,
            "wage_bill": [0.0] * n,
            "n_vacancies": [0] * n,
            "total_funds": [net_worth_init] * n,
            # Borrower columns
            "net_worth": [net_worth_init] * n,
            "credit_demand": [0.0] * n,
            "projected_fragility": [0.0] * n,
            "gross_profit": [0.0] * n,
            "net_profit": [0.0] * n,
            "retained_profit": [0.0] * n,
            # Loan-application queue (-1-padded bank ids + head index), mirroring
            # the labor market's job_app_* columns on Households.  Populated by
            # event 17 (firms_prepare_loan_applications) before each credit market.
            "loan_app_head": [0] * n,
        }
        schema = {
            "unique_id": pl.Int64,
            "production": pl.Float64,
            "production_prev": pl.Float64,
            "inventory": pl.Float64,
            "expected_demand": pl.Float64,
            "desired_production": pl.Float64,
            "labor_productivity": pl.Float64,
            "breakeven_price": pl.Float64,
            "price": pl.Float64,
            "desired_labor": pl.Int64,
            "current_labor": pl.Int64,
            "wage_offer": pl.Float64,
            "wage_bill": pl.Float64,
            "n_vacancies": pl.Int64,
            "total_funds": pl.Float64,
            "net_worth": pl.Float64,
            "credit_demand": pl.Float64,
            "projected_fragility": pl.Float64,
            "gross_profit": pl.Float64,
            "net_profit": pl.Float64,
            "retained_profit": pl.Float64,
            "loan_app_head": pl.Int64,
        }
        for k in range(max_H):
            data[f"loan_app_{k}"] = [-1] * n
            schema[f"loan_app_{k}"] = pl.Int64

        df = pl.DataFrame(data, schema=schema)
        self.add(df)

    def step(self) -> None:
        """Placeholder: event dispatch is handled by the model step."""
        pass


class Households(mf.AgentSetPolars):
    """Household agents (Worker + Consumer + Shareholder roles).

    Columns mirror Household.__init__ in comparison/runners/mesa/agents.py.

    The worker -> firm link is represented columnar-style, mirroring
    bamengine's ``wrk.employer`` (an Int64 array):

    - ``employer``: firm ``unique_id`` of the current employer, or ``-1`` when
      unemployed.  "Employed" is ALWAYS derived from ``employer >= 0``; there is
      no stored employed bool.
    - ``employer_prev``: firm ``unique_id`` of the previous employer (loyalty),
      or ``-1`` when none.

    The job-application queue (Mesa port's per-worker ``job_apps`` list) is
    stored as a fixed-width ``-1``-padded matrix of ``max_M`` columns
    ``job_app_0 .. job_app_{max_M-1}`` plus a ``job_app_head`` index, matching
    the Mesa port's application-queue pattern in columnar form.

    The goods-market visit queue (Mesa port's per-consumer ``shop_visits`` list)
    is stored as a fixed-width ``-1``-padded matrix of ``max_Z`` columns
    ``shop_visit_0 .. shop_visit_{max_Z-1}`` (firm unique_ids); the sequential
    shopping loop stops at the first ``-1`` sentinel (no count column).
    Loyalty is tracked via ``largest_prod_prev``: the firm unique_id of the
    largest producer visited last period (``-1`` = none), mirroring the Mesa
    port's ``Household.largest_prod_prev`` attribute.
    """

    def __init__(
        self,
        model: mf.ModelDF,
        n: int,
        params: dict,
        rng,  # numpy Generator - not used during construction
    ) -> None:
        super().__init__(model)

        savings_init: float = float(params["savings_init"])
        max_M: int = int(params["max_M"])
        max_Z: int = int(params["max_Z"])
        ids = _alloc_ids(model, n)

        data = {
            "unique_id": ids,
            # Worker columns
            "employer": [-1] * n,  # -1 = unemployed; employed := employer >= 0
            "employer_prev": [-1] * n,  # -1 = no previous employer
            "wage": [0.0] * n,
            "periods_left": [0] * n,
            "contract_expired": [False] * n,
            "fired": [False] * n,
            # Consumer columns
            "income": [0.0] * n,
            "savings": [savings_init] * n,
            "income_to_spend": [0.0] * n,
            "propensity": [0.0] * n,
            # Loyalty: firm unique_id of the largest producer visited last period.
            # -1 = none (mirrors Mesa port's Household.largest_prod_prev = None).
            "largest_prod_prev": [-1] * n,
            # Shareholder columns
            "dividends": [0.0] * n,
            # Job-application queue (-1-padded targets + head index).
            "job_app_head": [0] * n,
        }
        schema = {
            "unique_id": pl.Int64,
            "employer": pl.Int64,
            "employer_prev": pl.Int64,
            "wage": pl.Float64,
            "periods_left": pl.Int64,
            "contract_expired": pl.Boolean,
            "fired": pl.Boolean,
            "income": pl.Float64,
            "savings": pl.Float64,
            "income_to_spend": pl.Float64,
            "propensity": pl.Float64,
            "largest_prod_prev": pl.Int64,
            "dividends": pl.Float64,
            "job_app_head": pl.Int64,
        }
        for k in range(max_M):
            data[f"job_app_{k}"] = [-1] * n
            schema[f"job_app_{k}"] = pl.Int64
        # Shop-visit queue (-1-padded firm ids), populated by event 27.
        for k in range(max_Z):
            data[f"shop_visit_{k}"] = [-1] * n
            schema[f"shop_visit_{k}"] = pl.Int64

        df = pl.DataFrame(data, schema=schema)
        self.add(df)

    def step(self) -> None:
        """Placeholder: event dispatch is handled by the model step."""
        pass


class Banks(mf.AgentSetPolars):
    """Bank agents (Lender role).

    Columns mirror Bank.__init__ in comparison/runners/mesa/agents.py.
    """

    def __init__(
        self,
        model: mf.ModelDF,
        n: int,
        params: dict,
        rng,  # numpy Generator - not used during construction
    ) -> None:
        super().__init__(model)

        equity_base_init: float = float(params["equity_base_init"])
        ids = _alloc_ids(model, n)

        df = pl.DataFrame(
            {
                "unique_id": ids,
                "equity_base": [equity_base_init] * n,
                "credit_supply": [0.0] * n,
                "interest_rate": [0.0] * n,
                "opex_shock": [0.0] * n,
            },
            schema={
                "unique_id": pl.Int64,
                "equity_base": pl.Float64,
                "credit_supply": pl.Float64,
                "interest_rate": pl.Float64,
                "opex_shock": pl.Float64,
            },
        )
        self.add(df)

    def step(self) -> None:
        """Placeholder: event dispatch is handled by the model step."""
        pass
