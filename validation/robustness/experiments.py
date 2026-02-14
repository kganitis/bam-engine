"""Experiment definitions for sensitivity analysis (Section 3.10.1).

Defines the five parameter groups from the book's univariate sensitivity
analysis. Each experiment varies one parameter (or parameter group) at a
time while holding all others at baseline defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Experiment:
    """A single sensitivity experiment definition.

    Parameters
    ----------
    name : str
        Short identifier for the experiment.
    description : str
        Human-readable description.
    param : str | None
        Config parameter name to vary. None for multi-param experiments.
    values : list
        Parameter values to test. For multi-param experiments, each
        value is a dict of config overrides.
    labels : list[str] | None
        Display labels for each value. Defaults to str(value).
    baseline_value : Any
        The default/baseline value (for highlighting in reports).
    """

    name: str
    description: str
    param: str | None
    values: list[Any]
    labels: list[str] | None = None
    baseline_value: Any = None

    def get_labels(self) -> list[str]:
        """Return display labels for each parameter value."""
        if self.labels is not None:
            return self.labels
        return [str(v) for v in self.values]

    def get_config(self, idx: int) -> dict[str, Any]:
        """Return config overrides for the i-th parameter value."""
        val = self.values[idx]
        if self.param is not None:
            return {self.param: val}
        # Multi-param: value is already a dict of overrides
        return dict(val)


# ─── Experiment Definitions ─────────────────────────────────────────────────

CREDIT_MARKET = Experiment(
    name="credit_market",
    description=(
        "Local credit markets: number of banks each firm can borrow from (H). "
        "Section 3.10.1(i)."
    ),
    param="max_H",
    values=[1, 2, 3, 4, 6],
    baseline_value=2,
)

GOODS_MARKET = Experiment(
    name="goods_market",
    description=(
        "Local consumption goods markets: number of firms consumers can visit "
        "before purchasing (Z). Section 3.10.1(ii)."
    ),
    param="max_Z",
    values=[2, 3, 4, 5, 6],
    baseline_value=2,
)

LABOR_APPLICATIONS = Experiment(
    name="labor_applications",
    description=(
        "Local labour markets: number of job applications per unemployed "
        "worker (M). Section 3.10.1(iii)."
    ),
    param="max_M",
    values=[2, 3, 4, 5, 6],
    baseline_value=4,
)

CONTRACT_LENGTH = Experiment(
    name="contract_length",
    description=(
        "Employment contracts duration (theta, in periods/quarters). "
        "Section 3.10.1(iv). Extreme values (1, 14) may cause collapse."
    ),
    param="theta",
    values=[1, 4, 6, 8, 10, 12, 14],
    baseline_value=8,
)

ECONOMY_SIZE = Experiment(
    name="economy_size",
    description=(
        "Size and structure of the economy. Section 3.10.1(v). "
        "Proportional scaling (2x, 5x, 10x) and class-specific doubling."
    ),
    param=None,  # Multi-param experiment
    values=[
        # Proportional scaling (baseline ratios preserved)
        {"n_firms": 100, "n_households": 500, "n_banks": 10},  # baseline
        {"n_firms": 200, "n_households": 1000, "n_banks": 20},  # 2x
        {"n_firms": 500, "n_households": 2500, "n_banks": 50},  # 5x
        {"n_firms": 1000, "n_households": 5000, "n_banks": 100},  # 10x
        # Composition changes: double one class at a time
        {"n_firms": 100, "n_households": 500, "n_banks": 20},  # 2x banks
        {"n_firms": 100, "n_households": 1000, "n_banks": 10},  # 2x households
        {"n_firms": 200, "n_households": 500, "n_banks": 10},  # 2x firms
    ],
    labels=[
        "baseline (100/500/10)",
        "2x all (200/1000/20)",
        "5x all (500/2500/50)",
        "10x all (1000/5000/100)",
        "2x banks (100/500/20)",
        "2x households (100/1000/10)",
        "2x firms (200/500/10)",
    ],
    baseline_value={"n_firms": 100, "n_households": 500, "n_banks": 10},
)

# Registry of all experiments
EXPERIMENTS: dict[str, Experiment] = {
    "credit_market": CREDIT_MARKET,
    "goods_market": GOODS_MARKET,
    "labor_applications": LABOR_APPLICATIONS,
    "contract_length": CONTRACT_LENGTH,
    "economy_size": ECONOMY_SIZE,
}

ALL_EXPERIMENT_NAMES: list[str] = list(EXPERIMENTS.keys())
