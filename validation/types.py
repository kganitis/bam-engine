"""Core types and dataclasses for the validation package.

This module defines all the type definitions used across the validation
and calibration packages.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import bamengine as bam
    from bamengine import SimulationResults

# =============================================================================
# Constants
# =============================================================================

# Default seeds for stability testing across validation/calibration
DEFAULT_STABILITY_SEEDS: list[int] = list(range(20))

# Type alias for validation status
Status = Literal["PASS", "WARN", "FAIL"]


# =============================================================================
# Enums for MetricSpec configuration
# =============================================================================


class CheckType(Enum):
    """Type of check to perform on a metric."""

    MEAN_TOLERANCE = auto()  # value within target ± tolerance
    RANGE = auto()  # value within [min, max]
    PCT_WITHIN = auto()  # percentage meeting target threshold
    OUTLIER = auto()  # outlier percentage penalty
    BOOLEAN = auto()  # simple true/false check (e.g., > 0)


class MetricGroup(Enum):
    """Grouping for metrics in reports."""

    TIME_SERIES = auto()
    CURVES = auto()
    DISTRIBUTION = auto()
    GROWTH = auto()
    FINANCIAL = auto()
    GROWTH_RATE_DIST = auto()


class MetricFormat(Enum):
    """Display format for metric values."""

    DEFAULT = auto()  # standard decimal format
    PERCENT = auto()  # multiply by 100 and add %
    TREND = auto()  # high precision for trend coefficients
    INTEGER = auto()  # round to integer


# =============================================================================
# MetricSpec - The core abstraction
# =============================================================================


@dataclass
class MetricSpec:
    """Unified specification for a validation metric.

    This dataclass captures everything needed to validate a single metric.
    Target values are looked up from YAML using standardized keys:
    - MEAN_TOLERANCE: expects 'target' and 'tolerance' keys
    - RANGE: expects 'min' and 'max' keys
    - PCT_WITHIN: expects 'target' and 'min' keys
    - OUTLIER: expects 'max_outlier' key (and optional 'penalty_weight')
    - BOOLEAN: uses threshold defined here
    """

    name: str  # e.g., "unemployment_rate_mean"
    field: str  # attribute on Metrics dataclass
    check_type: CheckType
    target_path: str  # dot-separated path in YAML: "time_series.unemployment.mean"
    weight: float = 1.0
    group: MetricGroup = MetricGroup.TIME_SERIES
    format: MetricFormat = MetricFormat.DEFAULT

    # For BOOLEAN checks only
    threshold: float = 0.0  # value must be > threshold
    invert: bool = False  # if True, value must be < threshold

    # Custom target description (if None, auto-generated)
    target_desc: str | None = None


# =============================================================================
# Validation Result Types
# =============================================================================


@dataclass
class MetricResult:
    """Result of validating a single metric."""

    name: str
    status: Status
    actual: float
    target_desc: str
    score: float  # 0-1 score (1 = perfect match)
    weight: float = 1.0  # Weight for total score calculation
    message: str = ""
    group: MetricGroup = MetricGroup.TIME_SERIES
    format: MetricFormat = MetricFormat.DEFAULT


@dataclass
class ValidationScore:
    """Overall validation result with scoring for comparison."""

    metric_results: list[MetricResult]
    total_score: float  # Weighted average of all metric scores
    n_pass: int
    n_warn: int
    n_fail: int
    config: dict[str, Any] = field(default_factory=dict)  # Config used for this run

    @property
    def passed(self) -> bool:
        """True if no metrics failed validation."""
        return self.n_fail == 0

    def __str__(self) -> str:
        return (
            f"ValidationScore(total={self.total_score:.3f}, "
            f"pass={self.n_pass}, warn={self.n_warn}, fail={self.n_fail})"
        )


@dataclass
class MetricStats:
    """Statistics for a single metric across multiple seeds."""

    name: str
    mean_value: float
    std_value: float
    mean_score: float
    std_score: float
    pass_rate: float  # Fraction of seeds where this metric passed (not FAIL)
    format: MetricFormat = MetricFormat.DEFAULT


@dataclass
class StabilityResult:
    """Result of multi-seed stability testing."""

    seed_results: list[ValidationScore]  # Individual seed results

    # Aggregate score metrics
    mean_score: float  # Mean total score across seeds
    std_score: float  # Standard deviation of scores
    min_score: float  # Worst seed
    max_score: float  # Best seed

    pass_rate: float  # Fraction of seeds that passed (no FAILs)
    n_seeds: int  # Number of seeds tested

    # Per-metric stability
    metric_stats: dict[str, MetricStats]  # Stats for each metric

    @property
    def is_stable(self) -> bool:
        """True if pass_rate >= 80% and std_score <= 0.1."""
        return self.pass_rate >= 0.8 and self.std_score <= 0.1

    def __str__(self) -> str:
        return (
            f"StabilityResult(mean={self.mean_score:.3f}±{self.std_score:.3f}, "
            f"pass_rate={self.pass_rate:.0%}, seeds={self.n_seeds})"
        )


# =============================================================================
# Scenario Configuration
# =============================================================================


@dataclass
class Scenario:
    """Configuration for a validation scenario.

    This dataclass bundles everything needed to run validation for a specific
    scenario (baseline or growth_plus).
    """

    name: str
    metric_specs: list[MetricSpec]
    collect_config: dict[str, Any]
    targets_path: Path  # absolute path to scenario's targets.yaml
    default_config: dict[str, Any]  # default simulation config
    compute_metrics: Callable[[bam.Simulation, SimulationResults, int], Any]
    setup_hook: Callable[[bam.Simulation | None], None] | None = None
    """Optional hook called twice: first with ``None`` (to trigger imports/registration),
    then with the ``Simulation`` instance (to attach roles/extensions)."""
    title: str = ""  # report title, e.g. "BASELINE SCENARIO VALIDATION"
    stability_title: str = ""  # stability report title, e.g. "SEED STABILITY TEST"
