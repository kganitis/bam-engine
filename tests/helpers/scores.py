"""Reusable ValidationScore builders for calibration tests."""

from __future__ import annotations

from validation.types import MetricGroup, MetricResult, ValidationScore


def make_score(total: float, group: str = "TIME_SERIES") -> ValidationScore:
    """Create a minimal ValidationScore with a single metric.

    Parameters
    ----------
    total : float
        Score value used for both the metric score and total_score.
    group : str
        MetricGroup name (default ``"TIME_SERIES"``).
    """
    mr = MetricResult(
        name="test_metric",
        status="PASS",
        actual=total,
        target_desc="test",
        score=total,
        weight=1.0,
        group=MetricGroup[group],
    )
    return ValidationScore(
        metric_results=[mr],
        total_score=total,
        n_pass=1,
        n_warn=0,
        n_fail=0,
    )
