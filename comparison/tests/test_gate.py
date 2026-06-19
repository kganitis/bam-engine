import math
import warnings

from comparison.equivalence.gate import bamengine_stats, evaluate_gate, tolerances
from comparison.equivalence.metrics import METRIC_KEYS


def _rows(value):  # one metric dict per seed, all metrics = value
    return [{m: value for m in METRIC_KEYS} for _ in range(20)]


def test_tolerance_floored_by_validation_band():
    be = bamengine_stats(_rows(0.07))  # std == 0
    tol = tolerances(be)
    assert tol["mean_unemployment"] == 0.025  # floored to validation half-width


def test_matching_framework_passes():
    by = {"bamengine": _rows(0.07), "mesa": _rows(0.07)}
    report = evaluate_gate(by)
    assert report["frameworks"]["mesa"]["passed"] is True


def test_deviating_framework_fails():
    by = {"bamengine": _rows(0.07), "mesa": _rows(0.5)}
    report = evaluate_gate(by)
    assert report["frameworks"]["mesa"]["passed"] is False


def test_netlogo_non_blocking():
    by = {"bamengine": _rows(0.07), "netlogo": _rows(0.5)}
    report = evaluate_gate(by)
    assert report["frameworks"]["netlogo"]["blocking"] is False


def test_all_nan_metric_no_warning():
    """All-NaN column for a metric must return nan mean/std with no RuntimeWarning."""
    nan_rows = [{m: float("nan") for m in METRIC_KEYS} for _ in range(5)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        stats = bamengine_stats(nan_rows)
    for m in METRIC_KEYS:
        assert math.isnan(stats[m]["mean"]), f"{m}: expected nan mean"
        assert math.isnan(stats[m]["std"]), f"{m}: expected nan std"
