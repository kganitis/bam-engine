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


def test_nan_reference_metric_is_indeterminate():
    """When the reference (bamengine) mean for a metric is NaN, that metric must be
    marked indeterminate and must NOT contribute to a FAIL verdict for any framework,
    including bamengine itself.
    """
    # Pick the first metric key to be the NaN metric; the rest are finite.
    nan_metric = METRIC_KEYS[0]
    finite_val = 0.07

    def _mixed_rows():
        """20 rows where nan_metric is NaN and all others are finite."""
        return [
            {m: (float("nan") if m == nan_metric else finite_val) for m in METRIC_KEYS}
            for _ in range(20)
        ]

    by = {
        "bamengine": _mixed_rows(),
        "mesa": _mixed_rows(),  # matches bamengine on all finite metrics
    }
    report = evaluate_gate(by)

    # (a) the NaN reference metric is indeterminate with within=None
    for fw in ("bamengine", "mesa"):
        metric_info = report["frameworks"][fw]["metrics"][nan_metric]
        assert metric_info["indeterminate"] is True, (
            f"{fw}/{nan_metric}: expected indeterminate=True"
        )
        assert metric_info["within"] is None, f"{fw}/{nan_metric}: expected within=None"

    # (b) finite metrics are not indeterminate
    for fw in ("bamengine", "mesa"):
        for m in METRIC_KEYS:
            if m != nan_metric:
                assert report["frameworks"][fw]["metrics"][m]["indeterminate"] is False

    # (c) candidates matching on finite metrics still pass (indeterminate does not fail)
    assert report["frameworks"]["mesa"]["passed"] is True, (
        "mesa should pass despite indeterminate metric"
    )

    # (d) bamengine itself also passes
    assert report["frameworks"]["bamengine"]["passed"] is True, (
        "bamengine should pass despite indeterminate metric"
    )
