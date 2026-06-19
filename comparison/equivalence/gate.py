from __future__ import annotations

import numpy as np

from comparison.equivalence.metrics import METRIC_KEYS

VALIDATION_HALFWIDTH = {
    "mean_unemployment": 0.025,
    "mean_inflation": 0.03,
    "firm_size_skew": 4.5,
    "phillips_corr": 0.225,
    "okun_corr": 0.14,
    "beveridge_corr": 0.275,
}
NON_BLOCKING = {"netlogo"}


def bamengine_stats(metric_rows: list) -> dict:
    """Compute mean and std for each metric across rows.

    Parameters
    ----------
    metric_rows : list[dict]
        List of dicts, each with keys from METRIC_KEYS.

    Returns
    -------
    dict
        Per-metric stats: {metric: {"mean": float, "std": float}}.
    """
    out = {}
    for m in METRIC_KEYS:
        vals = np.array([r[m] for r in metric_rows], float)
        vals = vals[~np.isnan(vals)]
        out[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def tolerances(be_stats: dict, k: float = 2.0) -> dict:
    """Compute per-metric tolerances floored by validation band.

    Parameters
    ----------
    be_stats : dict
        Output from bamengine_stats().
    k : float
        Multiplier on standard deviation (default 2.0).

    Returns
    -------
    dict
        Per-metric tolerance: max(VALIDATION_HALFWIDTH[m], k * std).
    """
    return {
        m: max(VALIDATION_HALFWIDTH[m], k * be_stats[m]["std"]) for m in METRIC_KEYS
    }


def evaluate_gate(by_framework: dict, k: float = 2.0) -> dict:
    """Evaluate behavioral equivalence for frameworks.

    Parameters
    ----------
    by_framework : dict[str, list[dict]]
        {framework_name: metric_rows}, with "bamengine" as reference.
    k : float
        Multiplier on standard deviation (default 2.0).

    Returns
    -------
    dict
        {
            "tolerances": {metric: float},
            "bamengine": {metric: {"mean": float, "std": float}},
            "frameworks": {
                framework: {
                    "metrics": {
                        metric: {
                            "mean": float,
                            "bamengine_mean": float,
                            "deviation": float,
                            "tolerance": float,
                            "within": bool,
                        }
                    },
                    "passed": bool,
                    "blocking": bool,
                }
            }
        }
    """
    be = bamengine_stats(by_framework["bamengine"])
    tol = tolerances(be, k)
    frameworks = {}
    for fw, rows in by_framework.items():
        stats = bamengine_stats(rows)  # reuse mean/std computation
        metrics = {}
        ok = True
        for m in METRIC_KEYS:
            dev = abs(stats[m]["mean"] - be[m]["mean"])
            within = dev <= tol[m]
            metrics[m] = {
                "mean": stats[m]["mean"],
                "bamengine_mean": be[m]["mean"],
                "deviation": dev,
                "tolerance": tol[m],
                "within": within,
            }
            ok = ok and within
        blocking = fw not in NON_BLOCKING and fw != "bamengine"
        frameworks[fw] = {"metrics": metrics, "passed": ok, "blocking": blocking}
    return {"tolerances": tol, "bamengine": be, "frameworks": frameworks}
