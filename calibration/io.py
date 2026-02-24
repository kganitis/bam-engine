"""Central serialization for calibration results.

All save/load operations use a consistent JSON schema with version tracking.
Timestamped output directories keep results organized across runs.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from calibration.analysis import CalibrationResult
from calibration.morris import MorrisParameterEffect, MorrisResult
from calibration.sensitivity import (
    PairInteraction,
    PairwiseResult,
    ParameterSensitivity,
    SensitivityResult,
)

_SCHEMA_VERSION = 1

# Default output directory
OUTPUT_DIR = Path(__file__).parent / "output"


def create_run_dir(scenario: str, output_dir: Path | None = None) -> Path:
    """Create timestamped output directory.

    Parameters
    ----------
    scenario : str
        Scenario name (included in directory name).
    output_dir : Path, optional
        Parent directory. Defaults to calibration/output/.

    Returns
    -------
    Path
        Path to the created directory.
    """
    parent = output_dir or OUTPUT_DIR
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H%M%S")
    run_dir = parent / f"{timestamp}_{scenario}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# =============================================================================
# Sensitivity
# =============================================================================


def save_sensitivity(result: SensitivityResult, path: Path) -> None:
    """Save sensitivity result to JSON."""
    data = {
        "_schema_version": _SCHEMA_VERSION,
        "scenario": result.scenario,
        "baseline_score": result.baseline_score,
        "avg_time_per_run": result.avg_time_per_run,
        "n_seeds": result.n_seeds,
        "parameters": {
            p.name: {
                "sensitivity": p.sensitivity,
                "best_value": p.best_value,
                "best_score": p.best_score,
                "values": p.values,
                "scores": p.scores,
                "group_scores": p.group_scores,
            }
            for p in result.parameters
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_sensitivity(path: Path) -> SensitivityResult:
    """Load sensitivity result from JSON."""
    with open(path) as f:
        data = json.load(f)

    parameters = []
    for name, pdata in data["parameters"].items():
        parameters.append(
            ParameterSensitivity(
                name=name,
                values=pdata["values"],
                scores=pdata["scores"],
                best_value=pdata["best_value"],
                best_score=pdata["best_score"],
                sensitivity=pdata["sensitivity"],
                group_scores=pdata.get("group_scores", {}),
            )
        )

    return SensitivityResult(
        parameters=parameters,
        baseline_score=data["baseline_score"],
        scenario=data.get("scenario", "baseline"),
        avg_time_per_run=data.get("avg_time_per_run", 0.0),
        n_seeds=data.get("n_seeds", 1),
    )


# =============================================================================
# Morris
# =============================================================================


def save_morris(result: MorrisResult, path: Path) -> None:
    """Save Morris result to JSON."""
    data = {
        "_schema_version": _SCHEMA_VERSION,
        "scenario": result.scenario,
        "n_trajectories": result.n_trajectories,
        "n_evaluations": result.n_evaluations,
        "avg_time_per_run": result.avg_time_per_run,
        "n_seeds": result.n_seeds,
        "effects": {
            e.name: {
                "mu": e.mu,
                "mu_star": e.mu_star,
                "sigma": e.sigma,
                "elementary_effects": e.elementary_effects,
                "value_scores": {
                    str(v): scores for v, scores in e.value_scores.items()
                },
            }
            for e in result.effects
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_morris(path: Path) -> MorrisResult:
    """Load Morris result from JSON."""
    with open(path) as f:
        data = json.load(f)

    effects = []
    for name, edata in data["effects"].items():
        # Reconstruct value_scores with original types where possible
        value_scores: dict[Any, list[float]] = {}
        for k, v in edata.get("value_scores", {}).items():
            # Try to recover numeric types from string keys
            try:
                key: Any = int(k)
            except ValueError:
                try:
                    key = float(k)
                except ValueError:
                    key = k
            value_scores[key] = v

        effects.append(
            MorrisParameterEffect(
                name=name,
                mu=edata["mu"],
                mu_star=edata["mu_star"],
                sigma=edata["sigma"],
                elementary_effects=edata["elementary_effects"],
                value_scores=value_scores,
            )
        )

    return MorrisResult(
        effects=effects,
        n_trajectories=data["n_trajectories"],
        n_evaluations=data["n_evaluations"],
        scenario=data.get("scenario", "baseline"),
        avg_time_per_run=data.get("avg_time_per_run", 0.0),
        n_seeds=data.get("n_seeds", 1),
    )


# =============================================================================
# Screening
# =============================================================================


def save_screening(
    results: list[CalibrationResult],
    sensitivity: SensitivityResult,
    grid: dict[str, list[Any]],
    fixed: dict[str, Any],
    patterns: dict[str, dict[Any, int]],
    scenario: str,
    path: Path,
) -> None:
    """Save screening results to JSON."""
    data = {
        "_schema_version": _SCHEMA_VERSION,
        "scenario": scenario,
        "avg_time_per_run": sensitivity.avg_time_per_run,
        "sensitivity": {
            p.name: {"sensitivity": p.sensitivity, "best_value": p.best_value}
            for p in sensitivity.parameters
        },
        "grid_params": grid,
        "fixed_params": fixed,
        "patterns": {
            param: {str(v): c for v, c in counts.items()}
            for param, counts in patterns.items()
        },
        "results": [
            {
                "rank": i + 1,
                "params": r.params,
                "single_score": r.single_score,
                "n_pass": r.n_pass,
                "n_warn": r.n_warn,
                "n_fail": r.n_fail,
            }
            for i, r in enumerate(results)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_screening(
    path: Path,
) -> tuple[list[CalibrationResult], float]:
    """Load screening results from JSON. Returns (results, avg_time_per_run)."""
    with open(path) as f:
        data = json.load(f)

    results = [
        CalibrationResult(
            params=r["params"],
            single_score=r["single_score"],
            n_pass=r["n_pass"],
            n_warn=r["n_warn"],
            n_fail=r["n_fail"],
        )
        for r in data["results"]
    ]
    return results, data.get("avg_time_per_run", 0.0)


# =============================================================================
# Stability
# =============================================================================


def save_stability(
    results: list[CalibrationResult],
    scenario: str,
    path: Path,
) -> None:
    """Save stability testing results to JSON."""
    data = {
        "_schema_version": _SCHEMA_VERSION,
        "scenario": scenario,
        "results": [
            {
                "rank": i + 1,
                "params": r.params,
                "combined_score": r.combined_score,
                "mean_score": r.mean_score,
                "std_score": r.std_score,
                "pass_rate": r.pass_rate,
                "seed_scores": r.seed_scores,
                "single_score": r.single_score,
            }
            for i, r in enumerate(results)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_stability(path: Path) -> list[CalibrationResult]:
    """Load stability results from JSON."""
    with open(path) as f:
        data = json.load(f)

    return [
        CalibrationResult(
            params=r["params"],
            single_score=r["single_score"],
            n_pass=r.get("n_pass", 0),
            n_warn=r.get("n_warn", 0),
            n_fail=r.get("n_fail", 0),
            mean_score=r.get("mean_score"),
            std_score=r.get("std_score"),
            pass_rate=r.get("pass_rate"),
            combined_score=r.get("combined_score"),
            seed_scores=r.get("seed_scores"),
        )
        for r in data["results"]
    ]


# =============================================================================
# Pairwise
# =============================================================================


def save_pairwise(result: PairwiseResult, scenario: str, path: Path) -> None:
    """Save pairwise interaction results to JSON."""
    data = {
        "_schema_version": _SCHEMA_VERSION,
        "scenario": scenario,
        "baseline_score": result.baseline_score,
        "interactions": [
            {
                "param_a": ix.param_a,
                "value_a": ix.value_a,
                "param_b": ix.param_b,
                "value_b": ix.value_b,
                "combined_score": ix.combined_score,
                "individual_a_score": ix.individual_a_score,
                "individual_b_score": ix.individual_b_score,
                "interaction_strength": ix.interaction_strength,
            }
            for ix in result.ranked
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_pairwise(path: Path) -> PairwiseResult:
    """Load pairwise results from JSON."""
    with open(path) as f:
        data = json.load(f)

    interactions = [
        PairInteraction(
            param_a=ix["param_a"],
            param_b=ix["param_b"],
            value_a=ix["value_a"],
            value_b=ix["value_b"],
            individual_a_score=ix["individual_a_score"],
            individual_b_score=ix["individual_b_score"],
            combined_score=ix["combined_score"],
            baseline_score=data["baseline_score"],
            interaction_strength=ix["interaction_strength"],
        )
        for ix in data["interactions"]
    ]

    return PairwiseResult(
        interactions=interactions,
        scenario=data.get("scenario", "baseline"),
        baseline_score=data["baseline_score"],
    )
