"""One-At-a-Time (OAT) sensitivity analysis with pairwise interaction scanning.

This module provides sensitivity analysis functionality to identify which
parameters have the most impact on validation scores.

Supports multiple scenarios:
    - baseline: Standard BAM model (Section 3.9.1)
    - growth_plus: Endogenous productivity growth via R&D (Section 3.9.2)
    - buffer_stock: Buffer-stock consumption with R&D (Section 3.9.4)
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from calibration.parameter_space import get_default_values, get_parameter_grid
from validation import get_validation_func


@dataclass
class ParameterSensitivity:
    """Sensitivity result for a single parameter.

    Attributes
    ----------
    name : str
        Parameter name.
    values : list
        All values tested for this parameter.
    scores : list[float]
        Validation scores for each value (averaged across seeds).
    best_value : Any
        Value that produced the highest score.
    best_score : float
        Highest score achieved.
    sensitivity : float
        Score range (max - min), indicating parameter importance.
    group_scores : dict[str, list[float]]
        Per-metric-group scores for each value. Keys are MetricGroup names
        (e.g., "TIME_SERIES", "CURVES"), values are lists parallel to `scores`.
    """

    name: str
    values: list[Any]
    scores: list[float]
    best_value: Any
    best_score: float
    sensitivity: float
    group_scores: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class SensitivityResult:
    """Full sensitivity analysis result.

    Attributes
    ----------
    parameters : list[ParameterSensitivity]
        Sensitivity results for all parameters.
    baseline_score : float
        Score with all default values.
    scenario : str
        The scenario that was analyzed.
    avg_time_per_run : float
        Average wall-clock time per simulation run (seconds).
    n_seeds : int
        Number of seeds used per evaluation.
    """

    parameters: list[ParameterSensitivity]
    baseline_score: float
    scenario: str = "baseline"
    avg_time_per_run: float = 0.0
    n_seeds: int = 1

    @property
    def ranked(self) -> list[ParameterSensitivity]:
        """Parameters ranked by sensitivity (highest first)."""
        return sorted(self.parameters, key=lambda p: p.sensitivity, reverse=True)

    def get_important(
        self,
        sensitivity_threshold: float = 0.02,
    ) -> tuple[list[str], list[str]]:
        """Categorize parameters by sensitivity.

        Parameters
        ----------
        sensitivity_threshold : float
            Minimum sensitivity (Δ) for inclusion in grid search.

        Returns
        -------
        tuple[list[str], list[str]]
            (included, fixed) parameter name lists.
        """
        included, fixed = [], []
        for p in self.parameters:
            if p.sensitivity > sensitivity_threshold:
                included.append(p.name)
            else:
                fixed.append(p.name)
        return included, fixed

    def prune_grid(
        self,
        grid: dict[str, list[Any]],
        pruning_threshold: float | None,
    ) -> dict[str, list[Any]]:
        """Remove poorly-scoring values from grid based on OAT results.

        Parameters
        ----------
        grid : dict
            Parameter grid to prune (values per parameter).
        pruning_threshold : float or None
            Maximum score gap from best value. Values with
            ``(best_score - score) > pruning_threshold`` are dropped.
            ``None`` disables pruning (returns grid unchanged).

        Returns
        -------
        dict[str, list[Any]]
            Pruned grid. Always keeps at least the best value per parameter.
            Unknown parameters or values are kept (conservative).
        """
        if pruning_threshold is None:
            return grid

        # Build lookup: param_name -> {value: score}
        score_map: dict[str, dict[Any, float]] = {}
        for p in self.parameters:
            score_map[p.name] = dict(zip(p.values, p.scores, strict=True))

        pruned: dict[str, list[Any]] = {}
        for name, values in grid.items():
            if name not in score_map:
                # Unknown param — keep all values
                pruned[name] = values
                continue

            pscores = score_map[name]
            best = max(pscores.values())

            kept = []
            for v in values:
                if v not in pscores:
                    # Unknown value — keep (conservative)
                    kept.append(v)
                elif (best - pscores[v]) <= pruning_threshold:
                    kept.append(v)

            # Safety net: always keep at least the best value
            if not kept:
                best_val = max(pscores, key=pscores.get)  # type: ignore[arg-type]
                if best_val in values:
                    kept = [best_val]
                else:
                    kept = values[:1]

            pruned[name] = kept

        return pruned


def _evaluate_param_value(
    param_name: str,
    value: Any,
    baseline: dict[str, Any],
    scenario: str,
    seeds: list[int],
    n_periods: int,
) -> tuple[Any, float, dict[str, float], float]:
    """Evaluate a single parameter value across multiple seeds.

    Returns
    -------
    tuple
        (value, avg_score, group_scores, elapsed_seconds)
    """
    config = baseline.copy()
    config[param_name] = value
    validate = get_validation_func(scenario)

    total_score = 0.0
    group_totals: dict[str, float] = {}
    group_counts: dict[str, int] = {}
    t0 = time.monotonic()

    for seed in seeds:
        result = validate(seed=seed, n_periods=n_periods, **config)
        total_score += result.total_score

        # Accumulate per-group scores
        for mr in result.metric_results:
            gname = mr.group.name
            group_totals[gname] = group_totals.get(gname, 0.0) + mr.score * mr.weight
            group_counts[gname] = group_counts.get(gname, 0) + 1

    elapsed = time.monotonic() - t0
    n = len(seeds)
    avg_score = total_score / n

    # Average group scores (weighted score per metric, averaged across seeds)
    avg_groups = {}
    for gname in group_totals:
        avg_groups[gname] = group_totals[gname] / n

    return value, avg_score, avg_groups, elapsed


def run_sensitivity_analysis(
    scenario: str = "baseline",
    grid: dict[str, list[Any]] | None = None,
    baseline: dict[str, Any] | None = None,
    seed: int = 0,
    n_seeds: int = 1,
    n_periods: int = 1000,
    n_workers: int = 10,
) -> SensitivityResult:
    """Run OAT sensitivity analysis.

    Tests each parameter independently while holding others at baseline values.
    Supports multi-seed evaluation for more robust sensitivity measurement.

    Parameters
    ----------
    scenario : str
        Scenario to calibrate ("baseline", "growth_plus", or "buffer_stock").
    grid : dict, optional
        Parameter grid. Defaults to scenario-specific grid.
    baseline : dict, optional
        Baseline parameter values. Defaults to scenario-specific defaults.
    seed : int
        Base random seed (used as first seed).
    n_seeds : int
        Number of seeds per evaluation. Seeds are [seed, seed+1, ..., seed+n_seeds-1].
    n_periods : int
        Number of simulation periods.
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    SensitivityResult
        Sensitivity ranking of all parameters.
    """
    if grid is None:
        grid = get_parameter_grid(scenario)
    if baseline is None:
        baseline = get_default_values(scenario).copy()

    seeds = list(range(seed, seed + n_seeds))

    validate = get_validation_func(scenario)

    # Get baseline score (averaged across seeds)
    baseline_total = 0.0
    for s in seeds:
        baseline_result = validate(seed=s, n_periods=n_periods, **baseline)
        baseline_total += baseline_result.total_score
    baseline_score = baseline_total / n_seeds
    print(f"[{scenario}] Baseline score: {baseline_score:.4f} ({n_seeds} seed(s))")

    # Test each parameter
    results: list[ParameterSensitivity] = []
    total_runs = sum(len(values) for values in grid.values())
    completed = 0
    total_elapsed = 0.0
    total_sim_runs = 0

    for param_name, values in grid.items():
        print(f"Testing {param_name}...")

        # Run all values for this parameter in parallel
        param_results: list[tuple[Any, float, dict[str, float], float]] = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_param_value,
                    param_name,
                    value,
                    baseline,
                    scenario,
                    seeds,
                    n_periods,
                )
                for value in values
            ]
            for future in as_completed(futures):
                val, score, groups, elapsed = future.result()
                param_results.append((val, score, groups, elapsed))
                completed += 1
                total_elapsed += elapsed
                total_sim_runs += n_seeds
                print(
                    f"  [{completed}/{total_runs}] "
                    f"{param_name}={val}: {score:.4f} ({elapsed:.1f}s)"
                )

        # Sort by value order
        param_results.sort(key=lambda x: values.index(x[0]))
        vals = [x[0] for x in param_results]
        scores = [x[1] for x in param_results]

        # Build group_scores dict: {group_name: [score_per_value]}
        all_groups: set[str] = set()
        for _, _, groups, _ in param_results:
            all_groups.update(groups.keys())

        group_scores: dict[str, list[float]] = {}
        for gname in sorted(all_groups):
            group_scores[gname] = [r[2].get(gname, 0.0) for r in param_results]

        best_idx = scores.index(max(scores))
        sensitivity = max(scores) - min(scores)

        results.append(
            ParameterSensitivity(
                name=param_name,
                values=vals,
                scores=scores,
                best_value=vals[best_idx],
                best_score=scores[best_idx],
                sensitivity=sensitivity,
                group_scores=group_scores,
            )
        )

    avg_time = total_elapsed / total_sim_runs if total_sim_runs > 0 else 0.0

    return SensitivityResult(
        parameters=results,
        baseline_score=baseline_score,
        scenario=scenario,
        avg_time_per_run=avg_time,
        n_seeds=n_seeds,
    )


def print_sensitivity_report(
    result: SensitivityResult,
    sensitivity_threshold: float = 0.02,
) -> None:
    """Print formatted sensitivity analysis report with score decomposition.

    Parameters
    ----------
    result : SensitivityResult
        Result from run_sensitivity_analysis().
    sensitivity_threshold : float
        Threshold for INCLUDE/FIX classification (informational preview).
    """
    print("\n" + "=" * 80)
    print(f"SENSITIVITY ANALYSIS RESULTS ({result.scenario})")
    print("=" * 80)
    print(f"\nBaseline score: {result.baseline_score:.4f}")
    print(f"Seeds per eval: {result.n_seeds}")
    print(f"Avg time/run:   {result.avg_time_per_run:.1f}s")

    # Collect all group names across all parameters
    all_groups: set[str] = set()
    for p in result.parameters:
        all_groups.update(p.group_scores.keys())
    group_names = sorted(all_groups)

    # Short group name labels
    group_labels = {g: g[:4] for g in group_names}

    header = f"{'Parameter':<30} {'Δ':>6} {'Best':>12} {'Score':>6}"
    for g in group_names:
        header += f"  {group_labels[g]:>5}"
    print(f"\n{header}")
    print("-" * len(header))

    for p in result.ranked:
        line = f"{p.name:<30} {p.sensitivity:>6.3f} {p.best_value!s:>12} {p.best_score:>6.3f}"
        for g in group_names:
            if g in p.group_scores:
                gscores = p.group_scores[g]
                delta = max(gscores) - min(gscores)
                line += f"  {delta:>+5.3f}"
            else:
                line += f"  {'--':>5}"
        print(line)

    included, fixed = result.get_important(sensitivity_threshold)

    print("\n" + "=" * 80)
    print("PARAMETER IMPORTANCE")
    print("=" * 80)
    print(f"INCLUDE (Δ > {sensitivity_threshold}): {', '.join(included) or 'None'}")
    print(f"FIX (Δ ≤ {sensitivity_threshold}):     {', '.join(fixed) or 'None'}")
    print("=" * 80 + "\n")


# =============================================================================
# Pairwise Interaction Scanning
# =============================================================================


@dataclass
class PairInteraction:
    """Interaction result for a pair of parameters."""

    param_a: str
    param_b: str
    value_a: Any
    value_b: Any
    individual_a_score: float
    individual_b_score: float
    combined_score: float
    baseline_score: float
    interaction_strength: float  # |combined - a - b + baseline|


@dataclass
class PairwiseResult:
    """Full pairwise interaction analysis result."""

    interactions: list[PairInteraction]
    scenario: str
    baseline_score: float

    @property
    def ranked(self) -> list[PairInteraction]:
        """Interactions ranked by strength (highest first)."""
        return sorted(
            self.interactions, key=lambda x: x.interaction_strength, reverse=True
        )

    @property
    def synergies(self) -> list[PairInteraction]:
        """Positive interactions (combined > expected)."""
        return [
            x
            for x in self.ranked
            if x.combined_score
            > x.individual_a_score + x.individual_b_score - x.baseline_score
        ]

    @property
    def conflicts(self) -> list[PairInteraction]:
        """Negative interactions (combined < expected)."""
        return [
            x
            for x in self.ranked
            if x.combined_score
            < x.individual_a_score + x.individual_b_score - x.baseline_score
        ]


def _evaluate_pair(
    param_a: str,
    value_a: Any,
    param_b: str,
    value_b: Any,
    baseline: dict[str, Any],
    scenario: str,
    seeds: list[int],
    n_periods: int,
) -> tuple[Any, Any, float]:
    """Evaluate a pair of parameter values."""
    config = baseline.copy()
    config[param_a] = value_a
    config[param_b] = value_b
    validate = get_validation_func(scenario)

    total = 0.0
    for seed in seeds:
        result = validate(seed=seed, n_periods=n_periods, **config)
        total += result.total_score
    return value_a, value_b, total / len(seeds)


def run_pairwise_analysis(
    params: list[str],
    grid: dict[str, list[Any]],
    best_values: dict[str, Any],
    scenario: str = "baseline",
    seed: int = 0,
    n_seeds: int = 3,
    n_periods: int = 1000,
    n_workers: int = 10,
) -> PairwiseResult:
    """Run pairwise interaction analysis on included parameters.

    For each pair of included params, tests all value combinations while
    fixing others at best values. Measures interaction strength.

    Parameters
    ----------
    params : list[str]
        List of included parameter names.
    grid : dict
        Full parameter grid.
    best_values : dict
        Best value for each parameter (from sensitivity analysis).
    scenario : str
        Scenario name.
    seed : int
        Base random seed.
    n_seeds : int
        Seeds per evaluation.
    n_periods : int
        Simulation periods.
    n_workers : int
        Parallel workers.

    Returns
    -------
    PairwiseResult
        Pairwise interaction results.
    """
    seeds = list(range(seed, seed + n_seeds))
    validate = get_validation_func(scenario)

    # Baseline score
    baseline_config = best_values.copy()
    baseline_total = 0.0
    for s in seeds:
        r = validate(seed=s, n_periods=n_periods, **baseline_config)
        baseline_total += r.total_score
    baseline_score = baseline_total / n_seeds

    # Compute individual parameter scores
    individual_scores: dict[str, dict[Any, float]] = {}
    for param in params:
        individual_scores[param] = {}
        for value in grid.get(param, []):
            config = best_values.copy()
            config[param] = value
            total = 0.0
            for s in seeds:
                r = validate(seed=s, n_periods=n_periods, **config)
                total += r.total_score
            individual_scores[param][value] = total / n_seeds

    # Test all pairs
    interactions: list[PairInteraction] = []
    pairs = [
        (params[i], params[j])
        for i in range(len(params))
        for j in range(i + 1, len(params))
    ]

    for param_a, param_b in pairs:
        print(f"Testing pair: {param_a} × {param_b}")
        values_a = grid.get(param_a, [best_values.get(param_a)])
        values_b = grid.get(param_b, [best_values.get(param_b)])

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for va in values_a:
                for vb in values_b:
                    f = executor.submit(
                        _evaluate_pair,
                        param_a,
                        va,
                        param_b,
                        vb,
                        best_values,
                        scenario,
                        seeds,
                        n_periods,
                    )
                    futures[f] = (va, vb)

            for future in as_completed(futures):
                va, vb, combined = future.result()
                score_a = individual_scores[param_a].get(va, baseline_score)
                score_b = individual_scores[param_b].get(vb, baseline_score)
                expected = score_a + score_b - baseline_score
                strength = abs(combined - expected)

                interactions.append(
                    PairInteraction(
                        param_a=param_a,
                        param_b=param_b,
                        value_a=va,
                        value_b=vb,
                        individual_a_score=score_a,
                        individual_b_score=score_b,
                        combined_score=combined,
                        baseline_score=baseline_score,
                        interaction_strength=strength,
                    )
                )

    return PairwiseResult(
        interactions=interactions,
        scenario=scenario,
        baseline_score=baseline_score,
    )


def print_pairwise_report(result: PairwiseResult, top_n: int = 20) -> None:
    """Print formatted pairwise interaction report."""
    print("\n" + "=" * 80)
    print(f"PAIRWISE INTERACTION ANALYSIS ({result.scenario})")
    print("=" * 80)

    print(f"\nTop {top_n} strongest interactions:")
    print(
        f"{'Param A':<20} {'Val A':>8} {'Param B':<20} {'Val B':>8} "
        f"{'Combined':>8} {'Expected':>8} {'Strength':>8} {'Type':>8}"
    )
    print("-" * 100)

    for ix in result.ranked[:top_n]:
        expected = ix.individual_a_score + ix.individual_b_score - ix.baseline_score
        itype = "synergy" if ix.combined_score > expected else "conflict"
        print(
            f"{ix.param_a:<20} {ix.value_a!s:>8} {ix.param_b:<20} {ix.value_b!s:>8} "
            f"{ix.combined_score:>8.4f} {expected:>8.4f} "
            f"{ix.interaction_strength:>8.4f} {itype:>8}"
        )

    n_syn = len(result.synergies)
    n_con = len(result.conflicts)
    print(f"\nSynergies: {n_syn} | Conflicts: {n_con}")
    print("=" * 80 + "\n")
