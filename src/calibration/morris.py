"""Morris Method (Elementary Effects) screening for global sensitivity analysis.

This module implements the Morris Method (Morris 1991), which runs multiple
One-at-a-Time (OAT) trajectories from random starting points across the
parameter space. Unlike standard OAT (which depends on a single baseline),
Morris provides two measures per parameter:

- mu* (mu_star): Mean absolute elementary effect -- average importance
- sigma: Std of elementary effects -- interaction/nonlinearity indicator

Classification uses dual thresholds::

    INCLUDE: mu* > threshold OR sigma > threshold
    FIX:     mu* <= threshold AND sigma <= threshold

This catches interaction-prone parameters that OAT would miss: a parameter
with low mu* but high sigma means its effect varies wildly depending on other
parameters' values.

Supports multiple scenarios:
    - baseline: Standard BAM model (Section 3.9.1)
    - growth_plus: Endogenous productivity growth via R&D (Section 3.9.2)
    - buffer_stock: Buffer-stock consumption with R&D (Section 3.9.4)
"""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from calibration.parameter_space import get_parameter_grid
from calibration.sensitivity import ParameterSensitivity, SensitivityResult
from validation import get_validation_func


@dataclass
class MorrisParameterEffect:
    """Morris method results for a single parameter.

    Attributes
    ----------
    name : str
        Parameter name.
    mu : float
        Signed mean elementary effect (can cancel out).
    mu_star : float
        Mean absolute elementary effect (primary importance measure).
    sigma : float
        Standard deviation of elementary effects (interaction indicator).
    elementary_effects : list[float]
        Raw elementary effects from each trajectory.
    value_scores : dict[Any, list[float]]
        Observed scores for each grid value across trajectories.
        Used for best_value estimation and grid pruning.
    """

    name: str
    mu: float
    mu_star: float
    sigma: float
    elementary_effects: list[float]
    value_scores: dict[Any, list[float]] = field(default_factory=dict)


@dataclass
class MorrisResult:
    """Full Morris method screening result.

    Attributes
    ----------
    effects : list[MorrisParameterEffect]
        Per-parameter results.
    n_trajectories : int
        Number of Morris trajectories used.
    n_evaluations : int
        Number of unique configs evaluated.
    scenario : str
        The scenario that was analyzed.
    avg_time_per_run : float
        Average wall-clock time per simulation run (seconds).
    n_seeds : int
        Number of seeds used per evaluation.
    """

    effects: list[MorrisParameterEffect]
    n_trajectories: int
    n_evaluations: int
    scenario: str = "baseline"
    avg_time_per_run: float = 0.0
    n_seeds: int = 1

    @property
    def ranked(self) -> list[MorrisParameterEffect]:
        """Effects ranked by mu_star (highest first)."""
        return sorted(self.effects, key=lambda e: e.mu_star, reverse=True)

    def get_important(
        self,
        mu_star_threshold: float = 0.02,
        sigma_threshold: float = 0.02,
    ) -> tuple[list[str], list[str]]:
        """Categorize parameters using dual threshold.

        A parameter is INCLUDEd if it is either important (high mu*) OR
        interaction-prone (high sigma). It is FIXed only if both are low.

        Parameters
        ----------
        mu_star_threshold : float
            Minimum mu* for inclusion.
        sigma_threshold : float
            Minimum sigma for inclusion (catches interaction-prone params).

        Returns
        -------
        tuple[list[str], list[str]]
            (included, fixed) parameter name lists.
        """
        included, fixed = [], []
        for e in self.effects:
            if e.mu_star > mu_star_threshold or e.sigma > sigma_threshold:
                included.append(e.name)
            else:
                fixed.append(e.name)
        return included, fixed

    def to_sensitivity_result(self) -> SensitivityResult:
        """Convert to SensitivityResult for downstream compatibility.

        Maps mu* to sensitivity, reconstructs per-value scores from
        trajectory observations, enabling zero changes to build_focused_grid
        and all downstream calibration code.

        Returns
        -------
        SensitivityResult
            Compatible result that can be passed to build_focused_grid().
        """
        parameters = []
        for e in self.effects:
            # Reconstruct per-value average scores
            values = sorted(e.value_scores.keys(), key=_sort_key)
            scores = [
                statistics.mean(e.value_scores[v]) if e.value_scores[v] else 0.0
                for v in values
            ]

            if scores:
                best_idx = scores.index(max(scores))
                best_value = values[best_idx]
                best_score = scores[best_idx]
            else:
                best_value = values[0] if values else None
                best_score = 0.0

            parameters.append(
                ParameterSensitivity(
                    name=e.name,
                    values=values,
                    scores=scores,
                    best_value=best_value,
                    best_score=best_score,
                    sensitivity=e.mu_star,
                )
            )

        # Baseline score: average of all observed scores across all params
        all_scores: list[float] = []
        for e in self.effects:
            for vs in e.value_scores.values():
                all_scores.extend(vs)
        baseline_score = statistics.mean(all_scores) if all_scores else 0.0

        return SensitivityResult(
            parameters=parameters,
            baseline_score=baseline_score,
            scenario=self.scenario,
            avg_time_per_run=self.avg_time_per_run,
            n_seeds=self.n_seeds,
        )


def _sort_key(v: Any) -> tuple[int, Any]:
    """Sort key that handles mixed types (str, int, float, bool)."""
    if isinstance(v, bool):
        return (0, int(v))
    if isinstance(v, (int, float)):
        return (1, v)
    return (2, str(v))


def _config_key(config: dict[str, Any]) -> str:
    """Create a deterministic string key from a config dict."""
    return json.dumps(dict(sorted(config.items())), sort_keys=True, default=str)


def _generate_trajectory(
    grid: dict[str, list[Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Generate a single Morris trajectory through parameter space.

    Starting from a random point in the grid, perturbs each parameter
    exactly once in a random order, producing p+1 configs where consecutive
    configs differ in exactly one parameter.

    Parameters
    ----------
    grid : dict
        Parameter grid (param_name -> list of possible values).
        Parameters with a single value are included but not perturbed.
    rng : numpy.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    list[dict[str, Any]]
        List of p+1 config dicts forming the trajectory.
    """
    param_names = list(grid.keys())
    p = len(param_names)

    # Random starting point: one random value per parameter
    config: dict[str, Any] = {}
    for name in param_names:
        values = grid[name]
        config[name] = values[int(rng.integers(len(values)))]

    trajectory = [config.copy()]

    # Random parameter ordering
    order = rng.permutation(p)

    for idx in order:
        name = param_names[int(idx)]
        values = grid[name]

        if len(values) <= 1:
            # Single-value parameter: no perturbation possible
            trajectory.append(config.copy())
            continue

        # Pick a different value
        current = config[name]
        candidates = [v for v in values if v != current]
        new_value = candidates[int(rng.integers(len(candidates)))]

        config = config.copy()
        config[name] = new_value
        trajectory.append(config)

    return trajectory


def _evaluate_config(
    config: dict[str, Any],
    scenario: str,
    seeds: list[int],
    n_periods: int,
) -> tuple[dict[str, Any], float, float]:
    """Evaluate a complete config across multiple seeds.

    Worker function for parallel execution. Must be module-level
    for ProcessPoolExecutor pickling.

    Parameters
    ----------
    config : dict
        Complete parameter configuration.
    scenario : str
        Scenario name.
    seeds : list[int]
        Seeds to evaluate.
    n_periods : int
        Simulation periods.

    Returns
    -------
    tuple[dict, float, float]
        (config, avg_score, elapsed_seconds)
    """
    validate = get_validation_func(scenario)

    total_score = 0.0
    t0 = time.monotonic()

    try:
        for seed in seeds:
            result = validate(seed=seed, n_periods=n_periods, **config)
            total_score += result.total_score
    except Exception:
        # Extreme parameter combos can crash the simulation (e.g. full
        # economic collapse leading to division by zero in metrics).
        # Treat as worst-possible score.
        elapsed = time.monotonic() - t0
        return config, 0.0, elapsed

    elapsed = time.monotonic() - t0
    avg_score = total_score / len(seeds)

    return config, avg_score, elapsed


def run_morris_screening(
    scenario: str = "baseline",
    grid: dict[str, list[Any]] | None = None,
    n_trajectories: int = 10,
    seed: int = 0,
    n_seeds: int = 1,
    n_periods: int = 1000,
    n_workers: int = 10,
) -> MorrisResult:
    """Run Morris Method screening analysis.

    Generates multiple OAT trajectories from random starting points,
    evaluates all unique configs in parallel, then computes per-parameter
    elementary effects (mu*, sigma) for importance and interaction
    classification.

    Parameters
    ----------
    scenario : str
        Scenario to calibrate.
    grid : dict, optional
        Parameter grid. Defaults to scenario-specific grid.
    n_trajectories : int
        Number of Morris trajectories (more = more reliable estimates).
    seed : int
        Base random seed for trajectory generation and evaluation.
    n_seeds : int
        Number of seeds per config evaluation.
    n_periods : int
        Number of simulation periods.
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    MorrisResult
        Morris screening result with per-parameter mu*, sigma, and
        value scores.
    """
    if grid is None:
        grid = get_parameter_grid(scenario)

    # Separate multi-value params (active) from single-value params (fixed)
    active_grid = {k: v for k, v in grid.items() if len(v) > 1}
    fixed_grid = {k: v[0] for k, v in grid.items() if len(v) <= 1}

    param_names = list(active_grid.keys())
    sim_seeds = list(range(seed, seed + n_seeds))
    rng = np.random.default_rng(seed)

    # Step 1: Generate all trajectories
    print(
        f"[{scenario}] Generating {n_trajectories} Morris trajectories "
        f"({len(param_names)} active params)..."
    )

    trajectories: list[list[dict[str, Any]]] = []
    for _ in range(n_trajectories):
        traj = _generate_trajectory(active_grid, rng)
        # Add fixed single-value params to each config
        if fixed_grid:
            traj = [{**fixed_grid, **cfg} for cfg in traj]
        trajectories.append(traj)

    # Step 2: Collect unique configs for evaluation
    config_scores: dict[str, float] = {}
    config_map: dict[str, dict[str, Any]] = {}

    for traj in trajectories:
        for cfg in traj:
            key = _config_key(cfg)
            if key not in config_map:
                config_map[key] = cfg

    unique_configs = list(config_map.values())
    n_unique = len(unique_configs)
    total_sim_runs = n_unique * n_seeds

    print(
        f"  {n_unique} unique configs to evaluate "
        f"({total_sim_runs} sim runs across {n_seeds} seed(s))"
    )

    # Step 3: Evaluate all configs
    total_elapsed = 0.0
    completed = 0

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_config, cfg, scenario, sim_seeds, n_periods
                ): _config_key(cfg)
                for cfg in unique_configs
            }

            for future in as_completed(futures):
                cfg, score, elapsed = future.result()
                key = _config_key(cfg)
                config_scores[key] = score
                total_elapsed += elapsed
                completed += 1

                if completed % 20 == 0 or completed == n_unique:
                    pct = 100.0 * completed / n_unique
                    print(f"  [{completed}/{n_unique}] ({pct:.0f}%) evaluated")
    else:
        for cfg in unique_configs:
            cfg, score, elapsed = _evaluate_config(cfg, scenario, sim_seeds, n_periods)
            key = _config_key(cfg)
            config_scores[key] = score
            total_elapsed += elapsed
            completed += 1

            if completed % 20 == 0 or completed == n_unique:
                pct = 100.0 * completed / n_unique
                print(f"  [{completed}/{n_unique}] ({pct:.0f}%) evaluated")

    avg_time = total_elapsed / total_sim_runs if total_sim_runs > 0 else 0.0
    print(f"  Avg time per sim run: {avg_time:.1f}s")

    # Step 4: Compute elementary effects per parameter
    ee_by_param: dict[str, list[float]] = {name: [] for name in param_names}
    vs_by_param: dict[str, dict[Any, list[float]]] = {name: {} for name in param_names}

    for traj in trajectories:
        for i in range(1, len(traj)):
            cfg_before = traj[i - 1]
            cfg_after = traj[i]

            # Find which parameter changed
            changed_param = None
            for name in param_names:
                if cfg_before[name] != cfg_after[name]:
                    changed_param = name
                    break

            if changed_param is None:
                # Single-value param step or no change
                continue

            score_before = config_scores[_config_key(cfg_before)]
            score_after = config_scores[_config_key(cfg_after)]
            ee = score_after - score_before

            ee_by_param[changed_param].append(ee)

            # Track per-value scores for best_value estimation
            val_before = cfg_before[changed_param]
            vs_by_param[changed_param].setdefault(val_before, []).append(score_before)

            val_after = cfg_after[changed_param]
            vs_by_param[changed_param].setdefault(val_after, []).append(score_after)

    # Step 5: Aggregate into MorrisParameterEffect
    effects: list[MorrisParameterEffect] = []

    for name in param_names:
        ees = ee_by_param[name]
        if ees:
            mu = statistics.mean(ees)
            mu_star = statistics.mean(abs(e) for e in ees)
            sigma = statistics.stdev(ees) if len(ees) > 1 else 0.0
        else:
            mu = mu_star = sigma = 0.0

        effects.append(
            MorrisParameterEffect(
                name=name,
                mu=mu,
                mu_star=mu_star,
                sigma=sigma,
                elementary_effects=ees,
                value_scores=vs_by_param.get(name, {}),
            )
        )

    # Add single-value params with zero effects (for downstream compatibility)
    for name in grid:
        if len(grid[name]) <= 1:
            effects.append(
                MorrisParameterEffect(
                    name=name,
                    mu=0.0,
                    mu_star=0.0,
                    sigma=0.0,
                    elementary_effects=[],
                    value_scores={grid[name][0]: []} if grid[name] else {},
                )
            )

    return MorrisResult(
        effects=effects,
        n_trajectories=n_trajectories,
        n_evaluations=n_unique,
        scenario=scenario,
        avg_time_per_run=avg_time,
        n_seeds=n_seeds,
    )


def print_morris_report(
    result: MorrisResult,
    mu_star_threshold: float = 0.02,
    sigma_threshold: float = 0.02,
) -> None:
    """Print formatted Morris method screening report.

    Parameters
    ----------
    result : MorrisResult
        Result from run_morris_screening().
    mu_star_threshold : float
        Threshold for mu* classification.
    sigma_threshold : float
        Threshold for sigma classification.
    """
    print("\n" + "=" * 80)
    print(f"MORRIS METHOD SCREENING RESULTS ({result.scenario})")
    print("=" * 80)
    print(f"\nTrajectories:   {result.n_trajectories}")
    print(f"Unique configs: {result.n_evaluations}")
    print(f"Seeds per eval: {result.n_seeds}")
    print(f"Avg time/run:   {result.avg_time_per_run:.1f}s")

    header = (
        f"{'Parameter':<30} {'mu*':>6} {'sigma':>6} {'s/mu*':>6} "
        f"{'mu':>7} {'#EE':>4} {'Class':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    included, fixed = result.get_important(mu_star_threshold, sigma_threshold)

    for e in result.ranked:
        ratio = e.sigma / e.mu_star if e.mu_star > 0 else 0.0
        cls = "INCLUDE" if e.name in included else "FIX"
        reason = ""
        if cls == "INCLUDE":
            if e.mu_star > mu_star_threshold and e.sigma > sigma_threshold:
                reason = " (mu*+s)"
            elif e.sigma > sigma_threshold:
                reason = " (s)"

        line = (
            f"{e.name:<30} {e.mu_star:>6.3f} {e.sigma:>6.3f} {ratio:>6.2f} "
            f"{e.mu:>+7.3f} {len(e.elementary_effects):>4d} {cls:>8}{reason}"
        )
        print(line)

    print("\n" + "=" * 80)
    print("PARAMETER CLASSIFICATION")
    print("=" * 80)
    print(
        f"INCLUDE (mu* > {mu_star_threshold} OR sigma > {sigma_threshold}): "
        f"{', '.join(included) or 'None'}"
    )
    print(
        f"FIX (mu* <= {mu_star_threshold} AND sigma <= {sigma_threshold}): "
        f"{', '.join(fixed) or 'None'}"
    )
    print("=" * 80 + "\n")
