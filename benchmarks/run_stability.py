"""1000-seed stability benchmarking for validation scenarios.

Runs seed stability tests across all three scenarios (baseline, growth+,
buffer-stock) with 1000 seeds parallelized across 10 workers. Produces
JSON result files for the bamengine.org stability dashboard.

Usage:
    # Current working tree
    PYTHONPATH=src python benchmarks/run_stability.py

    # Single scenario
    PYTHONPATH=src python benchmarks/run_stability.py --scenario baseline

    # Historical commits (see --help for full options)
    python benchmarks/run_stability.py --tags v0.5.0..v0.6.2
"""

import argparse
import atexit
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

N_WORKERS = 10
SEEDS_PER_CHUNK = 100
N_CHUNKS = 10  # Total: 1000 seeds
N_PERIODS = 1000
OUTPUT_DIR = Path("benchmarks/results")

ALL_SCENARIOS = ["baseline", "growth_plus", "buffer_stock"]

SCENARIO_FUNCS = {
    "baseline": "run_stability_test",
    "growth_plus": "run_growth_plus_stability_test",
    "buffer_stock": "run_buffer_stock_stability_test",
}

GROUP_DISPLAY_NAMES = {
    "TIME_SERIES": "Time Series",
    "CURVES": "Curves",
    "DISTRIBUTION": "Distribution",
    "GROWTH": "Growth",
    "FINANCIAL": "Financial",
    "GROWTH_RATE_DIST": "Growth Rate Distribution",
}

MAX_COMMITS_WITHOUT_FORCE = 20


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    """Run a git command and return stripped stdout."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _get_git_metadata() -> dict:
    """Capture git metadata for the current HEAD."""
    commit = _git("rev-parse", "HEAD")
    commit_short = _git("rev-parse", "--short", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")

    # Auto-detect tag
    try:
        tag = _git("describe", "--tags", "--exact-match", "HEAD")
    except subprocess.CalledProcessError:
        tag = None

    # Package version
    try:
        import bamengine

        version = getattr(bamengine, "__version__", None)
    except ImportError:
        version = None

    return {
        "commit": commit,
        "commit_short": commit_short,
        "branch": branch,
        "tag": tag,
        "version": version,
    }


# ---------------------------------------------------------------------------
# Chunk worker (runs in subprocess)
# ---------------------------------------------------------------------------


def _run_chunk(chunk_id: int, scenario: str, n_periods: int, n_seeds: int) -> dict:
    """Run a chunk of seeds for one scenario. Called in worker process."""
    import validation

    func = getattr(validation, SCENARIO_FUNCS[scenario])
    start = chunk_id * SEEDS_PER_CHUNK
    end = min((chunk_id + 1) * SEEDS_PER_CHUNK, n_seeds)
    seeds = list(range(start, end))
    result = func(seeds=seeds, n_periods=n_periods)

    # Return serializable data — full per-seed results for proper aggregation
    seed_data = []
    for seed_idx, vs in enumerate(result.seed_results):
        seed_data.append(
            {
                "seed": seeds[seed_idx],
                "passed": vs.passed,
                "total_score": vs.total_score,
                "metric_results": [
                    {
                        "name": mr.name,
                        "status": mr.status,
                        "actual": mr.actual,
                        "score": mr.score,
                        "weight": mr.weight,
                        "group": mr.group.name,
                    }
                    for mr in vs.metric_results
                ],
            }
        )

    return {
        "chunk_id": chunk_id,
        "n_passed": sum(1 for vs in result.seed_results if vs.passed),
        "n_seeds": len(seeds),
        "seed_data": seed_data,
    }


# ---------------------------------------------------------------------------
# Aggregation — compute stats from all seeds (not chunk averages)
# ---------------------------------------------------------------------------


def _aggregate_results(all_seed_data: list[dict]) -> dict:
    """Aggregate per-seed results into summary + per-metric stats.

    Computes statistics from the full seed set (not by averaging chunk-level
    stats, which would give incorrect standard deviations).
    """
    import numpy as np

    n_seeds = len(all_seed_data)
    scores = np.array([s["total_score"] for s in all_seed_data])
    n_passed = sum(1 for s in all_seed_data if s["passed"])

    summary = {
        "n_passed": n_passed,
        "pass_rate": n_passed / n_seeds,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
    }

    # Per-metric stats — collect values across all seeds
    metric_values: dict[str, dict] = {}
    for seed in all_seed_data:
        for mr in seed["metric_results"]:
            name = mr["name"]
            if name not in metric_values:
                metric_values[name] = {
                    "actuals": [],
                    "scores": [],
                    "passed": [],
                    "weight": mr["weight"],
                    "group": mr["group"],
                }
            metric_values[name]["actuals"].append(mr["actual"])
            metric_values[name]["scores"].append(mr["score"])
            metric_values[name]["passed"].append(mr["status"] != "FAIL")

    metrics = {}
    for name, vals in metric_values.items():
        actuals = np.array(vals["actuals"])
        metric_scores = np.array(vals["scores"])
        passed = np.array(vals["passed"])
        metrics[name] = {
            "mean_value": float(np.mean(actuals)),
            "std_value": float(np.std(actuals)),
            "mean_score": float(np.mean(metric_scores)),
            "std_score": float(np.std(metric_scores)),
            "pass_rate": float(np.mean(passed)),
            "weight": vals["weight"],
            "group": vals["group"],
        }

    # Failing seeds
    failing_seeds = []
    for seed in all_seed_data:
        if not seed["passed"]:
            failed_metrics = [
                mr["name"] for mr in seed["metric_results"] if mr["status"] == "FAIL"
            ]
            failing_seeds.append(
                {
                    "seed": seed["seed"],
                    "failed_metrics": failed_metrics,
                }
            )

    return {
        "summary": summary,
        "metrics": metrics,
        "failing_seeds": sorted(failing_seeds, key=lambda x: x["seed"]),
    }


# ---------------------------------------------------------------------------
# Scenario runner — parallel chunks, aggregation, JSON output
# ---------------------------------------------------------------------------


def run_scenario(
    scenario: str,
    metadata: dict,
    n_seeds: int = N_CHUNKS * SEEDS_PER_CHUNK,
    n_workers: int = N_WORKERS,
    n_periods: int = N_PERIODS,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Run 1000-seed stability benchmark for one scenario.

    Returns the result dict (also written to JSON file).
    """
    print(f"\n  {scenario}:")

    t0 = time.time()
    all_seed_data: list[dict] = []

    # Compute chunk count from n_seeds
    n_chunks = n_seeds // SEEDS_PER_CHUNK
    if n_seeds % SEEDS_PER_CHUNK != 0:
        n_chunks += 1  # last chunk clipped to n_seeds in _run_chunk

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_run_chunk, i, scenario, n_periods, n_seeds): i
            for i in range(n_chunks)
        }
        for future in as_completed(futures):
            chunk_id = futures[future]
            try:
                data = future.result()
            except Exception as e:
                print(f"    Chunk {chunk_id}: ERROR - {e}", file=sys.stderr)
                continue
            print(f"    Chunk {chunk_id}: {data['n_passed']}/{data['n_seeds']} passed")
            all_seed_data.extend(data["seed_data"])

    if not all_seed_data:
        print(f"    ERROR: No seed data collected for {scenario}", file=sys.stderr)
        return {}

    elapsed = time.time() - t0
    aggregated = _aggregate_results(all_seed_data)

    result = {
        "metadata": {
            **metadata,
            "n_seeds": len(all_seed_data),
            "n_periods": n_periods,
            "n_workers": n_workers,
            "elapsed_seconds": round(elapsed),
        },
        "scenario": scenario,
        "summary": aggregated["summary"],
        "metrics": aggregated["metrics"],
        "group_display_names": GROUP_DISPLAY_NAMES,
        "failing_seeds": aggregated["failing_seeds"],
    }

    # Print summary line
    sr = aggregated["summary"]
    n_passed = sr["n_passed"]
    mins, secs = divmod(int(elapsed), 60)
    print(
        f"    RESULT: {n_passed}/{len(all_seed_data)} passed ({sr['pass_rate']:.1%})"
        f" | mean_score={sr['mean_score']:.3f} | {mins}m {secs:02d}s"
    )

    # Write JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = metadata["timestamp"].replace(":", "").replace("-", "")[:15]
    filename = f"{scenario}_{metadata['commit_short']}_{ts}.json"
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    Saved: {filepath}")

    return result


# ---------------------------------------------------------------------------
# Commit resolution
# ---------------------------------------------------------------------------


def _resolve_commits(args) -> list[dict]:
    """Resolve --commits or --tags into a list of {hash, label} dicts.

    Returns empty list for current working tree mode.
    """
    if args.commits and args.tags:
        print("ERROR: --commits and --tags are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    if not args.commits and not args.tags:
        return []  # current working tree

    if args.commits:
        return _resolve_commit_specs(args.commits, args.force)
    else:
        return _resolve_tag_specs(args.tags, args.force)


def _resolve_commit_specs(specs: list[str], force: bool) -> list[dict]:
    """Resolve commit specifications into {hash, label} list."""
    commits = []
    for spec in specs:
        if ".." in spec:
            # Range: X..Y
            start, end = spec.split("..", 1)
            hashes = _git(
                "rev-list", "--ancestry-path", f"{start}^..{end}"
            ).splitlines()
            for h in reversed(hashes):  # chronological order
                short = _git("rev-parse", "--short", h)
                commits.append({"hash": h, "label": short})
        else:
            # Single commit
            full = _git("rev-parse", spec)
            short = _git("rev-parse", "--short", spec)
            commits.append({"hash": full, "label": short})

    if len(commits) > MAX_COMMITS_WITHOUT_FORCE and not force:
        print(f"ERROR: {len(commits)} commits resolved. Use --force to proceed.")
        print("Commits:")
        for c in commits:
            print(f"  {c['label']}")
        sys.exit(1)

    return commits


def _resolve_tag_specs(specs: list[str], force: bool) -> list[dict]:
    """Resolve tag specifications into {hash, label} list."""
    commits = []
    for spec in specs:
        if ".." in spec:
            # Tag range: vX..vY
            start_tag, end_tag = spec.split("..", 1)
            # Get all tags sorted by version
            all_tags = _git(
                "tag", "--sort=version:refname", "--list", "v*"
            ).splitlines()
            in_range = False
            found_end = False
            for tag in all_tags:
                if tag == start_tag:
                    in_range = True
                if in_range:
                    full = _git("rev-parse", tag)
                    commits.append({"hash": full, "label": tag})
                if tag == end_tag:
                    found_end = True
                    break
            if not in_range:
                print(f"ERROR: start tag '{start_tag}' not found", file=sys.stderr)
                sys.exit(1)
            if not found_end:
                print(f"ERROR: end tag '{end_tag}' not found", file=sys.stderr)
                sys.exit(1)
        else:
            # Single tag
            full = _git("rev-parse", spec)
            commits.append({"hash": full, "label": spec})

    if len(commits) > MAX_COMMITS_WITHOUT_FORCE and not force:
        print(f"ERROR: {len(commits)} commits resolved. Use --force to proceed.")
        print("Tags:")
        for c in commits:
            print(f"  {c['label']}")
        sys.exit(1)

    return commits


# ---------------------------------------------------------------------------
# Worktree management for historical commits
# ---------------------------------------------------------------------------

_active_worktrees: list[str] = []


def _cleanup_worktrees():
    """Remove any active worktrees on exit."""
    for wt in _active_worktrees:
        with contextlib.suppress(Exception):
            subprocess.run(
                ["git", "worktree", "remove", "--force", wt],
                capture_output=True,
            )
    with contextlib.suppress(Exception):
        subprocess.run(["git", "worktree", "prune"], capture_output=True)


atexit.register(_cleanup_worktrees)


def _run_with_worktree(
    commit: dict,
    scenarios: list[str],
    n_seeds: int,
    n_workers: int,
    n_periods: int,
    output_dir: Path,
) -> list[dict]:
    """Run benchmarks against a historical commit using a git worktree.

    Creates a temporary worktree, symlinks current validation + extensions
    into it, and runs benchmarks with PYTHONPATH pointing to the worktree.
    """
    wt_dir = Path(f".worktrees/stability-{commit['label']}")
    current_src = Path("src").resolve()

    # Save state BEFORE try block to avoid UnboundLocalError in finally
    old_path = sys.path[:]
    old_pythonpath = os.environ.get("PYTHONPATH")

    try:
        # Create worktree
        _git("worktree", "add", str(wt_dir), commit["hash"])
        _active_worktrees.append(str(wt_dir))

        wt_src = wt_dir / "src"

        # Replace validation and extensions with symlinks to current
        for pkg in ("validation", "extensions"):
            wt_pkg = wt_src / pkg
            if wt_pkg.is_symlink():
                wt_pkg.unlink()
            elif wt_pkg.exists():
                shutil.rmtree(wt_pkg)
            os.symlink(current_src / pkg, wt_pkg)

        # Get metadata for this commit
        commit_short = _git("rev-parse", "--short", commit["hash"])
        try:
            tag = _git("describe", "--tags", "--exact-match", commit["hash"])
        except subprocess.CalledProcessError:
            tag = None

        # Try to get version from the worktree's bamengine
        version = None
        try:
            wt_init = wt_src / "bamengine" / "__init__.py"
            if wt_init.exists():
                import ast

                tree = ast.parse(wt_init.read_text())
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Assign)
                        and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == "__version__"
                        and isinstance(node.value, ast.Constant)
                    ):
                        version = node.value.value
                        break
        except Exception:
            pass

        metadata = {
            "commit": commit["hash"],
            "commit_short": commit_short,
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
            "branch": "main",
            "tag": tag,
            "version": version,
        }

        # Set PYTHONPATH to worktree src for spawned worker processes
        # Workers inherit os.environ["PYTHONPATH"] on spawn (macOS default)
        wt_src_str = str(wt_src)

        # Prepend worktree src to sys.path
        sys.path.insert(0, wt_src_str)
        os.environ["PYTHONPATH"] = wt_src_str

        # Clear cached modules so imports resolve from worktree
        mods_to_remove = [
            m
            for m in sys.modules
            if m.startswith(("bamengine", "validation", "extensions", "calibration"))
        ]
        for m in mods_to_remove:
            del sys.modules[m]

        results = []
        for scenario in scenarios:
            try:
                result = run_scenario(
                    scenario,
                    metadata,
                    n_seeds,
                    n_workers,
                    n_periods,
                    output_dir,
                )
                if result:
                    results.append(result)
            except Exception as e:
                print(
                    f"    ERROR ({scenario}): {e}",
                    file=sys.stderr,
                )
                continue

        return results

    finally:
        # Restore sys.path and modules
        sys.path[:] = old_path
        if old_pythonpath is not None:
            os.environ["PYTHONPATH"] = old_pythonpath
        elif "PYTHONPATH" in os.environ:
            del os.environ["PYTHONPATH"]

        # Clear worktree modules
        mods_to_remove = [
            m
            for m in sys.modules
            if m.startswith(("bamengine", "validation", "extensions", "calibration"))
        ]
        for m in mods_to_remove:
            del sys.modules[m]

        # Remove worktree
        with contextlib.suppress(Exception):
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(wt_dir)],
                capture_output=True,
            )
        if str(wt_dir) in _active_worktrees:
            _active_worktrees.remove(str(wt_dir))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="1000-seed stability benchmarking for validation scenarios"
    )
    parser.add_argument(
        "--scenario",
        choices=ALL_SCENARIOS,
        help="Run a single scenario (default: all three)",
    )
    parser.add_argument(
        "--commits",
        nargs="+",
        help="Specific commits or range (X..Y) to benchmark",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Specific tags or range (vX..vY) to benchmark",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=N_CHUNKS * SEEDS_PER_CHUNK,
        help=f"Total number of seeds (default: {N_CHUNKS * SEEDS_PER_CHUNK})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=N_WORKERS,
        help=f"Number of parallel workers (default: {N_WORKERS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow more than 20 commits without confirmation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )

    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else ALL_SCENARIOS
    output_dir = OUTPUT_DIR
    n_workers = args.workers

    # Resolve commits
    commits = _resolve_commits(args)

    if args.dry_run:
        _print_dry_run(commits, scenarios)
        return

    if commits:
        # Historical mode
        print(
            f"Stability Benchmark: {len(scenarios)} scenarios x "
            f"{args.seeds} seeds x {n_workers} workers x "
            f"{len(commits)} commits"
        )
        for commit in commits:
            print(f"\n{'=' * 70}")
            print(f"Commit: {commit['label']}")
            print(f"{'=' * 70}")
            _run_with_worktree(
                commit, scenarios, args.seeds, n_workers, N_PERIODS, output_dir
            )
    else:
        # Current working tree
        metadata = _get_git_metadata()
        metadata["timestamp"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")

        tag_str = f" ({metadata['tag']})" if metadata.get("tag") else ""
        print(
            f"Stability Benchmark: {len(scenarios)} scenarios x "
            f"{args.seeds} seeds x {n_workers} workers"
        )
        print(f"Commit: {metadata['commit_short']}{tag_str}")

        for scenario in scenarios:
            try:
                run_scenario(
                    scenario, metadata, args.seeds, n_workers, N_PERIODS, output_dir
                )
            except Exception as e:
                print(f"    ERROR ({scenario}): {e}", file=sys.stderr)
                continue

    print(f"\nResults saved to {output_dir}/")


def _print_dry_run(commits: list[dict], scenarios: list[str]) -> None:
    """Print what would be run without executing."""
    if not commits:
        metadata = _get_git_metadata()
        tag_str = f" ({metadata['tag']})" if metadata.get("tag") else ""
        print(
            f"Would benchmark current working tree: {metadata['commit_short']}{tag_str}"
        )
        n_runs = 1
    else:
        print(f"Would benchmark {len(commits)} commits:")
        for c in commits:
            print(f"  {c['label']}")
        n_runs = len(commits)

    print(f"\nScenarios: {', '.join(scenarios)}")
    est_minutes = n_runs * len(scenarios) * 7
    print(f"Estimated time: ~{est_minutes} minutes")


if __name__ == "__main__":
    main()
