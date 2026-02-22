"""Structural experiments (Section 3.10.2).

Implements two structural experiments that test model mechanisms:

1. **Preferential Attachment (PA) experiment**: Disable consumer loyalty
   and show that volatility drops and deep crises vanish. Then sweep Z
   with PA off.

2. **Entry neutrality experiment**: Apply heavy profit taxation without
   redistribution to increase bankruptcies, confirming that the automatic
   firm entry mechanism does NOT artificially drive recovery.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from validation.robustness.internal_validity import (
    InternalValidityResult,
    run_internal_validity,
)
from validation.robustness.sensitivity import (
    SensitivityResult,
    run_sensitivity_analysis,
)


@dataclass
class PAExperimentResult:
    """Result of the preferential attachment experiment (Section 3.10.2).

    Attributes
    ----------
    pa_off_validity : InternalValidityResult
        Internal validity with PA disabled (consumer_matching="random").
    pa_off_z_sweep : SensitivityResult
        Z-sweep sensitivity with PA disabled.
    baseline_validity : InternalValidityResult or None
        Optional baseline (PA on) for comparison.
    """

    pa_off_validity: InternalValidityResult
    pa_off_z_sweep: SensitivityResult
    baseline_validity: InternalValidityResult | None = None


@dataclass
class EntryExperimentResult:
    """Result of the entry neutrality experiment (Section 3.10.2).

    Attributes
    ----------
    tax_sweep : SensitivityResult
        Sensitivity analysis sweeping profit_tax_rate.
    """

    tax_sweep: SensitivityResult


def run_pa_experiment(
    n_seeds: int = 20,
    n_periods: int = 1000,
    burn_in: int = 500,
    n_workers: int = 10,
    verbose: bool = True,
    include_baseline: bool = True,
    setup_hook: Callable[..., None] | None = None,
    collect_config: dict[str, Any] | None = None,
    **config_overrides: Any,
) -> PAExperimentResult:
    """Run the preferential attachment experiment (Section 3.10.2).

    Phase 1: Run internal validity with PA disabled to show volatility
    drops and deep crises vanish.

    Phase 2: Run Z-sweep sensitivity with PA disabled.

    Optionally runs baseline (PA on) for comparison.

    Parameters
    ----------
    n_seeds : int
        Number of random seeds.
    n_periods : int
        Simulation periods per seed.
    burn_in : int
        Burn-in periods to discard.
    n_workers : int
        Parallel workers.
    verbose : bool
        Print progress.
    include_baseline : bool
        Also run baseline (PA on) for comparison.
    setup_hook : callable or None
        Global setup hook (e.g. Growth+ R&D).
    collect_config : dict or None
        Custom collection config.
    **config_overrides
        Additional config overrides.

    Returns
    -------
    PAExperimentResult
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  PA EXPERIMENT (Section 3.10.2)")
        print("=" * 60)

    # Phase 1: Internal validity with PA off
    if verbose:
        print("\n--- Phase 1: Internal validity (PA off) ---")
    pa_off_validity = run_internal_validity(
        n_seeds=n_seeds,
        n_periods=n_periods,
        burn_in=burn_in,
        n_workers=n_workers,
        verbose=verbose,
        setup_hook=setup_hook,
        collect_config=collect_config,
        consumer_matching="random",
        **config_overrides,
    )

    # Phase 2: Z-sweep with PA off
    if verbose:
        print("\n--- Phase 2: Z-sweep sensitivity (PA off) ---")
    pa_off_z_sweep = run_sensitivity_analysis(
        experiments=["goods_market_no_pa"],
        n_seeds=n_seeds,
        n_periods=n_periods,
        burn_in=burn_in,
        n_workers=n_workers,
        verbose=verbose,
        setup_hook=setup_hook,
        collect_config=collect_config,
        **config_overrides,
    )

    # Optional baseline for comparison
    baseline_validity = None
    if include_baseline:
        if verbose:
            print("\n--- Baseline: Internal validity (PA on) ---")
        baseline_validity = run_internal_validity(
            n_seeds=n_seeds,
            n_periods=n_periods,
            burn_in=burn_in,
            n_workers=n_workers,
            verbose=verbose,
            setup_hook=setup_hook,
            collect_config=collect_config,
            **config_overrides,
        )

    return PAExperimentResult(
        pa_off_validity=pa_off_validity,
        pa_off_z_sweep=pa_off_z_sweep,
        baseline_validity=baseline_validity,
    )


def run_entry_experiment(
    n_seeds: int = 20,
    n_periods: int = 1000,
    burn_in: int = 500,
    n_workers: int = 10,
    verbose: bool = True,
    setup_hook: Callable[..., None] | None = None,
    collect_config: dict[str, Any] | None = None,
    **config_overrides: Any,
) -> EntryExperimentResult:
    """Run the entry neutrality experiment (Section 3.10.2).

    Sweeps profit_tax_rate from 0% to 90% to show monotonic degradation,
    confirming that the automatic firm entry mechanism does NOT
    artificially drive recovery.

    Parameters
    ----------
    n_seeds : int
        Number of random seeds.
    n_periods : int
        Simulation periods per seed.
    burn_in : int
        Burn-in periods to discard.
    n_workers : int
        Parallel workers.
    verbose : bool
        Print progress.
    setup_hook : callable or None
        Global setup hook (e.g. Growth+ R&D).
    collect_config : dict or None
        Custom collection config.
    **config_overrides
        Additional config overrides.

    Returns
    -------
    EntryExperimentResult
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  ENTRY NEUTRALITY EXPERIMENT (Section 3.10.2)")
        print("=" * 60)

    tax_sweep = run_sensitivity_analysis(
        experiments=["entry_neutrality"],
        n_seeds=n_seeds,
        n_periods=n_periods,
        burn_in=burn_in,
        n_workers=n_workers,
        verbose=verbose,
        setup_hook=setup_hook,
        collect_config=collect_config,
        **config_overrides,
    )

    return EntryExperimentResult(tax_sweep=tax_sweep)
