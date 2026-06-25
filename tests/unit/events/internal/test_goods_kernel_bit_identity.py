"""
Bit-identity battery: Numba kernel vs. Python kernel across seeds x sizes.

For each (seed, n_firms) combination, two identical simulations are run:
one with ``goods_kernel="python"`` and one with ``goods_kernel="numba"``.
Every collected per-agent and economy array must be EXACTLY equal
(``np.array_equal`` / bit-for-bit).

The test is skipped when Numba is not installed.  When Numba IS installed
this is the load-bearing correctness proof that the ``@njit`` kernel
replicates the Python loop bit-for-bit.

Sizes:
  n_firms in {100, 300, 1000} with 5:1 household ratio and n_firms//10 banks
Seeds: {0, 1, 2, 3, 4}
"""

from __future__ import annotations

import numpy as np
import pytest

import bamengine as bam
from bamengine.events._internal.goods_market import HAS_NUMBA
from bamengine.simulation import Simulation

# Seeds and sizes to cover the battery
_SEEDS = [0, 1, 2, 3, 4]
_N_FIRMS = [100, 300, 1000]
_N_PERIODS = 30


def _run_full(n_firms: int, seed: int, kernel: str) -> bam.SimulationResults:
    """Run a simulation and return the full results object."""
    n_hh = n_firms * 5
    n_banks = max(1, n_firms // 10)
    sim = Simulation.init(
        n_firms=n_firms,
        n_households=n_hh,
        n_banks=n_banks,
        seed=seed,
        log_level="WARNING",
        goods_kernel=kernel,
    )
    return sim.run(n_periods=_N_PERIODS)


def _all_keys(results: bam.SimulationResults) -> list[str]:
    """Return all collected data keys (per-agent + economy)."""
    return list(results.available())


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("n_firms", _N_FIRMS)
def test_numba_kernel_bit_identical_to_python(n_firms: int, seed: int) -> None:
    """Numba and Python kernels produce bit-identical results for all seeds x sizes.

    This is the primary correctness proof for the ``@njit`` goods-market kernel.
    Any divergence indicates an arithmetic or operation-order difference in the
    kernel that must be fixed (relaxing to ``allclose`` is NOT acceptable).
    """
    res_py = _run_full(n_firms, seed, "python")
    res_nb = _run_full(n_firms, seed, "numba")

    keys_py = _all_keys(res_py)
    keys_nb = _all_keys(res_nb)

    # Both runs must collect the same keys
    assert set(keys_py) == set(keys_nb), (
        f"Collected keys differ: py={set(keys_py) - set(keys_nb)}, "
        f"nb={set(keys_nb) - set(keys_py)}"
    )

    # Every array must be bit-identical
    mismatches: list[str] = []
    for key in keys_py:
        arr_py = res_py[key]
        arr_nb = res_nb[key]
        if not np.array_equal(arr_py, arr_nb):
            # Report first few differing values for diagnosis
            mismatches.append(f"  [{key}]: py={arr_py.flat[:3]}, nb={arr_nb.flat[:3]}")

    assert not mismatches, (
        f"Bit-identity FAILED for n_firms={n_firms}, seed={seed}. "
        f"Mismatching arrays ({len(mismatches)}):\n" + "\n".join(mismatches)
    )
