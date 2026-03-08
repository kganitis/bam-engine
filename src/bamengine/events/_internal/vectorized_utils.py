"""
Vectorized utility functions for batch market matching.

These helpers underpin the vectorized market events, replacing Python
for-loops with pure NumPy operations.  They are used by
``vectorized_markets.py`` but kept separate for testability.

Three core primitives
---------------------
1. **grouped_cumsum** — per-group prefix sums via the subtract-offset trick.
2. **resolve_conflicts** — batch conflict resolution: when multiple senders
   target the same receiver, randomly accept up to *capacity* per target.
3. **pro_rata_ration** — proportional rationing: when aggregate demand on a
   target exceeds supply, each buyer's allocation is scaled down pro-rata.
"""

from __future__ import annotations

import numpy as np

from bamengine import Rng
from bamengine.typing import Bool1D, Float1D, Idx1D, Int1D

# ── 1. grouped_cumsum ────────────────────────────────────────────────────────


def grouped_cumsum(values: Float1D, group_starts: Int1D) -> Float1D:
    """Per-group prefix sums using the subtract-offset trick.

    Given a flat array *values* that is logically partitioned into contiguous
    groups whose boundaries are indicated by *group_starts*, return an array
    of the same length where each element is the cumulative sum within its
    group.

    Parameters
    ----------
    values : Float1D
        Values to accumulate (length *n*).
    group_starts : Int1D
        Sorted indices into *values* where each new group begins.
        ``group_starts[0]`` is typically 0.

    Returns
    -------
    Float1D
        Per-group cumulative sums, same shape as *values*.

    Examples
    --------
    >>> import numpy as np
    >>> vals = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    >>> starts = np.array([0, 3])  # two groups: [0:3], [3:5]
    >>> grouped_cumsum(vals, starts)
    array([ 1.,  3.,  6., 10., 30.])
    """
    if values.size == 0:
        return np.empty(0, dtype=np.float64)

    cs = np.cumsum(values)

    if group_starts.size <= 1:
        # Single group (or no groups) — global cumsum is correct.
        return cs

    # Map each position to its group, then subtract the cumsum value
    # just before that group's start.
    positions = np.arange(values.size)
    group_ids = np.searchsorted(group_starts, positions, side="right") - 1

    # Per-group offset: cs[group_starts[g] - 1] for g > 0, else 0
    group_offsets = np.zeros(group_starts.size, dtype=np.float64)
    inner_starts = group_starts[1:]
    valid = inner_starts < values.size
    if valid.any():
        group_offsets[1:][valid] = cs[inner_starts[valid] - 1]

    return cs - group_offsets[group_ids]


# ── 2. resolve_conflicts ─────────────────────────────────────────────────────


def resolve_conflicts(
    sender_ids: Idx1D,
    target_ids: Idx1D,
    capacity_per_target: Int1D,
    n_targets: int,
    rng: Rng,
) -> Bool1D:
    """Batch conflict resolution for oversubscribed targets.

    When multiple senders choose the same target, randomly accept up to
    ``capacity_per_target[t]`` senders for each target *t*.

    Parameters
    ----------
    sender_ids : Idx1D
        Sender indices (length *m*).  Used only for output alignment; the
        returned mask is ordered the same as ``sender_ids``.
    target_ids : Idx1D
        Which target each sender chose (length *m*).
    capacity_per_target : Int1D
        Maximum number of accepted senders per target (length *n_targets*).
    n_targets : int
        Total number of possible targets.
    rng : Rng
        NumPy random generator for tie-breaking.

    Returns
    -------
    Bool1D
        Boolean mask of length *m*; ``True`` ⇒ sender is accepted.

    Notes
    -----
    The algorithm:

    1. Sort senders by target (``np.argsort``).
    2. Find group boundaries per target (``np.searchsorted``).
    3. Within each over-subscribed group, randomly select *capacity* senders.

    ``np.add.at`` is not needed here — we only read capacities.
    """
    m = sender_ids.size
    if m == 0:
        return np.empty(0, dtype=np.bool_)

    # Sort senders by target
    order = np.argsort(target_ids, kind="stable")
    sorted_targets = target_ids[order]

    # Group boundaries via bincount → cumsum
    counts = np.bincount(sorted_targets, minlength=n_targets)
    boundaries = np.empty(n_targets + 1, dtype=np.intp)
    boundaries[0] = 0
    np.cumsum(counts, out=boundaries[1:])

    accepted_in_sorted = np.zeros(m, dtype=np.bool_)

    # Process targets that have at least one sender
    active_targets = np.where(counts > 0)[0]
    for t in active_targets:
        lo, hi = boundaries[t], boundaries[t + 1]
        group_size = hi - lo
        cap = int(capacity_per_target[t])
        if cap <= 0:
            continue
        if group_size <= cap:
            # All accepted
            accepted_in_sorted[lo:hi] = True
        else:
            # Randomly pick `cap` from the group
            chosen = rng.choice(group_size, size=cap, replace=False)
            accepted_in_sorted[lo + chosen] = True

    # Un-sort back to original order
    accepted = np.empty(m, dtype=np.bool_)
    accepted[order] = accepted_in_sorted
    return accepted


# ── 3. pro_rata_ration ────────────────────────────────────────────────────────


def pro_rata_ration(
    qty_wanted: Float1D,
    supply_per_target: Float1D,
    target_ids: Idx1D,
    n_targets: int,
) -> Float1D:
    """Proportional rationing when aggregate demand exceeds supply.

    For each target, if total demand from all buyers exceeds supply,
    each buyer's allocation is scaled proportionally.

    Parameters
    ----------
    qty_wanted : Float1D
        Desired quantity per buyer (length *m*).
    supply_per_target : Float1D
        Available supply at each target (length *n_targets*).
    target_ids : Idx1D
        Which target each buyer is visiting (length *m*).
    n_targets : int
        Total number of possible targets.

    Returns
    -------
    Float1D
        Actual quantity allocated to each buyer (length *m*),
        never exceeding ``qty_wanted`` or the buyer's share of supply.

    Notes
    -----
    The algorithm:

    1. Aggregate total demand per target using ``np.add.at``.
    2. Compute ratio = supply / total_demand per target (capped at 1.0).
    3. Scale each buyer's quantity by the ratio of their target.
    """
    m = qty_wanted.size
    if m == 0:
        return np.empty(0, dtype=np.float64)

    # Aggregate demand per target
    total_demand = np.zeros(n_targets, dtype=np.float64)
    np.add.at(total_demand, target_ids, qty_wanted)

    # Compute rationing ratio (supply / demand), capped at 1.0
    # Safe division: where demand is 0, ratio is 1.0 (no buyers to ration)
    safe_demand = np.where(total_demand > 0.0, total_demand, 1.0)
    ratio = np.minimum(supply_per_target / safe_demand, 1.0)

    # Scale each buyer's quantity by target's ratio
    qty_actual = qty_wanted * ratio[target_ids]
    return qty_actual
