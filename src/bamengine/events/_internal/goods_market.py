"""
System functions for goods market phase events.

This module contains the internal implementation functions for goods market events.
Event classes wrap these functions and provide the primary documentation.

See Also
--------
bamengine.events.goods_market : Event classes (primary documentation source)
"""

from __future__ import annotations

import numpy as np

from bamengine import Rng, logging, make_rng
from bamengine.roles import Consumer, Producer
from bamengine.utils import EPS, sample_k_per_row

# Optional Numba dependency: present when the user installs bamengine[fast].
try:
    import numba

    HAS_NUMBA = True
except ImportError:
    numba = None
    HAS_NUMBA = False

log = logging.getLogger(__name__)


def consumers_calc_propensity(
    con: Consumer,
    *,
    avg_sav: float,
    beta: float,
) -> None:
    """
    Calculate marginal propensity to consume based on relative savings.

    See Also
    --------
    bamengine.events.goods_market.ConsumersCalcPropensity : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Calculating Consumer Spending Propensity ---")
        log.info(f"  Inputs: Average Savings={avg_sav:.3f} | β={beta:.3f}")

    # Defensive operations to ensure valid calculations
    initial_negative_savings = np.sum(con.savings < EPS)
    if initial_negative_savings > 0:
        log.warning(
            f"  Found {initial_negative_savings} consumers with negative savings. "
            f"Clamping to 0.0."
        )

    np.maximum(con.savings, 0.0, out=con.savings)  # defensive clamp
    avg_sav = max(avg_sav, EPS)  # avoid division by zero

    # Core calculation
    savings_ratio = con.savings / avg_sav
    t = np.tanh(savings_ratio)  # ∈ [0, 1]
    con.propensity[:] = 1.0 / (1.0 + t**beta)

    # Summary statistics
    if info_enabled:
        min_propensity = con.propensity.min()
        max_propensity = con.propensity.max()
        avg_propensity = con.propensity.mean()

        log.info(f"  Propensity calculated for {con.propensity.size:,} consumers.")
        log.info(
            f"  Propensity range: [{min_propensity:.3f}, {max_propensity:.3f}], "
            f"Average: {avg_propensity:.3f}"
        )

    if log.isEnabledFor(logging.DEBUG):
        high_spenders = np.sum(con.propensity > 0.8)
        low_spenders = np.sum(con.propensity < 0.2)
        log.debug(
            f"  High spenders (>0.8): {high_spenders}, "
            f"Low spenders (<0.2): {low_spenders}"
        )
        log.debug(
            f"  First 10 propensities: "
            f"{np.array2string(con.propensity[:10], precision=3)}"
        )

    if info_enabled:
        log.info("--- Consumer Spending Propensity Calculation complete ---")


def consumers_decide_income_to_spend(con: Consumer) -> None:
    """
    Allocate wealth to spending budget based on propensity to consume.

    See Also
    --------
    bamengine.events.goods_market.ConsumersDecideIncomeToSpend : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Consumers Deciding Income to Spend ---")

        # Pre-calculation statistics
        total_initial_savings = con.savings.sum()
        total_income = con.income.sum()
        total_wealth = total_initial_savings + total_income
        avg_propensity = con.propensity.mean()

        log.info(
            f"  Initial state: Total Savings={total_initial_savings:,.2f}, "
            f"Total Income={total_income:,.2f}, Total Wealth={total_wealth:,.2f}"
        )
        log.info(f"  Average propensity to spend: {avg_propensity:.3f}")

    # Core calculation
    wealth = con.savings + con.income
    con.income_to_spend[:] = wealth * con.propensity
    con.savings[:] = wealth - con.income_to_spend
    con.income[:] = 0.0  # zero-out disposable income after allocation

    # Post-calculation statistics
    if info_enabled:
        total_spending_budget = con.income_to_spend.sum()
        total_final_savings = con.savings.sum()
        consumers_with_budget = np.sum(con.income_to_spend > EPS)

        log.info(
            f"  Spending decisions made for {con.income_to_spend.size:,} consumers."
        )
        log.info(f"  Total spending budget allocated: {total_spending_budget:,.2f}")
        log.info(f"  Total remaining savings: {total_final_savings:,.2f}")
        log.info(
            f"  Consumers with positive spending budget: {consumers_with_budget:,}"
        )

    if log.isEnabledFor(logging.DEBUG):
        consumers_with_budget = np.sum(con.income_to_spend > EPS)
        max_budget = con.income_to_spend.max()
        avg_budget = (
            con.income_to_spend[con.income_to_spend > 0].mean()
            if consumers_with_budget > 0
            else 0.0
        )
        log.debug(
            f"  Spending budget stats - Max: {max_budget:.2f}, "
            f"Avg (of spenders): {avg_budget:.2f}"
        )
        log.debug(
            f"  First 10 spending budgets: "
            f"{np.array2string(con.income_to_spend[:10], precision=2)}"
        )

        # Sanity check: wealth should be conserved
        total_spending_budget = con.income_to_spend.sum()
        total_final_savings = con.savings.sum()
        total_wealth = con.savings.sum() + con.income_to_spend.sum()
        wealth_check = total_spending_budget + total_final_savings
        if abs(wealth_check - total_wealth) > EPS:
            log.error(
                f"  WEALTH CONSERVATION ERROR: "
                f"Expected {total_wealth:.2f}, Got {wealth_check:.2f}"
            )

    if info_enabled:
        log.info("--- Consumer Income-to-Spend Decision complete ---")


def consumers_decide_firms_to_visit(
    con: Consumer,
    prod: Producer,
    *,
    max_Z: int,
    rng: Rng = make_rng(),
    consumer_matching: str = "loyalty",
) -> None:
    """
    Consumers select firms to visit and set loyalty BEFORE shopping.

    The loyalty (largest_prod_prev) is updated here based on the largest
    producer in the consideration set, BEFORE any shopping occurs.

    See Also
    --------
    bamengine.events.goods_market.ConsumersDecideFirmsToVisit : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Consumers Deciding Firms to Visit ---")

    n_firms = prod.inventory.size
    stride = max_Z

    # Initialize/flush all shopping queues
    con.shop_visits_targets.fill(-1)
    con.shop_visits_head.fill(-1)

    if n_firms == 0:
        if info_enabled:
            log.info("  No firms available. All shopping queues cleared.")
            log.info("--- Consumer Firm Selection complete ---")
        return

    # Identify consumers with budget (vectorized)
    has_budget = con.income_to_spend > EPS
    budget_indices = np.where(has_budget)[0]
    n_active = budget_indices.size

    if info_enabled:
        log.info(
            f"  {n_active:,} consumers with spending budget will select"
            f" up to {max_Z} firms each from {n_firms} firms."
        )

    if n_active == 0:
        if info_enabled:
            log.info(
                "  No consumers have spending budget. All shopping queues cleared."
            )
            log.info("--- Consumer Firm Selection complete ---")
        return

    # Get loyalty firms for active consumers
    loyalty_firms = con.largest_prod_prev[budget_indices]
    has_loyalty = loyalty_firms >= 0

    effective_Z = min(max_Z, n_firms)
    # Forced loyalty inclusion (replaces the priority=1.1 trick), distributionally identical.
    if consumer_matching == "loyalty":
        forced = np.where(has_loyalty, loyalty_firms.astype(np.intp), -1).astype(
            np.intp
        )
    else:
        forced = None
    top_k_indices = sample_k_per_row(rng, n_active, n_firms, effective_Z, forced=forced)

    # Sort selected firms by price (cheapest first) - vectorized
    prices_selected = prod.price[top_k_indices]
    if effective_Z == 2:
        # argsort over 2 columns returns [0,1] when prices_selected[:,0] <= [:,1]
        # (ties -> [0,1], matching numpy's argsort stable behaviour), else [1,0].
        # Build that directly without the O(n log n) overhead of np.argsort.
        swap = prices_selected[:, 0] > prices_selected[:, 1]
        price_order = np.empty((prices_selected.shape[0], 2), dtype=np.intp)
        price_order[:, 0] = swap  # 0 if no swap, 1 if swap
        price_order[:, 1] = ~swap  # 1 if no swap, 0 if swap
    else:
        price_order = np.argsort(prices_selected, axis=1)
    sorted_firms = np.take_along_axis(top_k_indices, price_order, axis=1)

    # Vectorized buffer write — fancy indexing replaces per-consumer loop
    con.shop_visits_targets[budget_indices, :effective_Z] = sorted_firms
    con.shop_visits_head[budget_indices] = budget_indices * stride

    # Update loyalty to largest producer in consideration set (vectorized)
    if consumer_matching == "loyalty":
        # For each consumer, find the firm with max production among selected firms
        production_selected = prod.production[sorted_firms]
        largest_local_idx = np.argmax(production_selected, axis=1)
        largest_firm_ids = sorted_firms[np.arange(n_active), largest_local_idx]
        con.largest_prod_prev[budget_indices] = largest_firm_ids

    # Compute statistics for logging
    loyalty_applied = has_loyalty.sum()
    total_selections_made = n_active * effective_Z
    loyalty_updates = n_active

    if info_enabled:
        avg_selections = effective_Z
        loyalty_rate = loyalty_applied / n_active if n_active > 0 else 0.0

        log.info(f"  Firm selection completed for {n_active:,} consumers with budget.")
        log.info(
            f"  Total firm selections made: {total_selections_made:,} "
            f"(Average: {avg_selections:.1f} per consumer)"
        )
        log.info(
            f"  Loyalty rule applied: "
            f"{loyalty_applied:,} times ({loyalty_rate:.1%} of consumers)"
        )
        log.info(
            f"  Loyalty updated (pre-shopping): "
            f"{loyalty_updates:,} consumers set loyalty to largest in consideration set"
        )

    debug_enabled = log.isEnabledFor(logging.DEBUG)
    if debug_enabled:
        active_shoppers = np.sum(con.shop_visits_head >= 0)
        log.debug(f"  Active shoppers with queued visits: {active_shoppers:,}")

        # Check firm popularity
        firm_selection_counts = np.bincount(
            con.shop_visits_targets[con.shop_visits_targets >= 0],
            minlength=n_firms,
        )
        most_popular_firm = np.argmax(firm_selection_counts)
        max_selections = firm_selection_counts[most_popular_firm]
        log.debug(
            f"  Most popular firm: {most_popular_firm} "
            f"(selected by {max_selections} consumers)"
        )

    if info_enabled:
        log.info("--- Consumer Firm Selection complete ---")


def consumers_finalize_purchases(con: Consumer) -> None:
    """
    Return unspent budget to savings after shopping rounds complete.

    See Also
    --------
    bamengine.events.goods_market.ConsumersFinalizePurchases : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Finalizing Consumer Purchases ---")

        # Pre-finalization statistics
        total_unspent = con.income_to_spend.sum()
        total_savings_before = con.savings.sum()
        consumers_with_unspent = np.sum(con.income_to_spend > EPS)

        log.info(
            f"  {consumers_with_unspent:,} consumers have unspent budget "
            f"totaling {total_unspent:,.2f}"
        )
        log.info(f"  Current total savings: {total_savings_before:,.2f}")

    # Core operation: move unspent budget to savings
    np.add(con.savings, con.income_to_spend, out=con.savings)
    con.income_to_spend.fill(0.0)

    # Post-finalization statistics
    if info_enabled:
        total_savings_after = con.savings.sum()
        # Note: total_savings_before and total_unspent computed above under info_enabled
        savings_increase = total_savings_after - total_savings_before

        log.info(
            f"  Unspent budget moved to savings. "
            f"New total savings: {total_savings_after:,.2f}"
        )
        log.info(f"  Savings increase: {savings_increase:,.2f}")

    if log.isEnabledFor(logging.DEBUG):
        avg_savings = con.savings.mean()
        max_savings = con.savings.max()
        consumers_with_savings = np.sum(con.savings > 0.0)

        log.debug(
            f"  Final savings stats - Average: {avg_savings:.2f}, "
            f"Maximum: {max_savings:.2f}"
        )
        log.debug(f"  Consumers with positive savings: {consumers_with_savings:,}")

        # Wealth conservation check (only if info was enabled and we have the values)
        if info_enabled:
            if abs(savings_increase - total_unspent) > EPS:
                log.error(
                    f"  WEALTH CONSERVATION ERROR: Expected savings increase of "
                    f"{total_unspent:.2f}, got {savings_increase:.2f}"
                )
            else:
                log.debug(
                    "  Wealth conservation verified: unspent budget properly saved"
                )

    if info_enabled:
        log.info("--- Purchase Finalization complete ---")


def _goods_buy_loop_py(
    buyer_order: np.ndarray,
    targets: list[list[int]],
    prices: list[float],
    inv: list[float],
    budget: list[float],
) -> None:
    """Pure-Python sequential buying loop (array-only signature).

    This is the extracted, dispatchable inner loop of the goods market.
    It is called by ``goods_market_round`` for the ``"python"`` kernel and as
    the fallback for ``"auto"`` when Numba is not installed.  The Numba
    ``@njit`` kernel (Task 2) will replicate this logic exactly.

    All randomness (buyer shuffle) is resolved BEFORE this function is
    called; the loop body is deterministic given its inputs.

    Parameters
    ----------
    buyer_order : np.ndarray
        1-D int array of consumer indices in shuffled visit order.
    targets : list[list[int]]
        Per-consumer lists of firm indices to visit (-1 sentinel = end).
        Built from ``con.shop_visits_targets.tolist()``.
    prices : list[float]
        Per-firm prices. Built from ``prod.price.tolist()``.
    inv : list[float]
        Per-firm inventory levels (mutated in-place).
        Built from ``prod.inventory.tolist()``.
    budget : list[float]
        Per-consumer remaining spending budget (mutated in-place).
        Built from ``con.income_to_spend.tolist()``.

    Notes
    -----
    ``inv`` and ``budget`` are mutated in-place.  The caller writes them
    back to the NumPy arrays after this function returns.
    """
    for c in buyer_order.tolist():
        bc = budget[c]
        for t in targets[c]:
            if t < 0:
                break
            if bc <= EPS:
                break
            it = inv[t]
            if it <= EPS:
                continue
            qty = bc / prices[t]
            if qty > it:
                qty = it
            bc -= qty * prices[t]
            inv[t] = it - qty
        budget[c] = bc


# ---------------------------------------------------------------------------
# Numba @njit kernel (defined only when numba is installed)
# ---------------------------------------------------------------------------
#
# The kernel takes ONLY NumPy arrays + scalars so that numba.njit can
# compile it.  The per-consumer target lists are represented as a
# ``-1``-padded 2D int array (``targets_2d[c, :]``), one row per consumer.
#
# ARITHMETIC ORDER: identical to ``_goods_buy_loop_py`` above.  No
# ``fastmath``, no reassociation.  This guarantees bit-for-bit identical
# results when both paths use the same IEEE-754 double precision.
#
# IMPORT SAFETY: the ``@njit`` decorator is only applied when numba is
# available (guarded by ``HAS_NUMBA``).  Importing this module WITHOUT
# numba installed works fine; ``_goods_buy_loop_nb`` is simply ``None``
# and the dispatch never reaches it.
# ---------------------------------------------------------------------------


def _goods_buy_loop_nb_impl(
    buyer_order: np.ndarray,
    targets_2d: np.ndarray,
    prices: np.ndarray,
    inv: np.ndarray,
    budget: np.ndarray,
    eps: float,
) -> None:
    """Inner body of the Numba goods-market kernel.

    Parameters
    ----------
    buyer_order : np.ndarray
        1-D int array of consumer indices in shuffled visit order.
    targets_2d : np.ndarray
        2-D int array of shape ``(n_consumers, max_Z)``.  Each row contains
        firm indices for that consumer; unused slots are ``-1``.
    prices : np.ndarray
        1-D float64 array of per-firm prices.
    inv : np.ndarray
        1-D float64 array of per-firm inventory (mutated in-place).
    budget : np.ndarray
        1-D float64 array of per-consumer spending budget (mutated in-place).
    eps : float
        Epsilon threshold (same ``EPS`` constant used by the Python path).

    Notes
    -----
    Arithmetic order is intentionally identical to ``_goods_buy_loop_py``.
    No ``fastmath``.
    """
    n_slots = targets_2d.shape[1]
    for i in range(buyer_order.shape[0]):
        c = buyer_order[i]
        bc = budget[c]
        for j in range(n_slots):
            t = targets_2d[c, j]
            if t < 0:
                break
            if bc <= eps:
                break
            it = inv[t]
            if it <= eps:
                continue
            qty = bc / prices[t]
            if qty > it:
                qty = it
            bc -= qty * prices[t]
            inv[t] = it - qty
        budget[c] = bc


if HAS_NUMBA:
    _goods_buy_loop_nb = numba.njit(cache=True)(_goods_buy_loop_nb_impl)
else:
    _goods_buy_loop_nb = None


def _goods_buy_loop_nb_wrapper(
    buyer_order: np.ndarray,
    targets_2d: np.ndarray,
    prices: np.ndarray,
    inv: np.ndarray,
    budget: np.ndarray,
) -> None:
    """Thin wrapper that calls the compiled Numba kernel.

    Callers build the padded 2D targets array once and pass it here.
    This wrapper simply forwards to the compiled function with the ``EPS``
    constant; it exists so the caller does not need to know about the eps
    argument.
    """
    assert _goods_buy_loop_nb is not None  # only called when HAS_NUMBA
    _goods_buy_loop_nb(buyer_order, targets_2d, prices, inv, budget, EPS)


def _select_goods_kernel(goods_kernel: str) -> str:
    """Resolve the effective kernel name from the config value.

    Parameters
    ----------
    goods_kernel : str
        Config value: ``"auto"``, ``"numba"``, or ``"python"``.

    Returns
    -------
    str
        Either ``"numba"`` or ``"python"``.

    Raises
    ------
    RuntimeError
        If ``goods_kernel="numba"`` but Numba is not installed.
    """
    if goods_kernel == "python":
        return "python"
    if goods_kernel == "numba":
        if not HAS_NUMBA:
            raise RuntimeError(
                "goods_kernel='numba' requires numba to be installed. "
                "Run: pip install bamengine[fast]"
            )
        return "numba"
    # "auto": use Numba if available, else Python
    return "numba" if HAS_NUMBA else "python"


def goods_market_round(
    con: Consumer,
    prod: Producer,
    *,
    max_Z: int,
    rng: Rng = make_rng(),
    goods_kernel: str = "auto",
) -> None:
    """Sequential goods market matching.

    Each consumer completes all shopping visits before the next consumer
    starts, matching the book's specification (Section 3.4).  Consumers
    are shuffled each period.

    The inner loop uses Python lists (via ``.tolist()``) to avoid NumPy
    per-element indexing overhead.  Results are written back to the
    original NumPy arrays after all purchases complete.

    Parameters
    ----------
    con : Consumer
        Consumer role (households).
    prod : Producer
        Producer role (firms).
    max_Z : int
        Maximum shopping visits per consumer.
    rng : Rng
        Random generator for consumer ordering.
    goods_kernel : str
        Which loop implementation to use: ``"auto"``, ``"numba"``, or
        ``"python"``.  Default ``"auto"`` selects Numba when available.
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Goods Market Round ---")

    # Identify consumers with budget
    buyers = np.where(con.income_to_spend > EPS)[0]
    if buyers.size == 0:
        if info_enabled:
            log.info("  No consumers with budget. Skipping.")
            log.info("--- Goods Market Round complete ---")
        return

    if prod.inventory.sum() <= EPS:
        if info_enabled:
            log.info("  No inventory available. Skipping.")
            log.info("--- Goods Market Round complete ---")
        return

    # Shuffle consumers: randomized order each period (RNG used HERE, not inside the loop)
    rng.shuffle(buyers)

    # Resolve kernel once; dispatch happens below.
    effective_kernel = _select_goods_kernel(goods_kernel)

    # Snapshot pre-loop state for INFO statistics (inventory/budget deltas avoid a
    # third divergent loop: the aggregate stats are derived AFTER the single buying
    # pass, regardless of which kernel ran).
    if info_enabled:
        inv_before = prod.inventory.copy()
        budget_before = con.income_to_spend.copy()

    if effective_kernel == "numba":
        # Numba path: pass NumPy arrays directly.
        # Build padded 2D targets array from the 2D shop_visits_targets buffer.
        # ``con.shop_visits_targets`` is already a 2D int array with -1 sentinels
        # (shape: n_consumers x max_Z), so we can pass it directly.
        inv = prod.inventory.copy()
        budget = con.income_to_spend.copy()
        _goods_buy_loop_nb_wrapper(
            buyers,
            con.shop_visits_targets,
            prod.price,
            inv,
            budget,
        )
        prod.inventory[:] = inv
        con.income_to_spend[:] = budget
    else:
        # Python path: use lists for hot-path performance.
        budget_list = con.income_to_spend.tolist()
        inv_list = prod.inventory.tolist()
        prices_list = prod.price.tolist()
        targets_list = con.shop_visits_targets.tolist()
        _goods_buy_loop_py(buyers, targets_list, prices_list, inv_list, budget_list)
        con.income_to_spend[:] = budget_list
        prod.inventory[:] = inv_list

    if info_enabled:
        # Derive aggregate statistics from the loop outputs (inventory/budget deltas).
        # This is the single source of truth: no third divergent loop.
        # Note: quantity sold != -delta_inv in general when prices differ per firm,
        # but revenue = delta_budget (spending equals budget decrease).
        delta_inv = inv_before - prod.inventory  # per-firm inventory decrease
        delta_budget = budget_before - con.income_to_spend  # per-consumer spending
        total_qty = float(delta_inv.sum())
        total_revenue = float(delta_budget.sum())
        # Count the number of purchasing transactions is not directly recoverable
        # from deltas alone; report qty and revenue which are exact.
        log.info(f"  qty_sold={total_qty:,.2f}, revenue={total_revenue:,.2f}")
        log.info("--- Goods Market Round complete ---")
