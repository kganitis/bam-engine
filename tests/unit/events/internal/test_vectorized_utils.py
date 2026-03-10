"""Unit tests for vectorized utility functions."""

from __future__ import annotations

import numpy as np

from bamengine.utils import (
    _flatten_and_shuffle_groups,
    grouped_cumsum,
    resolve_conflicts,
)

# ── grouped_cumsum ────────────────────────────────────────────────────────────


class TestGroupedCumsum:
    """Tests for grouped_cumsum."""

    def test_single_group(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        starts = np.array([0])
        result = grouped_cumsum(vals, starts)
        expected = np.cumsum(vals)
        np.testing.assert_array_almost_equal(result, expected)

    def test_two_groups(self):
        vals = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
        starts = np.array([0, 3])
        result = grouped_cumsum(vals, starts)
        expected = np.array([1.0, 3.0, 6.0, 10.0, 30.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_three_groups(self):
        vals = np.array([5.0, 5.0, 1.0, 2.0, 3.0, 100.0])
        starts = np.array([0, 2, 5])
        result = grouped_cumsum(vals, starts)
        expected = np.array([5.0, 10.0, 1.0, 3.0, 6.0, 100.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_empty_input(self):
        vals = np.array([], dtype=np.float64)
        starts = np.array([0])
        result = grouped_cumsum(vals, starts)
        assert result.size == 0

    def test_single_element_groups(self):
        vals = np.array([7.0, 3.0, 9.0])
        starts = np.array([0, 1, 2])
        result = grouped_cumsum(vals, starts)
        expected = np.array([7.0, 3.0, 9.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_element(self):
        vals = np.array([42.0])
        starts = np.array([0])
        result = grouped_cumsum(vals, starts)
        np.testing.assert_array_almost_equal(result, np.array([42.0]))

    def test_no_group_starts(self):
        """Empty group_starts => treat as single group."""
        vals = np.array([1.0, 2.0, 3.0])
        starts = np.array([], dtype=np.intp)
        result = grouped_cumsum(vals, starts)
        expected = np.cumsum(vals)
        np.testing.assert_array_almost_equal(result, expected)

    def test_large_groups(self):
        """Verify correctness with larger arrays."""
        rng = np.random.default_rng(42)
        vals = rng.random(100)
        starts = np.array([0, 25, 50, 75])
        result = grouped_cumsum(vals, starts)
        # Manually check each group
        for i in range(len(starts)):
            lo = starts[i]
            hi = starts[i + 1] if i + 1 < len(starts) else 100
            np.testing.assert_array_almost_equal(result[lo:hi], np.cumsum(vals[lo:hi]))


# ── resolve_conflicts ─────────────────────────────────────────────────────────


class TestResolveConflicts:
    """Tests for resolve_conflicts."""

    def test_no_conflicts(self):
        """Each sender targets a unique target — all accepted."""
        rng = np.random.default_rng(42)
        senders = np.array([0, 1, 2])
        targets = np.array([0, 1, 2])
        cap = np.array([1, 1, 1])
        accepted = resolve_conflicts(senders, targets, cap, 3, rng)
        assert accepted.all()

    def test_all_same_target_cap_one(self):
        """Three senders, one target, capacity=1 — exactly one accepted."""
        rng = np.random.default_rng(42)
        senders = np.array([0, 1, 2])
        targets = np.array([0, 0, 0])
        cap = np.array([1])
        accepted = resolve_conflicts(senders, targets, cap, 1, rng)
        assert accepted.sum() == 1

    def test_oversubscribed_cap_two(self):
        """Five senders target same, capacity=2 — exactly two accepted."""
        rng = np.random.default_rng(42)
        senders = np.arange(5)
        targets = np.array([0, 0, 0, 0, 0])
        cap = np.array([2])
        accepted = resolve_conflicts(senders, targets, cap, 1, rng)
        assert accepted.sum() == 2

    def test_empty_senders(self):
        rng = np.random.default_rng(42)
        senders = np.array([], dtype=np.intp)
        targets = np.array([], dtype=np.intp)
        cap = np.array([5, 5])
        accepted = resolve_conflicts(senders, targets, cap, 2, rng)
        assert accepted.size == 0

    def test_zero_capacity(self):
        """Zero capacity — no one accepted."""
        rng = np.random.default_rng(42)
        senders = np.array([0, 1])
        targets = np.array([0, 0])
        cap = np.array([0])
        accepted = resolve_conflicts(senders, targets, cap, 1, rng)
        assert accepted.sum() == 0

    def test_mixed_targets(self):
        """Multiple targets with different subscription levels."""
        rng = np.random.default_rng(42)
        senders = np.arange(6)
        targets = np.array([0, 0, 0, 1, 1, 2])
        cap = np.array([1, 2, 1])
        accepted = resolve_conflicts(senders, targets, cap, 3, rng)
        # Target 0: 3 senders, cap=1 → 1 accepted
        # Target 1: 2 senders, cap=2 → 2 accepted
        # Target 2: 1 sender, cap=1 → 1 accepted
        assert accepted.sum() == 4

    def test_under_subscribed(self):
        """Fewer senders than capacity — all accepted."""
        rng = np.random.default_rng(42)
        senders = np.array([0, 1])
        targets = np.array([0, 0])
        cap = np.array([5])
        accepted = resolve_conflicts(senders, targets, cap, 1, rng)
        assert accepted.all()

    def test_deterministic_with_seed(self):
        """Same seed produces same results."""
        senders = np.arange(10)
        targets = np.zeros(10, dtype=np.intp)
        cap = np.array([3])
        r1 = resolve_conflicts(senders, targets, cap, 1, np.random.default_rng(99))
        r2 = resolve_conflicts(senders, targets, cap, 1, np.random.default_rng(99))
        np.testing.assert_array_equal(r1, r2)

    def test_mask_aligns_with_original_order(self):
        """Accepted mask indices correspond to correct sender positions."""
        rng = np.random.default_rng(42)
        senders = np.array([10, 20, 30, 40, 50])
        targets = np.array([0, 0, 0, 1, 1])
        cap = np.array([1, 1])
        accepted = resolve_conflicts(senders, targets, cap, 2, rng)
        # Exactly 1 from target 0 (indices 0-2) and 1 from target 1 (indices 3-4)
        assert accepted[:3].sum() == 1
        assert accepted[3:].sum() == 1

    def test_many_targets_mixed_subscription(self):
        """50 targets, mix of over/under/exactly-subscribed — total accepted is correct."""
        rng = np.random.default_rng(123)
        n_targets = 50
        n_senders = 200
        targets = rng.integers(0, n_targets, size=n_senders)
        cap = rng.integers(1, 6, size=n_targets)  # 1..5 capacity each

        accepted = resolve_conflicts(
            np.arange(n_senders), targets, cap, n_targets, np.random.default_rng(42)
        )

        # Per-target: accepted count == min(senders_to_target, capacity)
        for t in range(n_targets):
            mask_t = targets == t
            senders_to_t = mask_t.sum()
            expected = min(senders_to_t, cap[t])
            assert accepted[mask_t].sum() == expected, (
                f"target {t}: expected {expected}"
            )


# ── _flatten_and_shuffle_groups ──────────────────────────────────────────────


class TestFlattenAndShuffleGroups:
    """Tests for _flatten_and_shuffle_groups."""

    def test_single_group(self):
        """Single group returns all items shuffled."""
        source = np.array([10, 20, 30, 40, 50])
        lo = np.array([1])
        hi = np.array([4])
        rng = np.random.default_rng(42)

        items, gidx, rank, sizes, _starts = _flatten_and_shuffle_groups(
            source, lo, hi, rng
        )
        assert items.size == 3  # hi - lo = 3
        assert set(items.tolist()) == {20, 30, 40}
        assert (gidx == 0).all()
        np.testing.assert_array_equal(rank, [0, 1, 2])
        np.testing.assert_array_equal(sizes, [3])

    def test_multiple_groups_different_sizes(self):
        """Multiple groups with different sizes preserve elements."""
        source = np.arange(10)  # [0..9]
        lo = np.array([0, 2, 7])
        hi = np.array([2, 7, 10])
        rng = np.random.default_rng(42)

        items, gidx, rank, sizes, _starts = _flatten_and_shuffle_groups(
            source, lo, hi, rng
        )
        assert items.size == 10
        np.testing.assert_array_equal(sizes, [2, 5, 3])
        # Each group's items are a permutation of the original slice
        assert set(items[gidx == 0].tolist()) == {0, 1}
        assert set(items[gidx == 1].tolist()) == {2, 3, 4, 5, 6}
        assert set(items[gidx == 2].tolist()) == {7, 8, 9}
        # Rank within each group is 0..size-1
        for g in range(3):
            g_rank = rank[gidx == g]
            np.testing.assert_array_equal(sorted(g_rank), np.arange(sizes[g]))

    def test_empty_groups(self):
        """Empty input returns empty arrays."""
        source = np.arange(5)
        lo = np.array([], dtype=np.intp)
        hi = np.array([], dtype=np.intp)
        rng = np.random.default_rng(42)

        items, _gidx, _rank, _sizes, _starts = _flatten_and_shuffle_groups(
            source, lo, hi, rng
        )
        assert items.size == 0

    def test_deterministic(self):
        """Same seed produces identical results."""
        source = np.arange(20)
        lo = np.array([0, 5, 15])
        hi = np.array([5, 15, 20])

        r1 = _flatten_and_shuffle_groups(source, lo, hi, np.random.default_rng(99))
        r2 = _flatten_and_shuffle_groups(source, lo, hi, np.random.default_rng(99))
        for a, b in zip(r1, r2, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_zero_size_group_skipped(self):
        """Groups where lo == hi contribute zero items."""
        source = np.arange(10)
        lo = np.array([0, 3, 3, 7])  # group 2 is empty (lo==hi)
        hi = np.array([3, 3, 7, 10])
        rng = np.random.default_rng(42)

        items, gidx, _rank, sizes, _starts = _flatten_and_shuffle_groups(
            source, lo, hi, rng
        )
        assert items.size == 10  # 3 + 0 + 4 + 3
        np.testing.assert_array_equal(sizes, [3, 0, 4, 3])
        # group 1 (empty) should have no items
        assert (gidx == 1).sum() == 0
