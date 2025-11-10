"""Tests for Relationship base class and registry."""

import numpy as np
import pytest

from bamengine.core import role, relationship
from bamengine.typing import Float1D


@pytest.fixture
def clean_relationship_registry():
    """
    Save relationship registry state, clear it for the test, then restore it.

    This fixture ensures relationship tests are isolated from real BAM
    relationships and from each other.
    """
    # noinspection PyProtectedMember
    from bamengine.core.registry import _RELATIONSHIP_REGISTRY

    # Save current state
    saved_relationships = dict(_RELATIONSHIP_REGISTRY)

    # Clear for test
    _RELATIONSHIP_REGISTRY.clear()

    yield

    # Restore original state
    _RELATIONSHIP_REGISTRY.clear()
    _RELATIONSHIP_REGISTRY.update(saved_relationships)


# ============================================================================
# Relationship Creation and Basic Operations
# ============================================================================


def test_relationship_creation():
    """Test creating a concrete relationship."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.array([0, 1, 2], dtype=np.int32),
        target_ids=np.array([10, 11, 12], dtype=np.int32),
        weight=np.array([1.0, 2.0, 3.0]),
        size=3,
        capacity=10,
    )

    assert rel.size == 3
    assert rel.capacity == 10
    assert len(rel.source_ids) == 3
    assert len(rel.target_ids) == 3
    np.testing.assert_array_equal(rel.weight, [1.0, 2.0, 3.0])


def test_relationship_slots():
    """Test that relationship uses slots (no __dict__)."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.zeros(5, dtype=np.int32),
        target_ids=np.zeros(5, dtype=np.int32),
        weight=np.zeros(5),
        size=0,
        capacity=5,
    )

    assert not hasattr(rel, "__dict__")


# ============================================================================
# Query Methods
# ============================================================================


def test_query_sources():
    """Test querying edges by source ID."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.array([0, 1, 0, 2, 0], dtype=np.int32),
        target_ids=np.array([10, 11, 12, 13, 14], dtype=np.int32),
        weight=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        size=5,
        capacity=10,
    )

    # Query source 0 (appears at indices 0, 2, 4)
    indices = rel.query_sources(0)
    np.testing.assert_array_equal(indices, [0, 2, 4])

    # Query source 1 (appears at index 1)
    indices = rel.query_sources(1)
    np.testing.assert_array_equal(indices, [1])

    # Query non-existent source
    indices = rel.query_sources(999)
    assert indices.size == 0


def test_query_targets():
    """Test querying edges by target ID."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        target_ids=np.array([10, 10, 10, 11, 12], dtype=np.int32),
        weight=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        size=5,
        capacity=10,
    )

    # Query target 10 (appears at indices 0, 1, 2)
    indices = rel.query_targets(10)
    np.testing.assert_array_equal(indices, [0, 1, 2])

    # Query target 11 (appears at index 3)
    indices = rel.query_targets(11)
    np.testing.assert_array_equal(indices, [3])

    # Query non-existent target
    indices = rel.query_targets(999)
    assert indices.size == 0


# ============================================================================
# Aggregate Methods
# ============================================================================


def test_aggregate_by_source_sum():
    """Test summing component values by source."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.array([0, 1, 0, 2, 0], dtype=np.int32),
        target_ids=np.array([10, 11, 12, 13, 14], dtype=np.int32),
        weight=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        size=5,
        capacity=10,
    )

    result = rel.aggregate_by_source(rel.weight, func="sum", n_sources=3)
    np.testing.assert_array_almost_equal(result, [9.0, 2.0, 4.0])  # 1+3+5, 2, 4


def test_aggregate_by_source_mean():
    """Test averaging component values by source."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.array([0, 1, 0, 2, 0], dtype=np.int32),
        target_ids=np.array([10, 11, 12, 13, 14], dtype=np.int32),
        weight=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        size=5,
        capacity=10,
    )

    result = rel.aggregate_by_source(rel.weight, func="mean", n_sources=3)
    np.testing.assert_array_almost_equal(result, [3.0, 2.0, 4.0])  # (1+3+5)/3, 2/1, 4/1


def test_aggregate_by_source_count():
    """Test counting edges by source."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.array([0, 1, 0, 2, 0], dtype=np.int32),
        target_ids=np.array([10, 11, 12, 13, 14], dtype=np.int32),
        weight=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        size=5,
        capacity=10,
    )

    result = rel.aggregate_by_source(rel.weight, func="count", n_sources=3)
    np.testing.assert_array_equal(result, [3, 1, 1])


def test_aggregate_by_target_sum():
    """Test summing component values by target."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        target_ids=np.array([0, 0, 0, 1, 2], dtype=np.int32),
        weight=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        size=5,
        capacity=10,
    )

    result = rel.aggregate_by_target(rel.weight, func="sum", n_targets=3)
    np.testing.assert_array_almost_equal(result, [6.0, 4.0, 5.0])  # 1+2+3, 4, 5


def test_aggregate_by_target_mean():
    """Test averaging component values by target."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        target_ids=np.array([0, 0, 0, 1, 2], dtype=np.int32),
        weight=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        size=5,
        capacity=10,
    )

    result = rel.aggregate_by_target(rel.weight, func="mean", n_targets=3)
    np.testing.assert_array_almost_equal(result, [2.0, 4.0, 5.0])  # (1+2+3)/3, 4/1, 5/1


def test_aggregate_by_target_count():
    """Test counting edges by target."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        target_ids=np.array([0, 0, 0, 1, 2], dtype=np.int32),
        weight=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        size=5,
        capacity=10,
    )

    result = rel.aggregate_by_target(rel.weight, func="count", n_targets=3)
    np.testing.assert_array_equal(result, [3, 1, 1])


def test_aggregate_empty_relationship():
    """Test aggregation on empty relationship."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.zeros(10, dtype=np.int32),
        target_ids=np.zeros(10, dtype=np.int32),
        weight=np.zeros(10),
        size=0,
        capacity=10,
    )

    result = rel.aggregate_by_source(rel.weight, func="sum", n_sources=5)
    np.testing.assert_array_equal(result, np.zeros(5))


def test_aggregate_with_preallocated_output():
    """Test aggregation with pre-allocated output array."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.array([0, 1, 0], dtype=np.int32),
        target_ids=np.array([10, 11, 12], dtype=np.int32),
        weight=np.array([1.0, 2.0, 3.0]),
        size=3,
        capacity=10,
    )

    out = np.ones(2)  # Pre-allocate with non-zero values
    result = rel.aggregate_by_source(rel.weight, func="sum", n_sources=2, out=out)

    # Should overwrite out array (zero then accumulate)
    np.testing.assert_array_almost_equal(result, [4.0, 2.0])
    assert result is out  # Should be same object


# ============================================================================
# Edge Management
# ============================================================================


def test_drop_rows():
    """Test removing edges by boolean mask."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        target_ids=np.array([10, 11, 12, 13, 14], dtype=np.int32),
        weight=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        size=5,
        capacity=10,
    )

    # Drop edges at indices 1 and 3
    mask = np.array([False, True, False, True, False])
    n_dropped = rel.drop_rows(mask)

    assert n_dropped == 2
    assert rel.size == 3
    np.testing.assert_array_equal(rel.source_ids[:3], [0, 2, 4])
    np.testing.assert_array_equal(rel.target_ids[:3], [10, 12, 14])
    # Note: weight not automatically compacted (subclass responsibility)


def test_drop_rows_empty():
    """Test dropping rows from empty relationship."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.zeros(10, dtype=np.int32),
        target_ids=np.zeros(10, dtype=np.int32),
        weight=np.zeros(10),
        size=0,
        capacity=10,
    )

    mask = np.zeros(10, dtype=bool)
    n_dropped = rel.drop_rows(mask)

    assert n_dropped == 0
    assert rel.size == 0


def test_purge_sources():
    """Test removing all edges from specific sources."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.array([0, 1, 0, 2, 1], dtype=np.int32),
        target_ids=np.array([10, 11, 12, 13, 14], dtype=np.int32),
        weight=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        size=5,
        capacity=10,
    )

    # Purge edges from sources 0 and 1
    n_removed = rel.purge_sources(np.array([0, 1]))

    assert n_removed == 4  # Edges at indices 0, 1, 2, 4
    assert rel.size == 1
    np.testing.assert_array_equal(rel.source_ids[:1], [2])
    np.testing.assert_array_equal(rel.target_ids[:1], [13])


def test_purge_targets():
    """Test removing all edges to specific targets."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        target_ids=np.array([10, 11, 10, 12, 11], dtype=np.int32),
        weight=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        size=5,
        capacity=10,
    )

    # Purge edges to targets 10 and 11
    n_removed = rel.purge_targets(np.array([10, 11]))

    assert n_removed == 4  # Edges at indices 0, 1, 2, 4
    assert rel.size == 1
    np.testing.assert_array_equal(rel.source_ids[:1], [3])
    np.testing.assert_array_equal(rel.target_ids[:1], [12])


def test_append_edges():
    """Test appending new edges."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.array([0, 1, -1, -1, -1], dtype=np.int32),
        target_ids=np.array([10, 11, -1, -1, -1], dtype=np.int32),
        weight=np.array([1.0, 2.0, 0.0, 0.0, 0.0]),
        size=2,
        capacity=5,
    )

    # Append 2 new edges
    new_sources = np.array([2, 3], dtype=np.int32)
    new_targets = np.array([12, 13], dtype=np.int32)
    rel.append_edges(new_sources, new_targets)

    assert rel.size == 4
    np.testing.assert_array_equal(rel.source_ids[:4], [0, 1, 2, 3])
    np.testing.assert_array_equal(rel.target_ids[:4], [10, 11, 12, 13])


def test_append_edges_exceeds_capacity():
    """Test that appending beyond capacity raises error."""

    @relationship
    class SimpleRelation:
        weight: Float1D

    rel = SimpleRelation(
        source_ids=np.array([0, 1], dtype=np.int32),
        target_ids=np.array([10, 11], dtype=np.int32),
        weight=np.array([1.0, 2.0]),
        size=2,
        capacity=3,
    )

    # Try to append 2 edges when only 1 slot available
    new_sources = np.array([2, 3], dtype=np.int32)
    new_targets = np.array([12, 13], dtype=np.int32)

    with pytest.raises(ValueError, match="Cannot append.*exceed capacity"):
        rel.append_edges(new_sources, new_targets)
