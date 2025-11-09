# tests/unit/test_ops.py
"""Unit tests for bamengine.ops module."""

import numpy as np

from bamengine import ops


class TestArithmetic:
    """Test arithmetic operations."""

    def test_add_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = ops.add(a, b)
        np.testing.assert_array_equal(result, [5.0, 7.0, 9.0])

    def test_add_scalar(self):
        a = np.array([1.0, 2.0, 3.0])
        result = ops.add(a, 10.0)
        np.testing.assert_array_equal(result, [11.0, 12.0, 13.0])

    def test_add_out_parameter(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        out = np.zeros(3)
        result = ops.add(a, b, out=out)
        assert result is out
        np.testing.assert_array_equal(out, [5.0, 7.0, 9.0])

    def test_subtract_arrays(self):
        a = np.array([5.0, 7.0, 9.0])
        b = np.array([1.0, 2.0, 3.0])
        result = ops.subtract(a, b)
        np.testing.assert_array_equal(result, [4.0, 5.0, 6.0])

    def test_multiply_arrays(self):
        a = np.array([2.0, 3.0, 4.0])
        b = np.array([5.0, 6.0, 7.0])
        result = ops.multiply(a, b)
        np.testing.assert_array_equal(result, [10.0, 18.0, 28.0])

    def test_multiply_scalar(self):
        a = np.array([2.0, 3.0, 4.0])
        result = ops.multiply(a, 2.0)
        np.testing.assert_array_equal(result, [4.0, 6.0, 8.0])

    def test_divide_arrays(self):
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([2.0, 4.0, 5.0])
        result = ops.divide(a, b)
        np.testing.assert_array_equal(result, [5.0, 5.0, 6.0])

    def test_divide_by_zero_safe(self):
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([0.0, 0.0, 0.0])
        result = ops.divide(a, b)
        # Should not raise, uses epsilon for safety
        assert np.all(np.isfinite(result))

    def test_divide_scalar_zero_safe(self):
        a = np.array([10.0, 20.0, 30.0])
        result = ops.divide(a, 0.0)
        # Should not raise
        assert np.all(np.isfinite(result))


class TestAssignment:
    """Test assignment operations."""

    def test_assign_scalar(self):
        target = np.array([1.0, 2.0, 3.0])
        ops.assign(target, 5.0)
        np.testing.assert_array_equal(target, [5.0, 5.0, 5.0])

    def test_assign_array(self):
        target = np.array([1.0, 2.0, 3.0])
        value = np.array([7.0, 8.0, 9.0])
        ops.assign(target, value)
        np.testing.assert_array_equal(target, [7.0, 8.0, 9.0])


class TestComparisons:
    """Test comparison operations."""

    def test_equal(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 0.0, 3.0])
        result = ops.equal(a, b)
        np.testing.assert_array_equal(result, [True, False, True])

    def test_not_equal(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 0.0, 3.0])
        result = ops.not_equal(a, b)
        np.testing.assert_array_equal(result, [False, True, False])

    def test_less(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 2.0, 2.0])
        result = ops.less(a, b)
        np.testing.assert_array_equal(result, [True, False, False])

    def test_less_equal(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 2.0, 2.0])
        result = ops.less_equal(a, b)
        np.testing.assert_array_equal(result, [True, True, False])

    def test_greater(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 2.0, 2.0])
        result = ops.greater(a, b)
        np.testing.assert_array_equal(result, [False, False, True])

    def test_greater_equal(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 2.0, 2.0])
        result = ops.greater_equal(a, b)
        np.testing.assert_array_equal(result, [False, True, True])

    def test_comparison_with_scalar(self):
        a = np.array([1.0, 2.0, 3.0])
        result = ops.greater(a, 2.0)
        np.testing.assert_array_equal(result, [False, False, True])


class TestLogical:
    """Test logical operations."""

    def test_logical_and(self):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        result = ops.logical_and(a, b)
        np.testing.assert_array_equal(result, [True, False, False, False])

    def test_logical_or(self):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        result = ops.logical_or(a, b)
        np.testing.assert_array_equal(result, [True, True, True, False])

    def test_logical_not(self):
        a = np.array([True, False, True, False])
        result = ops.logical_not(a)
        np.testing.assert_array_equal(result, [False, True, False, True])


class TestConditional:
    """Test conditional operations."""

    def test_where_arrays(self):
        condition = np.array([True, False, True, False])
        true_val = np.array([10.0, 20.0, 30.0, 40.0])
        false_val = np.array([1.0, 2.0, 3.0, 4.0])
        result = ops.where(condition, true_val, false_val)
        np.testing.assert_array_equal(result, [10.0, 2.0, 30.0, 4.0])

    def test_where_scalars(self):
        condition = np.array([True, False, True, False])
        result = ops.where(condition, 100.0, 0.0)
        np.testing.assert_array_equal(result, [100.0, 0.0, 100.0, 0.0])

    def test_select_single_condition(self):
        condition = np.array([True, False, True, False])
        result = ops.select([condition], [10.0], default=0.0)
        np.testing.assert_array_equal(result, [10.0, 0.0, 10.0, 0.0])

    def test_select_multiple_conditions(self):
        cond1 = np.array([True, False, False, False])
        cond2 = np.array([False, True, False, False])
        cond3 = np.array([False, False, True, False])
        result = ops.select([cond1, cond2, cond3], [10.0, 20.0, 30.0], default=0.0)
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0, 0.0])


class TestElementWise:
    """Test element-wise operations."""

    def test_maximum_arrays(self):
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([4.0, 2.0, 6.0])
        result = ops.maximum(a, b)
        np.testing.assert_array_equal(result, [4.0, 5.0, 6.0])

    def test_maximum_scalar(self):
        a = np.array([1.0, 5.0, 3.0])
        result = ops.maximum(a, 3.0)
        np.testing.assert_array_equal(result, [3.0, 5.0, 3.0])

    def test_minimum_arrays(self):
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([4.0, 2.0, 6.0])
        result = ops.minimum(a, b)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_minimum_scalar(self):
        a = np.array([1.0, 5.0, 3.0])
        result = ops.minimum(a, 3.0)
        np.testing.assert_array_equal(result, [1.0, 3.0, 3.0])

    def test_clip(self):
        a = np.array([1.0, 5.0, 10.0, 15.0])
        result = ops.clip(a, 3.0, 12.0)
        np.testing.assert_array_equal(result, [3.0, 5.0, 10.0, 12.0])

    def test_clip_out_parameter(self):
        a = np.array([1.0, 5.0, 10.0, 15.0])
        out = np.zeros(4)
        result = ops.clip(a, 3.0, 12.0, out=out)
        assert result is out
        np.testing.assert_array_equal(out, [3.0, 5.0, 10.0, 12.0])


class TestAggregation:
    """Test aggregation operations."""

    def test_sum_default(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        result = ops.sum(a)
        assert result == 10.0

    def test_sum_axis(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ops.sum(a, axis=0)
        np.testing.assert_array_equal(result, [4.0, 6.0])

    def test_mean_default(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        result = ops.mean(a)
        assert result == 2.5

    def test_mean_axis(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ops.mean(a, axis=0)
        np.testing.assert_array_equal(result, [2.0, 3.0])

    def test_any_true(self):
        a = np.array([False, False, True, False])
        result = ops.any(a)
        assert result is True

    def test_any_false(self):
        a = np.array([False, False, False])
        result = ops.any(a)
        assert result is False

    def test_all_true(self):
        a = np.array([True, True, True])
        result = ops.all(a)
        assert result is True

    def test_all_false(self):
        a = np.array([True, False, True])
        result = ops.all(a)
        assert result is False


class TestArrayCreation:
    """Test array creation operations."""

    def test_zeros(self):
        result = ops.zeros(5)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_ones(self):
        result = ops.ones(3)
        np.testing.assert_array_equal(result, [1.0, 1.0, 1.0])

    def test_full(self):
        result = ops.full(4, 7.5)
        np.testing.assert_array_equal(result, [7.5, 7.5, 7.5, 7.5])

    def test_empty_shape(self):
        result = ops.empty(3)
        assert result.shape == (3,)
        assert result.dtype == np.float64


class TestUtilities:
    """Test utility operations."""

    def test_unique(self):
        a = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
        result = ops.unique(a)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_bincount(self):
        a = np.array([0, 1, 1, 2, 2, 2], dtype=np.int64)
        result = ops.bincount(a)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_isin(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        test_elements = np.array([2.0, 4.0, 6.0])
        result = ops.isin(a, test_elements)
        np.testing.assert_array_equal(result, [False, True, False, True])

    def test_argsort(self):
        a = np.array([3.0, 1.0, 4.0, 2.0])
        result = ops.argsort(a)
        np.testing.assert_array_equal(result, [1, 3, 0, 2])

    def test_sort(self):
        a = np.array([3.0, 1.0, 4.0, 2.0])
        result = ops.sort(a)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])


class TestRandom:
    """Test random operations."""

    def test_uniform_shape(self):
        rng = np.random.default_rng(42)
        result = ops.uniform(rng, 0.0, 1.0, 10)
        assert result.shape == (10,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_uniform_bounds(self):
        rng = np.random.default_rng(42)
        result = ops.uniform(rng, 5.0, 10.0, 100)
        assert np.all(result >= 5.0)
        assert np.all(result <= 10.0)

    def test_uniform_deterministic(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        result1 = ops.uniform(rng1, 0.0, 1.0, 10)
        result2 = ops.uniform(rng2, 0.0, 1.0, 10)
        np.testing.assert_array_equal(result1, result2)
