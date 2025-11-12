"""
Array operations for custom events.

This module provides NumPy-free operations for writing custom events,
allowing economics researchers to work with BAM Engine without deep
Python/NumPy knowledge.

Design Notes
------------
The operations mirror NumPy's functionality but with:

- **Safe defaults**: Division by zero prevention (eps=1e-10)
- **Consistent naming**: Verb-based (multiply, divide vs * and /)
- **In-place operations**: Support `out=` parameter for performance
- **Type hints**: Better IDE support with Float, Int, Bool types

Operation Categories
--------------------
- **Arithmetic**: add, subtract, multiply, divide
- **Comparisons**: equal, less, greater, etc.
- **Logical**: logical_and, logical_or, logical_not
- **Conditional**: where (if-then-else), select (switch-case)
- **Element-wise**: maximum, minimum, clip
- **Aggregation**: sum, mean, any, all
- **Array creation**: zeros, ones, full, empty
- **Utilities**: unique, bincount, isin, argsort, sort
- **Random**: uniform (requires RNG)
- **Assignment**: assign (in-place array modification)

Examples
--------
Basic arithmetic operations:

>>> from bamengine import ops
>>> import numpy as np
>>> a = np.array([1.0, 2.0, 3.0])
>>> b = np.array([4.0, 5.0, 6.0])
>>> ops.add(a, b)
array([5., 7., 9.])
>>> ops.multiply(a, 2.0)
array([2., 4., 6.])

Safe division (handles zeros):

>>> wages = np.array([10.0, 12.0, 11.0])
>>> productivity = np.array([2.0, 3.0, 0.0])  # One zero!
>>> unit_cost = ops.divide(wages, productivity)  # No error
>>> unit_cost[2] > 1e9  # Zero productivity handled
True

In-place operations for performance:

>>> out = np.zeros(3)
>>> result = ops.add(a, b, out=out)
>>> result is out  # Same object
True

Conditional logic without NumPy:

>>> has_inventory = ops.greater(inventory, 0)
>>> new_price = ops.where(has_inventory, price * 0.95, price * 1.05)

Create a custom pricing event:

>>> from bamengine import event, ops
>>>
>>> @event
... class MarkupPricing:
...     def execute(self, sim):
...         prod = sim.get_role("Producer")
...         emp = sim.get_role("Employer")
...         # Calculate unit cost (safe division)
...         unit_cost = ops.divide(emp.wage_offered, prod.labor_productivity)
...         # Apply 20% markup
...         new_prices = ops.multiply(unit_cost, 1.2)
...         ops.assign(prod.price, new_prices)

Advanced Usage
--------------
For users comfortable with NumPy, direct NumPy operations can be mixed
with bamengine.ops for flexibility:

>>> import numpy as np
>>> # Mix ops and NumPy as needed
>>> from bamengine import event, ops
>>> @event
... class CustomEvent:
...     def execute(self, sim):
...         # Use ops for safety
...         unit_cost = ops.divide(wages, productivity)
...         # Use NumPy directly for complex operations
...         log_prices = np.log(prices)
...         weighted_avg = np.average(prices, weights=market_share)
```

See Also
--------
bamengine.typing : Type aliases (Float, Int, Bool, Agent)
tests.unit.ops.test_ops : Comprehensive operation tests
numpy : Underlying array library
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from bamengine.typing import Agent, Bool, Float, Int

if TYPE_CHECKING:  # pragma: no cover
    from numpy.random import Generator

__all__ = [
    # Arithmetic
    "add",
    "subtract",
    "multiply",
    "divide",
    # Assignment
    "assign",
    # Comparisons
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    # Logical
    "logical_and",
    "logical_or",
    "logical_not",
    # Conditional
    "where",
    "select",
    # Element-wise
    "maximum",
    "minimum",
    "clip",
    # Aggregation
    "sum",
    "mean",
    "any",
    "all",
    # Array creation
    "zeros",
    "ones",
    "full",
    "empty",
    # Utilities
    "unique",
    "bincount",
    "isin",
    "argsort",
    "sort",
    # Random
    "uniform",
]


# === Arithmetic Operations ===


def add(a: Float, b: float | Float, out: Optional[Float] = None) -> Float:
    """
    Add two arrays or array and scalar.

    Parameters
    ----------
    a : array
        First array.
    b : array or float
        Second array or scalar.
    out : array, optional
        Output array. If provided, result is written in-place.

    Returns
    -------
    array
        Result of addition.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> prices = np.array([1.0, 2.0, 3.0])
    >>> markup = 0.5
    >>> new_prices = ops.add(prices, markup)
    >>> # new_prices: [1.5, 2.5, 3.5]
    """
    if out is None:
        return a + b
    np.add(a, b, out=out)
    return out


def subtract(a: Float, b: float | Float, out: Optional[Float] = None) -> Float:
    """
    Subtract two arrays or array and scalar.

    Parameters
    ----------
    a : array
        First array.
    b : array or float
        Second array or scalar to subtract from first.
    out : array, optional
        Output array.

    Returns
    -------
    array
        Result of subtraction (a - b).

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> inventory = np.array([10.0, 5.0, 0.0])
    >>> sold = np.array([2.0, 3.0, 0.0])
    >>> remaining = ops.subtract(inventory, sold)
    >>> # remaining: [8.0, 2.0, 0.0]
    """
    if out is None:
        return a - b
    np.subtract(a, b, out=out)
    return out


def multiply(a: Float, b: float | Float, out: Optional[Float] = None) -> Float:
    """
    Multiply two arrays or array and scalar.

    Parameters
    ----------
    a : array
        First array.
    b : array or float
        Second array or scalar.
    out : array, optional
        Output array.

    Returns
    -------
    array
        Result of multiplication.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> prices = np.array([10.0, 20.0, 30.0])
    >>> # Increase all prices by 10%
    >>> new_prices = ops.multiply(prices, 1.1)
    >>> # new_prices: [11.0, 22.0, 33.0]
    """
    if out is None:
        return a * b
    np.multiply(a, b, out=out)
    return out


def divide(a: Float, b: float | Float, out: Optional[Float] = None) -> Float:
    """
    Divide two arrays or array and scalar (safe - avoids division by zero).

    Automatically replaces zeros in denominator with small epsilon (1e-10).

    Parameters
    ----------
    a : array
        Numerator array.
    b : array or float
        Denominator array or scalar.
    out : array, optional
        Output array.

    Returns
    -------
    array
        Result of division (a / b).

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> wage_bill = np.array([100.0, 200.0, 0.0])
    >>> production = np.array([10.0, 0.0, 0.0])  # Some zeros!
    >>> # Safe division - no divide by zero errors
    >>> unit_cost = ops.divide(wage_bill, production)
    >>> # unit_cost: [10.0, very_large, 0.0]

    Notes
    -----
    This function is safer than NumPy's divide as it prevents
    division by zero errors by replacing zero denominators with 1e-10.
    """
    # Make denominator safe
    b_safe: float | Float
    if isinstance(b, np.ndarray):
        b_safe = np.maximum(b, 1e-10)
    else:
        b_safe = max(b, 1e-10)

    result: Float
    if out is None:
        result = np.divide(a, b_safe)
        return result
    np.divide(a, b_safe, out=out)
    return out


# === Assignment ===


def assign(target: Float, value: float | Float) -> None:
    """
    Assign value to target array (in-place operation).

    This is equivalent to target[:] = value but more explicit.

    Parameters
    ----------
    target : array
        Array to modify.
    value : array or float
        Value(s) to assign.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> prices = np.array([1.0, 2.0, 3.0])
    >>> new_prices = np.array([1.5, 2.5, 3.5])
    >>> ops.assign(prices, new_prices)
    >>> # prices is now [1.5, 2.5, 3.5]
    """
    target[:] = value


# === Comparison Operations ===


def equal(a: Float, b: float | Float) -> Bool:
    """Element-wise equality comparison."""
    return np.equal(a, b)


def not_equal(a: Float, b: float | Float) -> Bool:
    """Element-wise inequality comparison."""
    return np.not_equal(a, b)


def less(a: Float, b: float | Float) -> Bool:
    """Element-wise less than comparison."""
    return a < b


def less_equal(a: Float, b: float | Float) -> Bool:
    """Element-wise less than or equal comparison."""
    return a <= b


def greater(a: Float, b: float | Float) -> Bool:
    """Element-wise greater than comparison."""
    return a > b


def greater_equal(a: Float, b: float | Float) -> Bool:
    """Element-wise greater than or equal comparison."""
    return a >= b


# === Logical Operations ===


def logical_and(a: Bool, b: Bool) -> Bool:
    """
    Element-wise logical AND.

    Parameters
    ----------
    a : bool array
        First condition.
    b : bool array
        Second condition.

    Returns
    -------
    bool array
        Result of a AND b.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> inventory = np.array([10, 0, 5])
    >>> price = np.array([120, 80, 110])
    >>> avg_price = 100
    >>> has_inventory = inventory > 0
    >>> price_high = price > avg_price
    >>> can_sell = ops.logical_and(has_inventory, price_high)
    """
    return np.logical_and(a, b)


def logical_or(a: Bool, b: Bool) -> Bool:
    """
    Element-wise logical OR.

    Parameters
    ----------
    a : bool array
        First condition.
    b : bool array
        Second condition.

    Returns
    -------
    bool array
        Result of a OR b.
    """
    return np.logical_or(a, b)


def logical_not(a: Bool) -> Bool:
    """
    Element-wise logical NOT.

    Parameters
    ----------
    a : bool array
        Condition to negate.

    Returns
    -------
    bool array
        Result of NOT a.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> employed = np.array([True, False, True])
    >>> unemployed = ops.logical_not(employed)
    >>> # unemployed: [False, True, False]
    """
    return np.logical_not(a)


# === Conditional Operations ===


def where(condition: Bool, true_val: float | Float, false_val: float | Float) -> Float:
    """
    Vectorized if-then-else (ternary operator).

    Parameters
    ----------
    condition : bool array
        Condition to check.
    true_val : array or float
        Value(s) when condition is True.
    false_val : array or float
        Value(s) when condition is False.

    Returns
    -------
    array
        Selected values based on condition.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> # Discount price if inventory > 0, premium otherwise
    >>> inventory = np.array([10, 0, 5])
    >>> base_price = np.array([100, 100, 100])
    >>> price = ops.where(
    ...     inventory > 0,
    ...     base_price * 0.95,  # 5% discount
    ...     base_price * 1.05   # 5% premium
    ... )
    >>> # price: [95, 105, 95]
    """
    return np.where(condition, true_val, false_val)


def select(
    conditions: list[Bool], choices: list[float | Float], default: float | Float = 0.0
) -> Float:
    """
    Select from multiple conditions (like switch/case statement).

    Conditions are evaluated in order. First true condition determines result.

    Parameters
    ----------
    conditions : list of bool arrays
        List of conditions to check in order.
    choices : list of arrays or floats
        List of values to select from (same length as conditions).
    default : array or float, optional
        Default value if no condition matches.

    Returns
    -------
    array
        Selected values.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> # Production decision with multiple cases
    >>> inventory = np.array([0, 10, 5])
    >>> price = np.array([110, 90, 100])
    >>> current_prod = np.array([100, 100, 100])
    >>> avg_price = 100
    >>>
    >>> # Case 1: No inventory + high price → increase
    >>> # Case 2: Has inventory + low price → decrease
    >>> # Default: keep same
    >>> new_prod = ops.select(
    ...     conditions=[
    ...         (inventory == 0) & (price >= avg_price),
    ...         (inventory > 0) & (price < avg_price),
    ...     ],
    ...     choices=[
    ...         current_prod * 1.1,  # Increase 10%
    ...         current_prod * 0.9,  # Decrease 10%
    ...     ],
    ...     default=current_prod  # Keep same
    ... )
    >>> # new_prod: [110, 90, 100]
    """
    return np.select(conditions, choices, default=default)


# === Element-wise Operations ===


def maximum(a: Float, b: float | Float) -> Float:
    """
    Element-wise maximum of array elements.

    Parameters
    ----------
    a : array
        First array.
    b : array or float
        Second array or scalar.

    Returns
    -------
    array
        Element-wise maximum values.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> # Enforce minimum wage
    >>> proposed_wage = np.array([0.8, 1.0, 1.2])
    >>> min_wage = 0.9
    >>> actual_wage = ops.maximum(proposed_wage, min_wage)
    >>> # actual_wage: [0.9, 1.0, 1.2]
    """
    return np.maximum(a, b)


def minimum(a: Float, b: float | Float) -> Float:
    """
    Element-wise minimum of array elements.

    Parameters
    ----------
    a : array
        First array.
    b : array or float
        Second array or scalar.

    Returns
    -------
    array
        Element-wise minimum values.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> # Cap maximum price
    >>> proposed_price = np.array([90, 100, 110])
    >>> max_price = 105
    >>> actual_price = ops.minimum(proposed_price, max_price)
    >>> # actual_price: [90, 100, 105]
    """
    return np.minimum(a, b)


def clip(
    a: Float, min_val: float, max_val: float, out: Optional[Float] = None
) -> Float:
    """
    Clip (limit) values to range [min_val, max_val].

    Parameters
    ----------
    a : array
        Array to clip.
    min_val : float
        Minimum value.
    max_val : float
        Maximum value.
    out : array, optional
        Output array (in-place operation).

    Returns
    -------
    array
        Clipped array.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> # Keep prices in reasonable range
    >>> prices = np.array([50, 100, 150, 200])
    >>> reasonable_prices = ops.clip(prices, 80, 120)
    >>> # reasonable_prices: [80, 100, 120, 120]
    """
    if out is None:
        return np.clip(a, min_val, max_val)
    np.clip(a, min_val, max_val, out=out)
    return out


# === Aggregation Operations ===


# noinspection PyShadowingBuiltins
def sum(
    a: Float, axis: int | None = None, where: Optional[Bool] = None
) -> float | Float:
    """
    Sum array elements, optionally over axis or subset.

    Parameters
    ----------
    a : array
        Input array.
    axis : int, optional
        Axis to sum over. If None, sum all elements.
    where : bool array, optional
        Mask for subset to sum over.

    Returns
    -------
    float or array
        Sum of array elements (scalar if axis=None, array otherwise).

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> production = np.array([10, 20, 30, 0])
    >>> # Total production
    >>> total = ops.sum(production)
    >>> # total: 60
    >>>
    >>> # Production only from active firms
    >>> active = production > 0
    >>> active_total = ops.sum(production, where=active)
    >>> # active_total: 60
    """
    if where is None:
        result = np.sum(a, axis=axis)
        return float(result) if axis is None else result
    return float(np.sum(a[where]))


def mean(
    a: Float, axis: int | None = None, where: Optional[Bool] = None
) -> float | Float:
    """
    Calculate mean, optionally over axis or subset.

    Parameters
    ----------
    a : array
        Input array.
    axis : int, optional
        Axis to average over. If None, average all elements.
    where : bool array, optional
        Mask for subset.

    Returns
    -------
    float or array
        Mean value (scalar if axis=None, array otherwise).

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> prices = np.array([100, 110, 90, 0])
    >>> # Average price of active firms (price > 0)
    >>> active = prices > 0
    >>> avg_price = ops.mean(prices, where=active)
    >>> # avg_price: 100
    """
    if where is None:
        if axis is None:
            return float(np.mean(a))
        result: Float = np.mean(a, axis=axis)
        return result
    return float(np.mean(a[where]))


# noinspection PyShadowingBuiltins
def any(a: Bool) -> bool:
    """
    Test whether any array element is True.

    Parameters
    ----------
    a : bool array
        Input array.

    Returns
    -------
    bool
        True if any element is True.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> bankrupt = np.array([False, False, True, False])
    >>> has_bankruptcies = ops.any(bankrupt)
    >>> # has_bankruptcies: True
    """
    return bool(np.any(a))


# noinspection PyShadowingBuiltins
def all(a: Bool) -> bool:
    """
    Test whether all array elements are True.

    Parameters
    ----------
    a : bool array
        Input array.

    Returns
    -------
    bool
        True if all elements are True.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> employed = np.array([True, True, True])
    >>> full_employment = ops.all(employed)
    >>> # full_employment: True
    """
    return bool(np.all(a))


# === Array Creation ===


def zeros(n: int) -> Float:
    """
    Create array of zeros.

    Parameters
    ----------
    n : int
        Number of elements.

    Returns
    -------
    array
        Array of n zeros.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> initial_inventory = ops.zeros(100)
    >>> # Array of 100 zeros
    """
    return np.zeros(n, dtype=np.float64)


def ones(n: int) -> Float:
    """
    Create array of ones.

    Parameters
    ----------
    n : int
        Number of elements.

    Returns
    -------
    array
        Array of n ones.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> initial_productivity = ops.ones(100)
    >>> # Array of 100 ones
    """
    return np.ones(n, dtype=np.float64)


def full(n: int, value: float) -> Float:
    """
    Create array filled with constant value.

    Parameters
    ----------
    n : int
        Number of elements.
    value : float
        Fill value.

    Returns
    -------
    array
        Array of n elements, all equal to value.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> initial_price = ops.full(100, 1.5)
    >>> # Array of 100 elements, all 1.5
    """
    return np.full(n, value, dtype=np.float64)


def empty(n: int) -> Float:
    """
    Create uninitialized array (for performance when values will be overwritten).

    Parameters
    ----------
    n : int
        Number of elements.

    Returns
    -------
    array
        Uninitialized array of n elements.

    Notes
    -----
    Use this when you're immediately going to fill the array.
    Values are undefined until assigned.
    """
    return np.empty(n, dtype=np.float64)


# === Utility Operations ===


def unique(a: Union[Float, Int, Agent]) -> Union[Float, Int, Agent]:
    """
    Find unique elements in array.

    Parameters
    ----------
    a : array
        Input array.

    Returns
    -------
    array
        Sorted unique elements.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> suppliers = np.array([1, 3, 2, 1, 3, 3])
    >>> unique_suppliers = ops.unique(suppliers)
    >>> # unique_suppliers: [1, 2, 3]
    """
    result = np.unique(a)
    return result  # type: ignore[return-value]


def bincount(a: Union[Int, Agent], minlength: int = 0) -> Int:
    """
    Count occurrences of each value in array of non-negative ints.

    Parameters
    ----------
    a : int array
        Array of non-negative integers.
    minlength : int, optional
        Minimum length of output array.

    Returns
    -------
    int array
        Count of occurrences of each value.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> # Count how many workers each firm employs
    >>> employer_ids = np.array([0, 0, 1, 2, 0, 1])  # 6 workers
    >>> workers_per_firm = ops.bincount(employer_ids, minlength=5)
    >>> # workers_per_firm: [3, 2, 1, 0, 0]
    >>> # Firm 0 has 3 workers, firm 1 has 2, firm 2 has 1
    """
    return np.bincount(a, minlength=minlength)


def isin(
    element: Union[Float, Int, Agent], test_elements: Union[Float, Int, Agent]
) -> Bool:
    """
    Test whether each element is in test_elements.

    Parameters
    ----------
    element : array
        Input array.
    test_elements : array
        Values to test against.

    Returns
    -------
    bool array
        True where element is in test_elements.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> firm_ids = np.array([0, 1, 2, 3, 4])
    >>> bankrupt_ids = np.array([1, 3])
    >>> is_bankrupt = ops.isin(firm_ids, bankrupt_ids)
    >>> # is_bankrupt: [False, True, False, True, False]
    """
    return np.isin(element, test_elements)


def argsort(a: Float) -> Agent:
    """
    Return indices that would sort the array.

    Parameters
    ----------
    a : array
        Array to sort.

    Returns
    -------
    int array
        Indices that sort the array.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> prices = np.array([30, 10, 20])
    >>> sorted_indices = ops.argsort(prices)
    >>> # sorted_indices: [1, 2, 0]
    >>> # prices[sorted_indices] would be [10, 20, 30]
    """
    return np.argsort(a)


def sort(a: Float) -> Float:
    """
    Return sorted copy of array.

    Parameters
    ----------
    a : array
        Array to sort.

    Returns
    -------
    array
        Sorted array.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> prices = np.array([30, 10, 20])
    >>> sorted_prices = ops.sort(prices)
    >>> # sorted_prices: [10, 20, 30]
    """
    return np.sort(a)


# === Random Operations ===


def uniform(rng: Generator, low: float, high: float, size: int) -> Float:
    """
    Draw samples from uniform distribution.

    Parameters
    ----------
    rng : Generator
        NumPy random number generator (from sim.rng).
    low : float
        Lower bound (inclusive).
    high : float
        Upper bound (exclusive).
    size : int
        Number of samples.

    Returns
    -------
    array
        Random samples.

    Examples
    --------
    >>> import bamengine.ops as ops
    >>> # In custom event:
    >>> def execute(self, sim):
    ...     # Random shocks for each firm
    ...     shocks = ops.uniform(sim.rng, 0.0, 0.1, sim.n_firms)
    """
    return rng.uniform(low, high, size=size)
