Type System
===========

.. automodule:: bamengine.typing
   :no-members:

.. currentmodule:: bamengine

Type Aliases
------------

The following type aliases are available for defining custom roles without needing to import NumPy directly:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Description
   * - :class:`Float`
     - Array of floating-point values (prices, quantities, rates, etc.)
   * - :class:`Int`
     - Array of integer values (counts, periods, durations, etc.)
   * - :class:`Bool`
     - Array of boolean values (flags, conditions, masks)
   * - :class:`AgentId`
     - Array of agent IDs (integer indices, -1 for unassigned)
   * - :class:`Rng`
     - Random number generator (alias for :class:`numpy.random.Generator`)

Detailed Reference
------------------

.. data:: bamengine.Float
   :type: type alias
   :value: NDArray[np.float64]

   1-D array of 64-bit floats. Used for prices, quantities, rates, and other
   continuous values.

.. data:: bamengine.Int
   :type: type alias
   :value: NDArray[np.int64]

   1-D array of 64-bit integers. Used for counts, periods, durations, and other
   discrete values.

.. data:: bamengine.Bool
   :type: type alias
   :value: NDArray[np.bool\_]

   1-D array of booleans. Used for flags, conditions, and masks.

.. data:: bamengine.AgentId
   :type: type alias
   :value: NDArray[np.intp]

   1-D array of agent IDs (platform-native integers). Convention: ``-1`` means
   unassigned.

.. data:: bamengine.Rng
   :type: type alias
   :value: numpy.random.Generator

   Random number generator instance. All simulation randomness flows through
   this type to ensure deterministic reproducibility.
