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

.. autosummary::
   :toctree: generated
   :nosignatures:

   Float
   Int
   Bool
   AgentId
   Rng
