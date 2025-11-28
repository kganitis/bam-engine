Custom Roles
============

.. note::

   This section is under construction. See the :doc:`examples </auto_examples/index>`
   for custom role usage patterns.

Roles are data containers that hold agent state as NumPy arrays.

Quick Example
-------------

.. code-block:: python

   from bamengine import role
   from bamengine.typing import Float, Int

   @role
   class Inventory:
       """Custom inventory role for agents."""
       goods_on_hand: Float
       reorder_point: Float
       days_until_delivery: Int

Topics to be covered:

* Role decorator syntax
* Type aliases (Float, Int, Bool, AgentId)
* Optional scratch buffers
* Role registration and retrieval
