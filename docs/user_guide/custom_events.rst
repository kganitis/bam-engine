Custom Events
=============

.. note::

   This section is under construction. See the :doc:`examples </auto_examples/index>`
   for custom event usage patterns.

Events are functions that modify agent state during simulation.

Quick Example
-------------

.. code-block:: python

   from bamengine import event, ops, Simulation


   @event
   class CustomPricing:
       """Apply markup pricing to all producers."""

       def execute(self, sim: Simulation) -> None:
           prod = sim.get_role("Producer")
           emp = sim.get_role("Employer")

           # Calculate unit labor cost
           unit_cost = ops.divide(emp.wage_offered, prod.labor_prod)

           # Apply 50% markup
           new_price = ops.multiply(unit_cost, 1.5)

           # Update prices in-place
           ops.assign(prod.price, new_price)

Topics to be covered:

* Event decorator syntax
* The execute() method
* Using the ops module
* Accessing roles and economy state
* Event registration and pipeline integration
