Simulation
==========

.. automodule:: bamengine.simulation
   :no-members:

The ``bamengine.simulation`` module provides the main interface for running
BAM simulations.

.. currentmodule:: bamengine

Classes
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Simulation
   SimulationResults

Key Methods
-----------

**Simulation Methods**

.. autosummary::
   :toctree: generated
   :nosignatures:

   Simulation.init
   Simulation.run
   Simulation.step
   Simulation.get_role
   Simulation.get_event
   Simulation.get_relationship
   Simulation.use_role

**Results Methods**

.. autosummary::
   :toctree: generated
   :nosignatures:

   SimulationResults.to_dataframe
   SimulationResults.get_role_data
   SimulationResults.get_array
   SimulationResults.data
   SimulationResults.economy_metrics
   SimulationResults.summary
