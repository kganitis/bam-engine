Overview
========

.. note::

   This section is under construction.

BAM Engine is a high-performance Python framework for agent-based macroeconomic
simulation, implementing the BAM (Bottom-Up Adaptive Macroeconomics) model.

Architecture
------------

BAM Engine uses an Entity-Component-System (ECS) inspired architecture:

- **Agents**: Lightweight entities (firms, households, banks) identified by type and ID
- **Roles**: Data containers holding agent state as NumPy arrays (Producer, Worker, Lender, etc.)
- **Events**: Functions that modify agent state during simulation steps
- **Relationships**: Many-to-many connections between agents (e.g., loans between firms and banks)
- **Pipeline**: Ordered sequence of events executed each simulation period

This design enables high performance through vectorized operations while maintaining
modularity and extensibility.

Topics to be covered:

* ECS architecture overview
* Agent types and their roles
* The simulation loop
* Registry and auto-registration
* Extending the framework
