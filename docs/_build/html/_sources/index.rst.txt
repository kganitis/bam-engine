.. BAM Engine documentation master file

BAM Engine Documentation
=========================

BAM Engine is a high-performance Python implementation of the BAM (Bottom-Up Adaptive Macroeconomics)
agent-based model, part of the CATS (Complex Adaptive Trivial Systems) family of macroeconomic models.

The model simulates three types of agents (households, firms, banks) interacting in three markets
(labor, credit, consumption goods) using a modern Entity-Component-System (ECS) architecture.

Quick Start
-----------

Install BAM Engine:

.. code-block:: bash

   pip install bamengine

Run a basic simulation:

.. code-block:: python

   import bamengine as bam

   # Initialize simulation
   sim = bam.Simulation.init(n_firms=100, n_households=500, seed=42)

   # Run for 100 periods
   results = sim.run(n_periods=100)

   # Export results to DataFrame
   df = results.to_dataframe()

Features
--------

- **High Performance**: Fully vectorized NumPy operations, 4-9x faster than reference implementations
- **Modular Design**: ECS architecture allows easy extension with custom roles, events, and relationships
- **Type Safe**: Comprehensive type hints with py.typed marker
- **Well Tested**: 99%+ test coverage with unit, integration, property-based, and performance tests
- **User Friendly**: Simple API with pandas integration for data analysis

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
