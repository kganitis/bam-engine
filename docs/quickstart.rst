.. _getting_started:

===============
Getting Started
===============

``BAM Engine`` is a high-performance Python implementation of the BAM (Bottom-Up
Adaptive Macroeconomics) agent-based model. It simulates three types of agents
(households, firms, banks) interacting in three markets (labor, credit,
consumption goods).

The purpose of this guide is to illustrate the main features of ``BAM Engine``.
Please refer to our :ref:`installation instructions <install>` to install
``BAM Engine``, or jump to the :ref:`next_steps_quickstart` section for
additional resources.


Running your first simulation
-----------------------------

``BAM Engine`` provides a simple API for running macroeconomic simulations.
The main entry point is the :class:`~bamengine.Simulation` class, which can be
initialized with :meth:`~bamengine.Simulation.init`:

.. code-block:: python

   >>> import bamengine as bam
   >>> sim = bam.Simulation.init(seed=42)
   >>> sim
   <Simulation: 100 firms, 500 households, 10 banks>

The :meth:`~bamengine.Simulation.init` method accepts various parameters to
configure the simulation. The ``seed`` parameter ensures reproducibility.

Once initialized, run the simulation for a number of periods using
:meth:`~bamengine.Simulation.run`:

.. code-block:: python

   >>> results = sim.run(n_periods=100, collect=True)
   >>> print(f"Final unemployment rate: {sim.ec.unemp_rate_history[-1]:.2%}")
   Final unemployment rate: 4.20%

The simulation executes a sequence of events each period: firms plan production,
workers seek jobs, banks provide credit, goods are produced and sold, and
bankrupt agents are replaced.


Collecting and analyzing results
--------------------------------

The :meth:`~bamengine.Simulation.run` method returns a
:class:`~bamengine.SimulationResults` object containing time series data
collected during the simulation:

.. code-block:: python

   >>> results.n_periods
   100
   >>> results.economy_data["unemployment_rate"]  # array of unemployment rates
   array([0.052, 0.048, 0.044, ...])

If you have ``pandas`` installed, you can export results to DataFrames for
analysis:

.. code-block:: python

   >>> df_economy = results.to_dataframe("economy")
   >>> df_economy[["unemployment_rate", "avg_price"]].head()
      unemployment_rate  avg_price
   0              0.052      1.024
   1              0.048      1.031
   2              0.044      1.028
   ...

You can also export agent-level data:

.. code-block:: python

   >>> df_firms = results.to_dataframe("firms")
   >>> df_households = results.to_dataframe("households")


Configuration
-------------

Simulations can be customized through parameters passed to
:meth:`~bamengine.Simulation.init`:

.. code-block:: python

   >>> sim = bam.Simulation.init(
   ...     n_firms=200,
   ...     n_households=1000,
   ...     n_banks=15,
   ...     seed=42
   ... )

For more complex configurations, you can use a YAML file:

.. code-block:: yaml

   # config.yml
   n_firms: 200
   n_households: 1000
   n_banks: 15
   h_rho: 0.10  # production growth shock cap

.. code-block:: python

   >>> sim = bam.Simulation.init(config="config.yml", seed=42)

Keyword arguments always take precedence over YAML configuration, allowing
you to override specific parameters:

.. code-block:: python

   >>> sim = bam.Simulation.init(config="config.yml", n_firms=200, seed=42)

See the :doc:`Configuration Guide </user_guide/configuration>` for a full
list of parameters.


Accessing simulation state
--------------------------

The simulation maintains agent state in **roles** - data structures holding
arrays of agent attributes. You can access roles directly or via getter methods:

.. code-block:: python

   >>> # Direct access via shortcuts
   >>> sim.prod.price       # firm prices (Producer role)
   >>> sim.wrk.wage         # worker wages (Worker role)
   >>> sim.lend.equity_base # bank equity (Lender role)

   >>> # Or via getter method
   >>> producer = sim.get_role("Producer")
   >>> producer.price
   array([1.02, 0.98, 1.05, ...])

Available role shortcuts:

- ``sim.prod`` - Producer (firms): prices, inventory, production
- ``sim.emp`` - Employer (firms): wages, labor, vacancies
- ``sim.bor`` - Borrower (firms): net worth, debt, credit demand
- ``sim.wrk`` - Worker (households): wages, employment status
- ``sim.con`` - Consumer (households): income, savings
- ``sim.lend`` - Lender (banks): equity, interest rates, credit supply

Economy-wide metrics are available through ``sim.ec``:

.. code-block:: python

   >>> sim.ec.avg_mkt_price      # average market price
   >>> sim.ec.unemp_rate_history # unemployment rate time series


.. _next_steps_quickstart:

Next steps
----------

This guide covered the basics of running simulations, collecting results,
configuring parameters, and accessing agent state. There is much more to
``BAM Engine``!

- :doc:`User Guide </user_guide/index>` - Detailed documentation on all features
- :doc:`Examples </auto_examples/index>` - Worked examples demonstrating various use cases
- :doc:`API Reference </api/index>` - Complete API documentation
