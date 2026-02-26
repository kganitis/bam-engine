Overview
========

What is BAM Engine?
-------------------

BAM Engine is a high-performance Python framework for agent-based macroeconomic
simulation. It implements the **BAM (Bottom-Up Adaptive Macroeconomics)** model,
part of the **CATS (Complex Adaptive Trivial Systems)** family of models originally
described in *Macroeconomics from the Bottom-up* by Delli Gatti, Gaffeo, Gallegati,
Giulioni, and Palestrini (2011).

The model simulates an economy populated by three types of agents — firms,
households, and banks — interacting through decentralized markets for labor,
credit, and consumption goods. Unlike traditional DSGE models, BAM uses
**bounded rationality** and **adaptive behavior**: agents follow simple heuristic
rules rather than solving optimization problems, and macroeconomic patterns
(business cycles, unemployment dynamics, firm size distributions) emerge from
the bottom up through agent interactions.

BAM Engine brings this model to Python with a design focused on performance,
modularity, and extensibility. All agent state is stored as NumPy arrays and
operations are fully vectorized, making simulations fast enough for large-scale
parameter sweeps and calibration studies.


Architecture
------------

BAM Engine uses an **Entity-Component-System (ECS)** inspired architecture,
adapted for agent-based economic modeling:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - ECS Concept
     - BAM Engine
     - Description
   * - Entity
     - Agent
     - A lightweight identifier (integer ID) for a firm, household, or bank
   * - Component
     - Role
     - A data container holding agent state as parallel NumPy arrays (e.g., ``Producer``, ``Worker``, ``Lender``)
   * - System
     - Event
     - A function that reads and modifies role data each simulation period (e.g., ``FirmsAdjustPrice``)
   * - —
     - Relationship
     - A sparse many-to-many connection between agents with edge data (e.g., ``LoanBook``)
   * - —
     - Pipeline
     - An ordered sequence of events executed each period

**Why ECS?** Traditional object-oriented agent models store state on individual
agent objects, leading to scattered memory access and Python-loop iteration.
The ECS pattern stores all agents of the same type in contiguous NumPy arrays,
enabling vectorized operations across entire populations in a single call.
This gives BAM Engine near-C performance while keeping the code readable and
modular.

The architecture separates **data** (roles) from **behavior** (events), so you
can add new agent attributes or behavioral rules without touching existing code.


Key Concepts
------------

Agents
~~~~~~

The BAM economy contains three agent types, each composed of multiple roles:

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Agent Type
     - Roles
     - Description
   * - Firms
     - :class:`~bamengine.roles.producer.Producer`, :class:`~bamengine.roles.employer.Employer`, :class:`~bamengine.roles.borrower.Borrower`
     - Produce goods, hire workers, borrow from banks
   * - Households
     - :class:`~bamengine.roles.worker.Worker`, :class:`~bamengine.roles.consumer.Consumer`, :class:`~bamengine.roles.shareholder.Shareholder`
     - Work for firms, consume goods, receive dividends
   * - Banks
     - :class:`~bamengine.roles.lender.Lender`
     - Supply credit to firms, earn interest income

Each agent is identified by an integer index (0 to N-1) shared across all its
roles. For example, firm 7's production data is at ``Producer.production[7]``,
its wage offer at ``Employer.wage_offer[7]``, and its net worth at
``Borrower.net_worth[7]``.

Roles
~~~~~

Roles are **dataclass-like containers** where each field is a NumPy array indexed
by agent ID. They hold all agent state:

.. code-block:: python

   prod = sim.get_role("Producer")
   prod.price  # shape (n_firms,) — current price per firm
   prod.inventory  # shape (n_firms,) — unsold goods per firm

Roles are defined with the ``@role`` decorator and type annotations:

.. code-block:: python

   from bamengine import role
   from bamengine.typing import Float, Int


   @role
   class Inventory:
       goods_on_hand: Float
       reorder_point: Float
       days_until_delivery: Int

See :doc:`custom_roles` for the full guide.

Events
~~~~~~

Events are the **behavioral rules** of the simulation. Each event reads role
data, performs calculations, and writes results back — typically using the
:mod:`~bamengine.ops` module for safe, vectorized operations:

.. code-block:: python

   from bamengine import event, ops


   @event(after="firms_collect_revenue")
   class FirmsCalcBonus:
       def execute(self, sim):
           bor = sim.get_role("Borrower")
           bonus = ops.where(bor.net_profit > 0, ops.multiply(bor.net_profit, 0.05), 0)
           ops.assign(bor.total_funds, ops.add(bor.total_funds, bonus))

Events are executed in a fixed order defined by the pipeline. See :doc:`custom_events`
for the full guide.

Relationships
~~~~~~~~~~~~~

Relationships represent **many-to-many connections** between agents, stored
in a sparse COO (Coordinate) format. The built-in ``LoanBook`` tracks loans
between firms and banks:

.. code-block:: python

   loans = sim.get_relationship("LoanBook")
   loans.principal  # principal amount per active loan
   loans.rate  # interest rate per active loan
   loans.debt_per_borrower(n_borrowers=sim.n_firms)  # total debt per firm

See :doc:`custom_relationships` for the full guide.

Pipeline
~~~~~~~~

The **pipeline** defines the execution order of events within each simulation
period. BAM Engine uses **explicit ordering** (not dependency-based topological
sort) to match the original model specification:

.. code-block:: python

   # Inspect the pipeline
   for entry in sim.pipeline:
       print(entry.name)

The default pipeline has 8 phases: Planning, Labor Market, Credit Market,
Production, Goods Market, Revenue, Bankruptcy, and Entry. See :doc:`pipelines`
for customization options.

The Simulation Loop
~~~~~~~~~~~~~~~~~~~

Each call to ``sim.step()`` executes one period of the economy:

1. **Planning** — Firms set production targets and labor needs
2. **Labor Market** — Workers search for jobs, firms hire
3. **Credit Market** — Firms borrow from banks to finance production
4. **Production** — Firms pay wages and produce goods
5. **Goods Market** — Households shop for consumption goods
6. **Revenue** — Firms collect revenue, repay debts, pay dividends
7. **Bankruptcy** — Insolvent firms and banks exit
8. **Entry** — Replacement firms and banks are created

See :doc:`bam_model` for the full economic detail of each phase.


Auto-Registration
-----------------

BAM Engine uses Python decorators that **automatically register** components
in a global registry:

- ``@role`` registers a new role class
- ``@event`` registers a new event class
- ``@relationship`` registers a new relationship class

Registration happens at class definition time (via ``__init_subclass__``), so
importing a module is enough to make its components available:

.. code-block:: python

   from bamengine import get_role, get_event, get_relationship

   # Look up registered components by name
   Producer = get_role("Producer")
   evt_cls = get_event("firms_adjust_price")
   LoanBook = get_relationship("LoanBook")

This means extensions just need to be imported before ``sim.run()`` — no
explicit registration calls needed. However, **pipeline hooks** (``after=``,
``before=``, ``replace=``) still require explicit activation via
:meth:`~bamengine.Simulation.use_events`.


Deterministic RNG
-----------------

All randomness in BAM Engine flows through a single ``numpy.random.Generator``
seeded at initialization:

.. code-block:: python

   sim = bam.Simulation.init(seed=42)
   # sim.rng is a numpy.random.Generator seeded with 42

This guarantees **perfect reproducibility**: the same seed always produces the
same simulation trajectory, regardless of platform or Python version. Custom
events should always use ``sim.rng`` rather than ``numpy.random`` directly:

.. code-block:: python

   # Correct — deterministic
   shock = sim.rng.uniform(0, 0.1, size=sim.n_firms)

   # Wrong — breaks reproducibility
   shock = np.random.uniform(0, 0.1, size=sim.n_firms)


What's Next
-----------

.. list-table::
   :widths: 30 70

   * - :doc:`bam_model`
     - Understand the economic theory and mathematical rules behind each phase
   * - :doc:`running_simulations`
     - Learn how to initialize and run simulations
   * - :doc:`/quickstart`
     - Jump straight into a working example
   * - :doc:`configuration`
     - Configure parameters and customize behavior
   * - :doc:`extensions`
     - Explore built-in model extensions (R&D, buffer-stock, taxation)
