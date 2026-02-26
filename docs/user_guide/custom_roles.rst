Custom Roles
============

Roles are the data layer of BAM Engine's ECS architecture. Each role is a
dataclass-like container where every field is a NumPy array indexed by agent ID.
Custom roles let you add new state variables to agents without modifying the
core framework.


Quick Example
-------------

.. code-block:: python

   from bamengine import role
   from bamengine.typing import Float, Int


   @role
   class Inventory:
       """Track physical inventory for firms."""

       goods_on_hand: Float
       reorder_point: Float
       days_until_delivery: Int


The ``@role`` Decorator
-----------------------

The ``@role`` decorator transforms a class definition into a role:

.. code-block:: python

   from bamengine import role
   from bamengine.typing import Float


   @role
   class RnDCapability:
       rd_intensity: Float
       patents_held: Float

The decorator:

1. Converts the class into a dataclass with ``slots=True``
2. Registers it in the global role registry
3. Derives the role name from the class name (``"RnDCapability"``)

You can override the name:

.. code-block:: python

   @role(name="R&D")
   class RnDCapability:
       rd_intensity: Float


Type Aliases
------------

Role fields must use BAM Engine's type aliases, which map to NumPy dtypes:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Alias
     - NumPy dtype
     - Typical Use
   * - ``Float``
     - ``np.float64``
     - Prices, wages, production, net worth — any continuous value
   * - ``Int``
     - ``np.int64``
     - Contract duration, vacancy count — discrete quantities
   * - ``Bool``
     - ``np.bool_``
     - Employment status, bankruptcy flags — binary state
   * - ``AgentId``
     - ``np.intp``
     - Employer ID, bank ID — references to other agents

Import from ``bamengine.typing``:

.. code-block:: python

   from bamengine.typing import Float, Int, Bool, AgentId


Initializing Custom Roles
--------------------------

After creating a simulation, attach your role with
:meth:`~bamengine.Simulation.use_role`:

.. code-block:: python

   import bamengine as bam

   sim = bam.Simulation.init(seed=42)

   # Firm-level role (default: n_agents = n_firms)
   sim.use_role(RnDCapability)

   # Household-level role (specify agent count)
   sim.use_role(BufferStock, n_agents=sim.n_households)

   # Bank-level role
   sim.use_role(BankRegulation, n_agents=sim.n_banks)

All fields are initialized to zero by default.


Accessing Role Data
-------------------

Retrieve a role by name and access its fields as NumPy arrays:

.. code-block:: python

   rnd = sim.get_role("RnDCapability")

   rnd.rd_intensity  # shape (n_firms,) — array of floats
   rnd.patents_held  # shape (n_firms,) — array of floats
   rnd.rd_intensity[0]  # scalar — first firm's R&D intensity

Built-in roles have shortcut attributes (see :doc:`running_simulations`):

.. code-block:: python

   sim.prod.price  # equivalent to sim.get_role("Producer").price
   sim.wrk.employer  # equivalent to sim.get_role("Worker").employer


Built-in Roles
--------------

BAM Engine provides 7 built-in roles across the three agent types:

**Firm Roles**

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Role
     - Key Fields
   * - :class:`~bamengine.roles.producer.Producer`
     - ``price``, ``production``, ``production_prev``, ``inventory``, ``desired_production``,
       ``expected_demand``, ``labor_productivity``, ``breakeven_price``
   * - :class:`~bamengine.roles.employer.Employer`
     - ``desired_labor``, ``current_labor``, ``wage_offer``, ``wage_bill``,
       ``n_vacancies``, ``total_funds``
   * - :class:`~bamengine.roles.borrower.Borrower`
     - ``net_worth``, ``total_funds``, ``credit_demand``, ``projected_fragility``,
       ``gross_profit``, ``net_profit``, ``retained_profit``, ``wage_bill``

**Household Roles**

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Role
     - Key Fields
   * - :class:`~bamengine.roles.worker.Worker`
     - ``employer``, ``wage``, ``periods_left``, ``contract_expired``, ``fired``,
       ``.employed`` (computed property)
   * - :class:`~bamengine.roles.consumer.Consumer`
     - ``income``, ``savings``, ``income_to_spend``, ``propensity``,
       ``largest_prod_prev``
   * - :class:`~bamengine.roles.shareholder.Shareholder`
     - ``dividends`` (per-period dividend income, reset each period)

**Bank Roles**

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Role
     - Key Fields
   * - :class:`~bamengine.roles.lender.Lender`
     - ``equity_base``, ``credit_supply``, ``interest_rate``


Tips
----

- **Fields must be annotated**: Every field needs a type annotation (``Float``,
  ``Int``, ``Bool``, or ``AgentId``). Unannotated attributes are ignored.
- **Don't subclass existing roles**: Create new roles instead. The ECS pattern
  favors composition over inheritance.
- **Roles are dataclasses**: Under the hood, roles use ``@dataclass(slots=True)``.
  This means you get efficient attribute access and memory layout.
- **Zero initialization**: All fields start as zero-filled arrays. If you need
  non-zero defaults, set them in a custom event that runs at the start of the
  pipeline.


.. seealso::

   - :doc:`overview` for the ECS architecture
   - :doc:`custom_events` for using role data in events
   - :doc:`extensions` for examples of roles in built-in extensions
