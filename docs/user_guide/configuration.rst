Configuration
=============

BAM Engine uses a three-tier configuration system that balances sensible defaults
with full customizability. This page covers the configuration precedence rules,
all available parameters, YAML file format, and logging configuration.


Configuration Precedence
------------------------

Parameters are resolved in order of increasing priority:

1. **Package defaults** — built-in values from ``bamengine/config/defaults.yml``
2. **User YAML file** — custom configuration file (if provided)
3. **Keyword arguments** — passed directly to ``Simulation.init()`` (highest priority)

.. code-block:: python

   import bamengine as bam

   # 1. Defaults only
   sim = bam.Simulation.init(seed=42)

   # 2. YAML file overrides defaults
   sim = bam.Simulation.init(config="my_config.yml", seed=42)

   # 3. kwargs override both defaults and YAML
   sim = bam.Simulation.init(config="my_config.yml", n_firms=300, seed=42)


YAML Configuration Files
-------------------------

Create a YAML file with any parameters you want to override:

.. code-block:: yaml

   # my_config.yml
   n_firms: 200
   n_households: 1000
   n_banks: 15

   h_rho: 0.15
   theta: 6
   delta: 0.15

   max_M: 6
   max_H: 3
   max_Z: 3

   logging:
     default_level: WARNING

Pass the file path to ``Simulation.init()``:

.. code-block:: python

   sim = bam.Simulation.init(config="my_config.yml", seed=42)

Only include parameters you want to change — omitted parameters use defaults.


Parameter Reference
-------------------

Population Sizes
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Parameter
     - Default
     - Description
   * - ``n_firms``
     - 100
     - Number of firms (Producer/Employer/Borrower agents). Determines production
       sector size and labor demand.
   * - ``n_households``
     - 500
     - Number of households (Worker/Consumer/Shareholder agents). Recommended:
       :math:`\geq 5 \times` ``n_firms`` for realistic labor markets.
   * - ``n_banks``
     - 10
     - Number of banks (Lender agents). Controls credit market diversity.
   * - ``n_periods``
     - 1000
     - Default number of simulation periods when ``n_periods`` is not passed
       to ``sim.run()``.

Stochastic Shocks
~~~~~~~~~~~~~~~~~

All shocks are drawn from uniform distributions :math:`U(0, h)` where :math:`h`
is the half-width parameter. One period represents one quarter.

.. list-table::
   :header-rows: 1
   :widths: 18 12 70

   * - Parameter
     - Default
     - Description
   * - ``h_rho``
     - 0.10
     - Production growth shock cap. Controls how aggressively firms adjust
       production targets. Also determines the quantization trap threshold
       (see :doc:`bam_model`).
   * - ``h_xi``
     - 0.05
     - Wage growth shock cap. Controls the maximum wage increase when a firm
       has vacancies.
   * - ``h_phi``
     - 0.10
     - Bank operating expense shock cap. Affects the spread between the
       policy rate and bank lending rates.
   * - ``h_eta``
     - 0.10
     - Price adjustment shock cap. Controls how fast firms adjust prices in
       response to market conditions.

Search Frictions
~~~~~~~~~~~~~~~~

Control how many partners agents can contact per period. Higher values reduce
frictions but increase the number of pipeline events (matching rounds).

.. list-table::
   :header-rows: 1
   :widths: 18 12 70

   * - Parameter
     - Default
     - Description
   * - ``max_M``
     - 4
     - Job applications per unemployed worker per period. Each round, workers
       send one application and firms hire from their queue.
   * - ``max_H``
     - 2
     - Loan applications per firm per period. Firms contact banks sorted by
       interest rate (lowest first).
   * - ``max_Z``
     - 2
     - Shops a consumer can visit per period. Consumers visit their loyalty
       firm first, then random firms.

Structural Parameters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Parameter
     - Default
     - Description
   * - ``labor_productivity``
     - 0.50
     - Goods produced per worker per period (:math:`\varphi`). Constant in
       the baseline model; endogenous in the Growth+ extension.
   * - ``theta``
     - 8
     - Employment contract length in periods (:math:`\theta`). Workers are
       locked into their current employer for this many periods.
   * - ``beta``
     - 2.50
     - Consumption propensity exponent (:math:`\beta`). Controls how strongly
       relative savings affect the marginal propensity to consume.
   * - ``delta``
     - 0.10
     - Dividend payout ratio (:math:`\delta`). Fraction of positive net
       profit paid as dividends to households.
   * - ``v``
     - 0.10
     - Bank capital requirement coefficient (:math:`\nu`). Maximum bank
       leverage is :math:`1/\nu = 10`.
   * - ``r_bar``
     - 0.02
     - Baseline (policy) interest rate (:math:`\bar{r}`). Banks add a markup
       with random shock.
   * - ``min_wage_rev_period``
     - 4
     - Periods between minimum wage revisions. The minimum wage is adjusted
       for inflation every this many periods.

Initial Conditions
~~~~~~~~~~~~~~~~~~

Balance sheet values at :math:`t = 0`. Can be scalars (broadcast to all agents).

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Parameter
     - Default
     - Description
   * - ``price_init``
     - 0.5
     - Initial goods price for all firms.
   * - ``min_wage_ratio``
     - 0.5
     - Initial minimum wage as a fraction of the mean initial wage.
   * - ``net_worth_ratio``
     - 6.0
     - Initial firm net worth as a multiple of initial revenue.
   * - ``equity_base_init``
     - 5.0
     - Initial bank equity.
   * - ``savings_init``
     - 1.0
     - Initial household savings.

New Firm Entry
~~~~~~~~~~~~~~

Control how replacement firms are initialized after bankruptcy. New firms inherit
attributes scaled from the trimmed mean (90th percentile) of survivors.

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Description
   * - ``new_firm_size_factor``
     - 0.50
     - Net worth of new firm as fraction of survivor average.
   * - ``new_firm_production_factor``
     - 0.50
     - Production capacity of new firm as fraction of survivor average.
   * - ``new_firm_wage_factor``
     - 0.50
     - Wage offer of new firm as fraction of survivor average.
   * - ``new_firm_price_markup``
     - 1.15
     - Price of new firm as markup over average market price.

Implementation Variants
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Parameter
     - Default
     - Description
   * - ``max_loan_to_net_worth``
     - 2
     - Maximum individual loan as a multiple of borrower's net worth.
   * - ``max_leverage``
     - 10
     - Cap on the financial fragility metric for interest rate calculation.
   * - ``job_search_method``
     - ``"all_firms"``
     - How workers sample firms: ``"all_firms"`` (sample from all) or
       ``"vacancies_only"`` (only firms with openings).
   * - ``consumer_matching``
     - ``"loyalty"``
     - Consumer firm selection: ``"loyalty"`` (preferential attachment) or
       ``"random"`` (no loyalty).

Other
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Parameter
     - Default
     - Description
   * - ``seed``
     - 0
     - Random number generator seed. Use ``None`` for non-deterministic.
   * - ``pipeline_path``
     - ``None``
     - Path to a custom pipeline YAML file. See :doc:`pipelines`.


Logging Configuration
---------------------

Control log verbosity globally or per-event:

.. code-block:: python

   sim = bam.Simulation.init(
       logging={
           "default_level": "WARNING",  # Global level
           "log_file": "simulation.log",  # Optional file output
           "events": {
               "firms_hire_workers": "DEBUG",  # Verbose for one event
               "workers_send_one_round": "ERROR",  # Suppress another
           },
       },
       seed=42,
   )

Available log levels (from most to least verbose):

- ``TRACE`` (5) — Very detailed internal state
- ``DEBUG`` (10) — Diagnostic information
- ``INFO`` (20) — General progress (default)
- ``WARNING`` (30) — Potential issues
- ``ERROR`` (40) — Errors only
- ``CRITICAL`` (50) — Fatal errors only

.. tip::

   Setting ``default_level`` to ``"ERROR"`` or ``"WARNING"`` for production runs
   provides a 10-20% speedup by avoiding log message formatting overhead.

YAML logging configuration:

.. code-block:: yaml

   # In your config YAML file
   logging:
     default_level: WARNING
     log_file: output/simulation.log
     events:
       firms_adjust_price: TRACE
       firms_hire_workers: DEBUG


Extension Parameters
--------------------

Any keyword argument passed to ``Simulation.init()`` that is not a core
configuration parameter is stored in ``extra_params`` and accessible as an
attribute on the simulation object:

.. code-block:: python

   # Pass custom parameters for an extension
   sim = bam.Simulation.init(
       seed=42,
       sigma_min=0.0,  # R&D extension parameter
       sigma_max=0.1,
       sigma_decay=-1.0,
   )

   # Access in custom events
   sigma_min = sim.sigma_min  # Direct attribute access
   sim.extra_params["sigma_min"]  # Also available via dict

This enables model extensions to define their own parameters without modifying
core configuration code. See :doc:`extensions` for the built-in extensions and
their parameters.

.. tip::

   Extensions typically export a ``*_CONFIG`` dictionary with default values.
   Use :meth:`~bamengine.Simulation.use_config` to apply them:

   .. code-block:: python

      from extensions.rnd import RND_CONFIG

      sim.use_config(RND_CONFIG)


Configuration Validation
------------------------

BAM Engine validates all parameters at initialization time. Invalid values
raise descriptive errors:

- **Type validation**: Parameters must match expected types (int, float, str)
- **Range validation**: Numeric parameters must be within valid ranges
  (e.g., ``n_firms > 0``, ``0 < delta < 1``)
- **Relationship validation**: Cross-parameter constraints are checked
  (e.g., ``n_households >= n_firms``)
- **Pipeline validation**: If ``pipeline_path`` is set, the pipeline file must
  exist and contain valid event names

.. code-block:: python

   # These will raise ValueError with descriptive messages
   sim = bam.Simulation.init(n_firms=-1)  # Range error
   sim = bam.Simulation.init(delta=2.0)  # Range error
   sim = bam.Simulation.init(n_firms="abc")  # Type error


.. seealso::

   - :doc:`bam_model` for the economic meaning of each parameter
   - :doc:`running_simulations` for initialization patterns
   - :doc:`extensions` for extension-specific parameters
