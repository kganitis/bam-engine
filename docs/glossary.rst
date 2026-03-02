Glossary
========

This glossary defines key terms used throughout the BAM Engine documentation.
Terms listed here can be cross-referenced from any page using the ``:term:`` role.


Economic Concepts
-----------------

.. glossary::
   :sorted:

   agent
      An autonomous decision-making entity in the model. BAM Engine has three
      agent types: :term:`firms <firm>`, :term:`households <household>`, and
      :term:`banks <bank>`.

   firm
      An agent that produces consumption goods, hires workers, and borrows from
      banks. Each firm has three roles: :term:`Producer`, :term:`Employer`, and
      :term:`Borrower`.

   household
      An agent that supplies labor, consumes goods, and receives dividends.
      Each household has three roles: :term:`Worker`, :term:`Consumer`, and
      :term:`Shareholder`.

   bank
      An agent that supplies credit to firms, sets interest rates, and absorbs
      bad debt losses. Each bank has the :term:`Lender` role.

   labor market
      The market where firms post vacancies and unemployed workers search for
      jobs. Matching uses :term:`preferential attachment` with configurable
      rounds (``max_M``).

   credit market
      The market where firms apply for loans and banks supply credit based on
      financial fragility assessments. Matching uses configurable rounds
      (``max_H``).

   goods market
      The market where consumers purchase consumption goods from firms. Also
      called the consumption goods market. Matching uses
      :term:`preferential attachment` with configurable rounds (``max_Z``).

   desired production
      The quantity of goods a firm plans to produce, based on past production
      adjusted by a demand signal. Drives hiring and borrowing decisions.

   actual production
      The quantity of goods actually produced, constrained by available labor
      and the :term:`production function`.

   production function
      A linear function relating output to labor input:
      ``Y = phi * L`` where ``phi`` is :term:`labor productivity` and ``L``
      is the number of workers.

   labor productivity
      Goods produced per worker (``phi``). In the base model this is constant;
      the Growth+ extension makes it endogenous through R&D investment.

   markup
      The multiplicative factor applied above :term:`breakeven price` to set
      the firm's selling price. Adjusted adaptively each period based on
      inventory levels.

   breakeven price
      The minimum price at which a firm covers its costs. Calculated as
      ``wage_bill / expected_output`` (planning phase) or
      ``wage_bill / (labor_productivity * labor)`` (production phase).

   financial fragility
      A measure of a firm's leverage, defined as the ratio of debt to net
      worth. Used by banks to rank loan applications and set interest rates.
      Related to the :term:`Minsky classification`.

   net worth
      A firm's or bank's equity: assets minus liabilities. For firms,
      accumulates through retained profits. Negative net worth triggers
      :term:`bankruptcy`.

   wage bill
      Total wages a firm must pay its workers in a given period. Equals the
      sum of individual worker wages.

   inventory
      Unsold goods carried over between periods. Affects pricing decisions
      through the markup adjustment rule.

   consumption budget
      The amount a household plans to spend on goods in a period, determined
      by the :term:`propensity to consume` and available wealth.

   propensity to consume
      The fraction of wealth a household allocates to consumption. In the
      base model this is uniform; the :term:`buffer stock` extension makes
      it adaptive.

   reservation wage
      The minimum wage a worker will accept for employment. Adjusts
      downward during unemployment and upward during employment.

   buffer stock
      A consumption theory (Carroll, 1997) where households target a wealth-to-income
      ratio. The buffer-stock extension (Section 3.9.4) replaces the uniform
      propensity to consume with individual adaptive MPCs.

   credit supply
      The maximum amount a bank can lend, determined by its equity and
      regulatory constraints (capital adequacy ratio ``v``).

   interest rate
      The rate charged on loans, composed of a base rate plus a risk premium
      that increases with borrower :term:`financial fragility`. Each bank
      also applies an idiosyncratic operational shock.

   bad debt
      Losses incurred by banks when borrowers default. Computed as
      ``principal - recovery``, where recovery depends on the bankrupt
      firm's remaining net worth.

   maximum leverage
      The upper bound on a firm's debt-to-equity ratio, beyond which banks
      refuse to lend. Configured via ``max_leverage``.

   preferential attachment
      A matching mechanism where agents are more likely to select partners
      they have interacted with before (loyalty effect). Used in both the
      :term:`labor market` and :term:`goods market`.

   matching
      The process of pairing agents in a market. BAM Engine supports
      interleaved matching (agents processed one at a time across multiple
      rounds) and cascade matching (all agents processed per round).

   vacancy
      An open position posted by a firm seeking to hire workers. The number
      of vacancies equals the gap between desired and current labor.

   GDP
      Gross Domestic Product. Computed as the sum of firm revenues
      (price times quantity sold) across all firms.

   unemployment
      The fraction of the labor force without employment. A key emergent
      property of the model, driven by hiring, firing, and bankruptcy dynamics.

   inflation
      The rate of change in the :term:`price index` between periods. An
      emergent property arising from individual firm pricing decisions.

   price index
      The production-weighted average of firm prices. Used to compute
      :term:`inflation`.

   business cycle
      Endogenous fluctuations in aggregate output, employment, and prices
      that emerge from agent interactions without exogenous shocks.

   Kalecki profit equation
      The macroeconomic identity showing that aggregate profits are
      guaranteed by the circular flow of income. In BAM, this creates the
      :term:`Kalecki trap` where firms self-finance and stop borrowing.

   Kalecki trap
      A structural issue where high net-worth-to-wage-bill ratios eliminate
      borrowing demand, causing credit market inactivity. Only the dividend
      payout rate (``delta``) is an effective parameter lever.

   Minsky classification
      Categorization of firms by financial fragility: **hedge** (cash flow
      covers all obligations), **speculative** (covers interest but not
      principal), **Ponzi** (cannot cover interest). Based on Hyman Minsky's
      Financial Instability Hypothesis.

   HP filter
      Hodrick-Prescott filter. A statistical method for decomposing a time
      series into trend and cyclical components. Used in robustness analysis
      to extract business cycle fluctuations.

   cross-correlation
      A measure of similarity between two time series as a function of lag.
      Used in robustness analysis to verify co-movement structure between
      macroeconomic variables.

   bankruptcy
      The exit of a firm or bank from the market when its :term:`net worth`
      becomes negative. Bankrupt agents are replaced by new entrants with
      initial conditions set by configuration parameters.

   scenario
      A specific model configuration and set of validation targets
      corresponding to a section of the reference book. BAM Engine has
      three scenarios: baseline (3.9.1), Growth+ (3.9.2), and buffer-stock
      (3.9.4).

   dividend
      The fraction of net profit distributed to shareholders each period,
      controlled by the dividend payout rate ``delta``.

   operational shock
      A bank-specific idiosyncratic cost shock (``phi_k``) that affects
      the interest rate charged on loans, creating heterogeneity among banks.

   quantization trap
      A structural issue in the labor market where the ceiling function in
      the hiring rule creates a one-way ratchet: small firms can increase
      but never decrease their workforce.


Framework Concepts
------------------

.. glossary::
   :sorted:

   ECS
      Entity-Component-System. An architectural pattern where entities are
      lightweight IDs, components hold data (see :term:`role`), and systems
      contain behavior (see :term:`event`). BAM Engine adapts this pattern
      from game development for agent-based modeling.

   role
      A component in the :term:`ECS` architecture. A dataclass whose fields
      are parallel NumPy arrays, one element per agent. Defined with the
      ``@role`` decorator. Examples: :term:`Producer`, :term:`Worker`,
      :term:`Lender`.

   Producer
      The firm role holding production-related state: price, markup,
      production levels, inventory, net worth, and financial metrics.

   Employer
      The firm role holding employment-related state: wage offers, vacancies,
      and labor demand.

   Borrower
      The firm role holding credit-related state: loan applications and
      projected fragility.

   Worker
      The household role holding labor-related state: employment status,
      employer ID, wage, and reservation wage.

   Consumer
      The household role holding consumption-related state: consumption
      budget, preferred firm, and spending behavior.

   Shareholder
      A built-in household role tracking per-period dividend income. Added
      in v0.3.0 to support buffer-stock MPC adjustment.

   Lender
      The bank role holding credit-related state: credit supply, interest
      rate, equity, bad debt exposure, and operational shock.

   event
      A system in the :term:`ECS` architecture. A class with an ``execute(sim)``
      method that reads and modifies role data. Defined with the ``@event``
      decorator. Events are executed in sequence by the :term:`pipeline`.

   relationship
      A many-to-many connection between agents stored in COO sparse format.
      The primary relationship is the ``LoanBook`` tracking active loans
      between firms and banks.

   pipeline
      The ordered sequence of :term:`events <event>` executed each simulation
      period. Defined in YAML (``default_pipeline.yml``) with special syntax
      for repetition, interleaving, and parameter substitution.

   simulation
      The top-level object (``Simulation``) that manages agent populations,
      roles, relationships, the pipeline, configuration, and RNG state.

   economy
      The ``Economy`` object holding aggregate state variables (GDP, unemployment,
      price index, etc.) that are updated each period by economy-level events.

   decorator registration
      The mechanism by which ``@role``, ``@event``, and ``@relationship``
      decorators automatically register classes in a global registry via
      ``__init_subclass__``. This enables lookup by name at runtime.

   auto-registration
      See :term:`decorator registration`.

   pipeline hook
      Metadata on an event class (set via ``@event(after=...)``,
      ``before=...``, or ``replace=...``) that positions it relative to
      another event in the pipeline. Applied explicitly via
      ``sim.use_events()``.

   vectorized operations
      NumPy-based array computations that process all agents simultaneously
      instead of looping. The ``ops`` module provides safe wrappers
      (e.g., ``ops.divide()`` handles division by zero).

   results
      The ``SimulationResults`` object returned by ``sim.run(collect=True)``.
      Contains per-period role data, economy data, and relationship snapshots.

   economy data
      Time series of aggregate variables (GDP, unemployment, inflation, etc.)
      stored in ``results.economy_data``. Accessed by key name.

   collect flag
      The ``collect=True`` parameter passed to ``sim.run()`` to enable data
      recording. Without it, the simulation runs but does not store results.

   extension
      A package that adds new roles, events, and configuration to the base
      model. Extensions export ``*_EVENTS`` lists and ``*_CONFIG`` dicts,
      activated via ``sim.use_role()``, ``sim.use_events()``, and
      ``sim.use_config()``.

   use_events
      The ``Simulation.use_events(*event_classes)`` method that applies
      :term:`pipeline hooks <pipeline hook>` from event classes, inserting
      them into the pipeline at the specified positions.

   use_role
      The ``Simulation.use_role(cls, n_agents=N)`` method that registers
      a new role with the simulation. The ``n_agents`` parameter sets the
      array size (defaults to ``n_firms``).

   use_config
      The ``Simulation.use_config(config_dict)`` method that applies
      extension default configuration with "don't overwrite" semantics.

   Config
      The configuration dataclass holding all simulation parameters.
      Constructed from three tiers: package defaults (``defaults.yml``),
      user YAML file, and keyword arguments.

   defaults.yml
      The YAML file (``src/bamengine/config/defaults.yml``) containing
      default values for all simulation parameters.

   default_pipeline.yml
      The YAML file (``src/bamengine/config/default_pipeline.yml``)
      defining the default event execution order.

   target
      A validation reference value (with bounds) from the reference book,
      defined in scenario-specific ``targets.yaml`` files. Used to assess
      whether simulation output matches expected behavior.

   metric
      A computed statistic from simulation results (e.g., mean unemployment,
      GDP growth rate) that is compared against a :term:`target` during
      validation.

   scoring
      The validation system that assigns a 0-1 score to each metric based
      on how close it is to its target. Scores are weighted and aggregated
      into a ``total_score``.

   weight
      A numeric value (0.5-5.0) assigned to each validation metric,
      indicating its relative importance. Affects both the weighted
      ``total_score`` and :term:`fail escalation`.

   fail escalation
      A mechanism where high-weight metrics have stricter PASS/FAIL
      thresholds. Formula: ``clamp(5 - 2*weight, 0.5, 5.0)``. At weight
      2.0, the escalation multiplier is 1.0 (neutral).
