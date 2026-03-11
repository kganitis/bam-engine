.. _implementation-decisions:

========================
Implementation Decisions
========================

BAM Engine implements the model described in *Macroeconomics from the Bottom-up*
(Delli Gatti et al., 2011). The book describes mechanisms at a high level, often using
continuous mathematics and leaving algorithmic details to the implementer. This page
documents every concrete economic and algorithmic choice made in BAM Engine that was
not clearly specified by the book.

Each entry states the choice, what the book says (or doesn't say), alternatives
considered, and the reasoning behind the decision.


Pricing
-------

.. _decision-breakeven-data:

Breakeven Price: Past vs Current Period Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** Breakeven price is calculated during the planning phase using previous
period's costs (wage bill, interest payments) and ``desired_production`` as the
denominator.

**Book reference:** Section 3.6 describes pricing during the planning phase but does
not specify which costs or denominator to use for the breakeven floor.

**Alternatives considered:**

- *Production-phase breakeven* using current-period costs and actual output
  (``labor_productivity × current_labor``) --- requires markets to have already cleared,
  so inapplicable at planning time. This alternative was implemented and tested but has
  been removed from the codebase.

**Reasoning:** At planning time, wage bill and interest payments are inherently stale
(previous period's values), and ``desired_production`` is the only available output
estimate. This is more stable than actual labor (which can be zero for firms that
failed to hire). Implemented by
:class:`~bamengine.events.planning.FirmsPlanBreakevenPrice` /
:class:`~bamengine.events.planning.FirmsPlanPrice`.


.. _decision-price-cut-floor:

Price Cut Breakeven Floor
~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** When a firm tries to cut its price, the breakeven floor can push the price
ABOVE the old price.

**Book reference:** Section 3.6 says firms "cannot sell below unit cost" but does not
address whether a price cut that hits the breakeven floor should be capped at the old
price.

**Alternatives considered:**

- *Cap the floor at old price* so a "cut" never results in an increase --- tested during
  Kalecki trap analysis; sell-through gain outweighed markup loss with no benefit for
  credit market activation. This alternative has been removed from the codebase.

**Reasoning:** Allowing the increase is simpler and avoids masking real cost pressures.
When breakeven exceeds the old price, the firm genuinely cannot afford to sell cheaper
--- forcing a cap would hide that signal.


.. _decision-avg-price:

Average Market Price: Production-Weighted Mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** Average market price is a production-weighted mean --- larger producers
have more influence on the reference price.

**Book reference:** Section 3.6 says "average price" without specifying weighting.

**Alternatives considered:**

- *Simple arithmetic mean* (all firms equal weight) --- gives equal voice to tiny firms.
- *Median or trimmed unweighted mean* --- robust but ignores volume.

**Reasoning:** A production-weighted mean better reflects the price a random unit of
goods was sold at, which is the economically relevant aggregate. Uses
``trimmed_weighted_mean()`` with ``trim_pct=0.0`` (no trimming by default). Falls back
to the previous period's price if all production is zero.


Market Matching
---------------

.. _decision-batch-matching:

Matching Structure: Batch Matching with Conflict Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** Both labor and credit markets use vectorized batch matching --- all
applications in a round are processed simultaneously using NumPy operations, with
conflict resolution for cases where multiple agents target the same counterparty.

**Book reference:** Sections 3.3--3.4 describe "sequential" matching without
specifying the implementation approach.

**Alternatives considered:**

- *Sequential loop-based matching* --- process each agent one at a time in a Python
  loop. Simpler but much slower. This alternative was the original implementation
  and has been removed from the codebase.
- *Cascade matching* --- each agent walks their entire ranked queue in a single pass.
  This alternative was implemented and tested but has been removed from the codebase.

**Reasoning:** Batch matching with conflict resolution produces equivalent economic
outcomes to sequential processing while being significantly faster through NumPy
vectorization. When multiple workers apply to the same firm, the conflict resolver
randomly selects one winner per vacancy, preserving fairness.


.. _decision-labor-conflict:

Within-Round Labor Matching: Batch Conflict Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** Within each matching round, all workers apply simultaneously. When
multiple workers target the same firm, a random conflict resolution step selects
winners based on available vacancies.

**Book reference:** Not specified.

**Alternatives considered:**

- *Sequential FIFO* --- process workers one at a time in shuffled order. Equivalent
  outcomes but slower due to Python loops. This alternative was the original
  implementation and has been removed from the codebase.
- *Simultaneous crowding* --- all workers apply at once, firms select randomly from
  crowded queues. Tested but produced artificial unemployment at high-wage firms
  due to crowding. This alternative has been removed from the codebase.

**Reasoning:** Batch conflict resolution produces the same statistical properties
as sequential FIFO (random ordering determines priority) while enabling vectorized
NumPy operations for performance.


.. _decision-search-pool:

Job Search Pool: All Firms
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** Unemployed workers sample from ALL firms, including those without
vacancies. Applications to non-hiring firms are wasted.

**Book reference:** Section 3.3 says "randomly chosen firms" --- does not specify
whether the pool is filtered by vacancies.

**Alternatives considered:**

- *Vacancies only* --- sample only from firms with open vacancies (no wasted
  applications). Tested but produced less realistic unemployment dynamics.

**Reasoning:** ``all_firms`` creates realistic search frictions --- workers lack
perfect information about which firms are hiring. Wasted applications are an
important source of frictional unemployment. Configurable via
``job_search_method`` in ``defaults.yml``.


.. _decision-batch-sequential-shopping:
.. _decision-sequential-shopping:

Goods Market: Sequential Shopping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** Consumers are shuffled and each completes all ``Z`` firm visits
before the next consumer starts. Consumer order is randomized each period.
The inner loop uses Python lists (converted from NumPy arrays via ``.tolist()``)
to avoid per-element NumPy overhead.

**Book reference:** Section 3.4 describes consumers visiting firms sequentially.

**Alternatives considered:**

- *Batch-sequential* --- consumers divided into ~10 batches, each batch processed
  with vectorized NumPy operations. Previously implemented (v0.6.0) but created
  phantom goods (~160 within-batch inventory collisions per period) where firms
  earned revenue for goods never produced. Removed.
- *Round-robin* --- all consumers visit one firm each, then all visit another, for
  ``Z`` rounds. Implemented and tested but produced different dynamics due to
  visit separation. Removed.

**Reasoning:** Sequential processing eliminates phantom goods entirely while adding
minimal overhead (~1-4% of total simulation time at all scales up to 10x). The
Python-list hot path makes sequential competitive with batch approaches by avoiding
NumPy's per-element indexing overhead.


Wages & Contracts
-----------------

.. _decision-min-wage:

Minimum Wage: Bidirectional (No Ratchet)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** Minimum wage adjusts bidirectionally with inflation --- it can decrease
during deflation.

**Book reference:** Section 3.4 says wages are "revised upward," implying a ratchet,
but the implementation allows decreases.

**Alternatives considered:**

- *Upward-only ratchet:* ``min_wage *= max(1, 1 + inflation)`` --- wage never decreases.
  Tested but caused permanent ratcheting during inflationary episodes, creating a
  price-wage spiral floor. This alternative has been removed from the codebase.

**Reasoning:** Bidirectional adjustment prevents the minimum wage from permanently
ratcheting up. During deflation, wage decreases help firms maintain competitiveness.


.. _decision-initial-wage:

Initial Wage Offer: price_init / 3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** Initial wage offer for all firms is ``price_init / 3`` (with
``price_init=0.5``, this gives ~0.167).

**Book reference:** Not specified --- the book gives the structure but not initial
conditions.

**Alternatives considered:**

- A direct parameter (``wage_offer_init``) --- adds a free parameter without clear
  benefit.
- Derived from labor productivity or minimum wage --- less direct link to goods price.

**Reasoning:** Derived from ``price_init`` to ensure internal consistency --- wages
must be a fraction of the goods price for firms to be viable. The ``/3`` ratio was
calibrated to produce stable early-period dynamics where firms can cover wage costs
from revenue.


Credit Market
-------------

.. _decision-leverage-cap:

Maximum Leverage Cap for Interest Rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** The fragility ratio used in the interest rate formula
(``r_bar × (1 + φ × fragility)``) is capped at ``max_leverage=10``.

**Book reference:** Equation 3.7 has no cap on the ``μ(ℓ)`` function.

**Alternatives considered:**

- *No cap* (``max_leverage=100``, effectively uncapped) --- extreme rates for highly
  leveraged firms create a death spiral that guarantees default.

**Reasoning:** Without a cap, firms with very high leverage face astronomical interest
rates that guarantee default, creating a destructive positive-feedback loop. The cap at
10 (= 1/v, matching the inverse of the bank capital requirement) prevents extreme rates
while still penalizing leverage.


.. _decision-loan-cap:

Maximum Loan to Net Worth Ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** Individual loans are capped at ``2 × borrower's net_worth``.

**Book reference:** Not mentioned.

**Alternatives considered:**

- *No cap* (``max_loan_to_net_worth=100``, effectively uncapped) --- a single loan
  could dwarf the firm's equity, creating extreme leverage in one transaction.

**Reasoning:** The cap keeps loan sizes proportional to firm capacity. Evolved from
100 → 5 → 2 through calibration. At 2×, firms can still borrow meaningfully but
cannot become dangerously leveraged in a single period.


.. _decision-bad-debt:

Bad Debt Recovery: Proportional to Principal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** When a firm defaults, each bank recovers
``clip(frac × net_worth, 0, principal)`` where
``frac = this_loan_principal / total_principal``.
Loss = ``principal − recovery``.

**Book reference:** Section 3.5 mentions bad debt but does not specify the recovery
formula.

**Alternatives considered:**

- *Full write-off* (bank loses entire loan) --- too harsh, amplifies financial contagion.
- *Proportional to total debt including interest* --- penalizes banks that charged higher
  rates, creating perverse incentives.

**Reasoning:** Proportional recovery based on principal (not total debt) ensures banks
with larger principal exposure bear proportionally larger losses. Capping at principal
prevents negative losses. Using net worth as the recovery pool reflects the firm's
remaining equity.


Inflation
---------

.. _decision-inflation:

Inflation: Year-over-Year (4-Period Lookback)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** Inflation is calculated as ``(P_now − P_{t−4}) / P_{t−4}`` --- comparing
the current price to 4 quarters ago (annual comparison).

**Book reference:** Section 3.4 says "price index change" without specifying the
method.

**Alternatives considered:**

- *Annualized quarterly:* ``(1 + quarterly_rate)^4 − 1`` --- requires only 2 periods
  of history but amplifies short-term fluctuations. This alternative has been removed
  from the codebase.

**Reasoning:** Year-over-year is the standard macroeconomic measure, preferred by
statistical agencies for exactly this reason: it smooths seasonal and short-run noise.
Requires 5 periods of price history (returns 0 until then).


Revenue & Dividends
-------------------

.. _decision-equal-dividends:

Equal Dividend Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** Total dividends are divided equally among ALL households:
``dividend_per_household = total_dividends / n_households``.

**Book reference:** Section 3.7 mentions "firm ownership" but does not formalize a
capitalist class or ownership structure.

**Alternatives considered:**

- *Proportional to savings/wealth* --- creates wealth concentration feedback.
- *Proportional to shares* --- requires introducing an ownership/equity market.
- *Only to employed workers* --- conflates labor income and capital income.

**Reasoning:** Equal distribution is a modeling simplification that avoids introducing
a separate Capitalist agent type. It preserves stock-flow consistency (firm debits =
household credits) while maintaining model parsimony.


.. _decision-dividends-savings:

Dividends Credited to Savings (Not Income)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** Dividends are added to household savings, not income.

**Book reference:** Not specified.

**Alternatives considered:**

- *Add to income* --- dividends would be spent via the propensity function in the same
  period, creating an immediate spending boost.

**Reasoning:** Crediting to savings means dividends are available for spending next
period via the wealth calculation (``wealth = savings + income``). If credited to
income, dividends would be subject to the propensity function immediately, distorting
consumption patterns by creating a same-period spending boost.


Bankruptcy
----------

.. _decision-dual-bankruptcy:

Dual Bankruptcy Criteria: Net Worth OR Zero Production
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** A firm goes bankrupt if ``net_worth < 0`` OR ``production_prev = 0``
(the "ghost firm" rule).

**Book reference:** Section 3.6 specifies "negative net worth" as the sole
bankruptcy condition.

**Alternatives considered:**

- *Only net-worth trigger* (book specification) --- allows zombie firms to persist
  indefinitely.

**Reasoning:** The zero-production trigger prevents "zombie firms" --- firms that
have no workers, produce nothing, but maintain positive net worth from accumulated
savings. Without this rule, zombie firms occupy slots that could be filled by active
entrants, distorting the firm size distribution.


Balance Sheet Design
--------------------

.. _decision-total-funds:

Separation of Liquid Cash from Balance-Sheet Equity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** Firms have two separate financial variables: ``total_funds`` (liquid cash
that changes during a period) and ``net_worth`` (balance-sheet equity fixed during the
period, updated only at the bankruptcy phase).

**Book reference:** The book uses a single "equity" concept (A\ :sub:`i`) without
distinguishing liquid cash from book equity.

**Alternatives considered:**

- *Single variable* serving both purposes (as the book implicitly assumes) --- but then
  intra-period readers see an unstable moving target.

**Reasoning:** During a period, multiple events need to read ``net_worth`` for
calculations (fragility ratio, loan caps, bad debt recovery fractions) while
``total_funds`` fluctuates with every transaction (loans in → wages out → revenue
in → debt out → dividends out). Separating them ensures that intra-period financial
decisions use a stable reference point. They reconverge at the bankruptcy phase:
``net_worth += retained_profit; total_funds = max(net_worth, 0)``.

**Where net_worth is read (must stay fixed within a period):**

- Fragility ratio: ``credit_demand / net_worth``
- Loan cap: ``net_worth × max_loan_to_net_worth``
- Bad debt recovery: ``frac × net_worth``

**Where total_funds is modified (intra-period cash flow):**

- Loans credited (+)
- Wages debited (−)
- Revenue credited (+)
- Debt repaid (−)
- Dividends debited (−)


Measurement Timing
------------------

.. _decision-capture-timing:

Data Capture Timing: Event-Hooked vs End-of-Period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choice:** Variables are captured at specific pipeline points using
``capture_timing`` configs, not at end-of-period.

**Book reference:** Does not discuss measurement timing --- this is a pure
implementation concern.

**Alternatives considered:**

- *Capture all variables at end-of-period* (after all events including
  bankruptcy/entry) --- includes newly spawned firms with initial values, which
  skew averages.

**Reasoning:** End-of-period capture includes newly spawned firms with initial
values, skewing aggregate statistics. Event-hooked capture ensures each variable
reflects its intended economic meaning at the right moment. Scenario-specific
``COLLECT_CONFIG`` dictionaries define the timing per variable.

**Key timing decisions:**

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Variable
     - Captured After
     - Rationale
   * - ``Worker.employed``
     - ``firms_run_production``
     - Measures who WORKED this period. Capturing later (after contract expiry)
       counts expired workers as unemployed even though they worked.
   * - ``Worker.wage``
     - ``workers_receive_wage``
     - After wages are credited.
   * - ``Producer.production``
     - ``firms_run_production``
     - Immediately after production occurs.
   * - ``Producer.price``
     - ``firms_plan_price``
     - The price consumers will see in the goods market.
   * - ``Producer.inventory``
     - ``consumers_finalize_purchases``
     - Remaining unsold inventory after all shopping completes.
   * - ``Producer.labor_productivity``
     - ``firms_apply_productivity_growth`` (Growth+) or end-of-period (baseline)
     - In Growth+, productivity changes due to R&D. In baseline, it is constant.
   * - ``Employer.n_vacancies``
     - ``firms_decide_vacancies``
     - Demand-side labor signal before matching occurs.
   * - ``Borrower.net_worth``
     - ``firms_run_production``
     - Mid-period balance-sheet equity, before revenue/debt flows.
   * - ``Consumer.savings``
     - End-of-period
     - Must reflect final state after all income/spending flows.
   * - ``LoanBook.*``
     - ``credit_market_round``
     - After credit matching, showing all loans granted this period.
