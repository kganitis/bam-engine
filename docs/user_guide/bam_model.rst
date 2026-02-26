The BAM Model
=============

The BAM (Bottom-Up Adaptive Macroeconomics) model is an agent-based macroeconomic
model from the CATS (Complex Adaptive Trivial Systems) family, originally described
in *Macroeconomics from the Bottom-up* by Delli Gatti, Gaffeo, Gallegati, Giulioni,
and Palestrini (2011), Chapter 3. This page describes the economic rules implemented
by BAM Engine — the behavioral equations, market mechanisms, and institutional
structures that generate emergent macroeconomic dynamics.

.. contents:: On this page
   :local:
   :depth: 2


Notation
--------

The following symbols are used throughout this page:

.. list-table::
   :header-rows: 1
   :widths: 12 40 48

   * - Symbol
     - Meaning
     - BAM Engine parameter / field
   * - :math:`\varphi`
     - Labor productivity (goods per worker)
     - ``labor_productivity``
   * - :math:`\theta`
     - Employment contract duration (periods)
     - ``theta``
   * - :math:`\delta`
     - Dividend payout ratio
     - ``delta``
   * - :math:`\beta`
     - Consumption propensity exponent
     - ``beta``
   * - :math:`\nu`
     - Bank capital requirement coefficient
     - ``v``
   * - :math:`\bar{r}`
     - Baseline (policy) interest rate
     - ``r_bar``
   * - :math:`h_\rho`
     - Production shock half-width
     - ``h_rho``
   * - :math:`h_\xi`
     - Wage shock half-width
     - ``h_xi``
   * - :math:`h_\eta`
     - Price shock half-width
     - ``h_eta``
   * - :math:`h_\phi`
     - Bank cost shock half-width
     - ``h_phi``
   * - :math:`M`
     - Max job applications per worker per period
     - ``max_M``
   * - :math:`H`
     - Max loan applications per firm per period
     - ``max_H``
   * - :math:`Z`
     - Max shops visited per consumer per period
     - ``max_Z``
   * - :math:`\hat{w}_t`
     - Minimum wage at time :math:`t`
     - ``Economy.min_wage``
   * - :math:`\bar{P}_t`
     - Average market price at time :math:`t`
     - ``Economy.avg_mkt_price``

Subscripts: :math:`i` indexes firms, :math:`j` indexes households, :math:`k`
indexes banks, :math:`t` indexes time periods.


Agent Types
-----------

The model simulates three types of agents, each composed of one or more
**roles** (see :doc:`overview` for the ECS architecture):

.. list-table::
   :header-rows: 1
   :widths: 15 30 20 35

   * - Agent Type
     - Roles
     - Count
     - Key State Variables
   * - Firms
     - Producer, Employer, Borrower
     - ``n_firms``
     - price, inventory, production, wage_offer, net_worth, credit_demand
   * - Households
     - Worker, Consumer, Shareholder
     - ``n_households``
     - wage, savings, income, propensity, employer, dividends
   * - Banks
     - Lender
     - ``n_banks``
     - equity_base, credit_supply, interest_rate

**Firms** are the central agents: they plan production, hire workers, borrow from
banks, produce goods, set prices, and sell to consumers. Their survival depends
on maintaining positive net worth.

**Households** supply labor and demand consumption goods. They save unspent income
and receive dividend income from profitable firms.

**Banks** are passive credit intermediaries. They supply credit up to a leverage
constraint and set interest rates with a risk premium based on borrower fragility.


The Economic Cycle
------------------

Each simulation period executes events in 8 sequential phases. This section
describes the economic logic and mathematical rules of each phase.


Phase 1: Planning
~~~~~~~~~~~~~~~~~

Firms decide how much to produce and how many workers to hire, based on
adaptive expectations about demand.

**Expected demand.** Firms adjust production based on two signals: whether they
have unsold inventory (:math:`S_{i,t-1} > 0`) and whether their price is above
or below the market average (:math:`P_{i,t-1} \gtrless \bar{P}_{t-1}`):

.. math::

   Y^d_{i,t} =
   \begin{cases}
   Y_{i,t-1} \times (1 + \rho_{i,t})
     & \text{if } S_{i,t-1} = 0 \text{ and } P_{i,t-1} \geq \bar{P}_{t-1} \\
   Y_{i,t-1} \times (1 - \rho_{i,t})
     & \text{if } S_{i,t-1} > 0 \text{ and } P_{i,t-1} < \bar{P}_{t-1} \\
   Y_{i,t-1}
     & \text{otherwise}
   \end{cases}

where :math:`\rho_{i,t} \sim U(0, h_\rho)` is a random production shock. The
first case (sold everything at a competitive price) triggers expansion; the
second case (unsold inventory at an uncompetitive price) triggers contraction;
the remaining cases maintain the status quo.

.. note::

   The production signal :math:`Y_{i,t-1}` uses **actual production** from the
   previous period, not desired production. This creates a feedback loop: if a
   firm's desired contraction is absorbed by the ``ceil()`` quantization (see
   below), actual production stays the same, and the signal resets.

**Desired labor.** Firms compute the workforce needed for their production target:

.. math::

   L^d_{i,t} = \left\lceil \frac{Y^d_{i,t}}{\varphi} \right\rceil

The ceiling function ensures integer labor demand but creates a **quantization
effect**: small production decreases may not reduce the workforce at all. For a
firm with :math:`L` workers and shock width :math:`h_\rho`, the probability that
a decrease actually reduces workforce is
:math:`\max(0, 1 - 1/(L \times h_\rho))`. With the default :math:`h_\rho = 0.10`
and typical firm size :math:`L \approx 5`, this probability is zero — firms
effectively cannot shrink through this channel alone.

**Vacancies and firing.** If :math:`L^d_{i,t} > L^{current}_{i,t}`, the firm
posts vacancies:

.. math::

   V_{i,t} = \max(0,\; L^d_{i,t} - L^{current}_{i,t})

If :math:`L^d_{i,t} < L^{current}_{i,t}`, the firm fires
:math:`L^{current}_{i,t} - L^d_{i,t}` workers selected at random.

**Breakeven price.** In the default (planning-phase) pricing, firms also
compute a cost-covering price floor using the previous period's wage bill:

.. math::

   P^b_{i,t} = \frac{WB_{i,t-1} + \sum_k r_{i,k} \cdot B_{i,k}}{Y^d_{i,t}}

**Price adjustment.** Firms then adjust their price based on inventory and
market position, subject to the breakeven floor:

.. math::

   P_{i,t} =
   \begin{cases}
   \max(P^b_{i,t},\; P_{i,t-1} \times (1 - \eta_{i,t}))
     & \text{if } S_{i,t-1} > 0 \text{ and } P_{i,t-1} \geq \bar{P}_{t-1} \\
   \max(P^b_{i,t},\; P_{i,t-1} \times (1 + \eta_{i,t}))
     & \text{if } S_{i,t-1} = 0 \text{ and } P_{i,t-1} < \bar{P}_{t-1} \\
   \max(P^b_{i,t},\; P_{i,t-1})
     & \text{otherwise}
   \end{cases}

where :math:`\eta_{i,t} \sim U(0, h_\eta)`. The breakeven floor prevents
price cuts below unit cost, ensuring every sale is profitable.


Phase 2: Labor Market
~~~~~~~~~~~~~~~~~~~~~

Workers search for jobs and firms hire to fill vacancies.

**Inflation and minimum wage.** The economy-wide inflation rate is computed as
the year-over-year change in average market price:

.. math::

   \pi_t = \frac{\bar{P}_t - \bar{P}_{t-4}}{\bar{P}_{t-4}}

The statutory minimum wage is revised every ``min_wage_rev_period`` periods
(default: 4) to keep pace with inflation:

.. math::

   \hat{w}_{t} = \hat{w}_{t-1} \times (1 + \pi_t)

**Wage setting.** Firms with vacancies set a wage offer with a random markup
above the minimum wage:

.. math::

   w^b_{i,t} =
   \begin{cases}
   \max\bigl(\hat{w}_t,\; w_{i,t-1}\bigr)
     & \text{if } V_{i,t} = 0 \\
   \max\bigl(\hat{w}_t,\; w_{i,t-1} \times (1 + \xi_{i,t})\bigr)
     & \text{if } V_{i,t} > 0
   \end{cases}

where :math:`\xi_{i,t} \sim U(0, h_\xi)` is a wage increase shock.

**Job search.** Each unemployed worker sends :math:`M` applications. Workers
whose contracts expired in the previous period send their first application to
their former employer, then :math:`M - 1` to randomly selected firms. Fully
unemployed workers apply to :math:`M` random firms.

**Matching.** Labor market matching is **decentralized and interleaved**: in each
of :math:`M` rounds, all workers simultaneously send one application, then all
firms simultaneously hire from their application queues. Workers apply to their
highest-wage firm first, so matching favors high-wage firms — a form of
**preferential attachment**.

**Contracts.** Hired workers sign contracts of length :math:`\theta` periods
(default: 8). During the contract, the worker's wage is fixed at the firm's
offer at the time of hiring. When the contract expires, the worker becomes
unemployed and must search for a new job.


Phase 3: Credit Market
~~~~~~~~~~~~~~~~~~~~~~

Firms that cannot self-finance their wage bill borrow from banks.

**Credit supply.** Each bank's lending capacity is constrained by its equity
and the capital requirement coefficient:

.. math::

   C_{k,t} = \frac{E_{k,t}}{\nu}

With the default :math:`\nu = 0.10`, a bank with equity 5.0 can lend up to 50.0.
Credit supply is reduced as loans are granted during matching rounds.

**Interest rate.** Banks set interest rates as a markup over the policy rate,
with a random operating cost shock:

.. math::

   r_{k,t} = \bar{r} \times (1 + \phi_{k,t})

where :math:`\phi_{k,t} \sim U(0, h_\phi)`. The interest rate is further
adjusted for borrower risk via the financial fragility premium (see below).

**Credit demand.** Firms borrow only when their net worth is insufficient to
cover the wage bill:

.. math::

   B_{i,t} = \max(0,\; WB_{i,t} - A_{i,t})

where :math:`WB_{i,t}` is the wage bill and :math:`A_{i,t}` is net worth. When
net worth exceeds the wage bill, the firm self-finances entirely.

**Financial fragility.** Each firm's projected leverage ratio is used to
prioritize loan applications:

.. math::

   \ell_{i,t} = \frac{B_{i,t}}{A_{i,t}}

This metric is capped at ``max_leverage`` (default: 10) for firms with very low
net worth. Banks process applications in **ascending fragility order** — the
least leveraged firms get served first.

**Matching.** Credit market matching mirrors the labor market: :math:`H`
interleaved rounds where firms send one application per round and banks process
queues. Each firm contacts :math:`H` banks sorted by interest rate (lowest
first). Individual loans are capped at ``max_loan_to_net_worth`` times the
borrower's net worth (default: 2).

**Post-credit firing.** After credit matching, firms that still cannot cover
their full wage bill fire workers until the wage bill fits within available funds
(net worth + loans received).


Phase 4: Production
~~~~~~~~~~~~~~~~~~~

Firms pay wages and produce goods.

**Wage payment.** Each firm pays wages to all its employees:

.. math::

   WB_{i,t} = \sum_{j \in \text{employees}_i} w_{j}

The wage bill is deducted from the firm's available funds. Workers receive their
wage as income.

**Production.** Output is determined by the production function:

.. math::

   Y_{i,t} = \varphi \times L_{i,t}

where :math:`\varphi` is labor productivity (constant in the baseline model,
endogenous in the Growth+ extension) and :math:`L_{i,t}` is the current
workforce. Production is added to the firm's inventory.

**Contract updates.** Each employed worker's remaining contract duration is
decremented by one period. Workers whose contracts reach zero are marked as
unemployed and become available for the next period's labor market.


Phase 5: Goods Market
~~~~~~~~~~~~~~~~~~~~~

Households decide how much to spend and shop for goods.

**Marginal propensity to consume.** Each household's spending propensity is a
decreasing function of their relative savings position:

.. math::

   c_{j,t} = \frac{1}{1 + \left[\tanh\!\left(\frac{SA_{j,t}}{\overline{SA}_t}\right)\right]^\beta}

where :math:`SA_{j,t}` is the household's accumulated savings,
:math:`\overline{SA}_t` is the economy-wide average savings, and :math:`\beta`
is the propensity exponent (default: 2.5). Households with below-average savings
spend a larger fraction of income; those with above-average savings spend less.

**Spending budget.** The household allocates income for consumption:

.. math::

   \text{budget}_{j,t} = c_{j,t} \times \text{income}_{j,t}

**Consumer search.** Each consumer visits :math:`Z` firms. The first visit goes
to the **largest producer from the previous period** (preferential attachment /
loyalty), and the remaining :math:`Z - 1` visits go to randomly selected firms.
Consumers rank their selected firms by price (lowest first) and buy sequentially
until their budget is exhausted or all firms are visited.

**Rationing.** When a consumer's demand exceeds a firm's available inventory,
the consumer buys only what is available and moves to the next firm. Unspent
budget is returned to savings at the end of the shopping phase.


Phase 6: Revenue & Dividends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Firms collect revenue and distribute profits.

**Revenue and gross profit.**

.. math::

   R_{i,t} &= P_{i,t} \times \text{sold}_{i,t} \\
   \Pi^g_{i,t} &= R_{i,t} - WB_{i,t}

**Debt repayment and net profit.** Firms repay principal plus interest on all
outstanding loans. Loans are settled from available funds; if funds are
insufficient, the shortfall becomes bad debt for the lending bank:

.. math::

   \Pi_{i,t} = \Pi^g_{i,t} - \sum_k r_{i,k} \times B_{i,k}

**Dividends.** Firms with positive net profit pay a fraction as dividends to
households:

.. math::

   \text{div}_{i,t} &= \delta \times \max(0, \Pi_{i,t}) \\
   \Pi^r_{i,t} &= \Pi_{i,t} - \text{div}_{i,t}

where :math:`\delta` is the dividend payout ratio (default: 0.10). Dividends
are distributed equally across all households via the
:class:`~bamengine.roles.shareholder.Shareholder` role.


Phase 7: Bankruptcy
~~~~~~~~~~~~~~~~~~~

Insolvent agents are detected and removed.

**Net worth update.** Firm net worth evolves according to retained profits:

.. math::

   A_{i,t} = A_{i,t-1} + \Pi^r_{i,t}

**Insolvency condition.** A firm goes bankrupt when its net worth becomes
negative:

.. math::

   A_{i,t} < 0 \implies \text{firm } i \text{ exits}

Bankrupt firms fire all workers, and their outstanding loans become bad debt for
the lending banks:

.. math::

   \text{recovery}_k = \text{clip}\!\left(\text{frac} \times A_{i,t},\; 0,\; B_{i,k}\right)

The actual loss for bank :math:`k` is :math:`B_{i,k} - \text{recovery}_k`.

**Bank insolvency.** Banks whose equity falls below zero (from accumulated bad
debt losses) also go bankrupt and are replaced.


Phase 8: Entry
~~~~~~~~~~~~~~

Bankrupt agents are replaced one-for-one, maintaining constant population sizes.

**Replacement firms** are initialized with attributes derived from the
**trimmed mean** (excluding top and bottom 5%) of surviving firms:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Initialization Rule
   * - Net worth
     - ``new_firm_size_factor`` :math:`\times` survivor mean (default: 50%)
   * - Production capacity
     - ``new_firm_production_factor`` :math:`\times` survivor mean (default: 50%)
   * - Wage offer
     - ``new_firm_wage_factor`` :math:`\times` survivor mean (default: 50%)
   * - Price
     - ``new_firm_price_markup`` :math:`\times` average market price (default: 1.15)

New entrants start below average size, consistent with the empirical regularity
that young firms are smaller than incumbents.

**Replacement banks** are re-initialized with the default ``equity_base_init``.


Market Matching
---------------

All three markets use **decentralized matching** with search frictions — a key
departure from Walrasian general equilibrium models where a central auctioneer
clears markets instantaneously.

**Interleaved rounds.** Matching proceeds in multiple rounds. In each round,
seekers (workers, firms, or consumers) send one application/visit, then
providers (firms, banks, or firms) process their queues. This interleaving
means early matches reduce available capacity for later rounds, creating
realistic market frictions.

**Preferential attachment.** In the labor and goods markets, agents have a
tendency to return to previous partners: workers whose contracts expired apply
first to their former employer; consumers visit their previous period's largest
producer first. This creates endogenous firm size persistence and market
concentration.

**Search frictions.** The parameters :math:`M`, :math:`H`, and :math:`Z` control
how many partners each agent can contact per period. Lower values create
more friction (less efficient matching); higher values approach perfect
information. The defaults (:math:`M = 4`, :math:`H = 2`, :math:`Z = 2`) create
moderate frictions consistent with the reference model.


Key Parameters
--------------

The following table summarizes the most important model parameters. See
:doc:`configuration` for the full parameter reference and :doc:`bam_model`
for the economic interpretation of each phase.

.. list-table::
   :header-rows: 1
   :widths: 20 15 10 55

   * - Parameter
     - Symbol
     - Default
     - Description
   * - ``n_firms``
     - —
     - 100
     - Number of firms in the economy
   * - ``n_households``
     - —
     - 500
     - Number of households (recommended: :math:`\geq 5 \times` ``n_firms``)
   * - ``n_banks``
     - —
     - 10
     - Number of banks
   * - ``labor_productivity``
     - :math:`\varphi`
     - 0.50
     - Goods produced per worker per period
   * - ``theta``
     - :math:`\theta`
     - 8
     - Employment contract length (periods)
   * - ``delta``
     - :math:`\delta`
     - 0.10
     - Dividend payout ratio (fraction of net profit)
   * - ``beta``
     - :math:`\beta`
     - 2.50
     - Consumption propensity exponent
   * - ``v``
     - :math:`\nu`
     - 0.10
     - Bank capital requirement coefficient (max leverage = :math:`1/\nu`)
   * - ``r_bar``
     - :math:`\bar{r}`
     - 0.02
     - Baseline (policy) interest rate
   * - ``h_rho``
     - :math:`h_\rho`
     - 0.10
     - Max production growth shock
   * - ``h_xi``
     - :math:`h_\xi`
     - 0.05
     - Max wage growth shock
   * - ``h_eta``
     - :math:`h_\eta`
     - 0.10
     - Max price adjustment shock
   * - ``h_phi``
     - :math:`h_\phi`
     - 0.10
     - Max bank operating cost shock
   * - ``max_M``
     - :math:`M`
     - 4
     - Job applications per unemployed worker per period
   * - ``max_H``
     - :math:`H`
     - 2
     - Loan applications per firm per period
   * - ``max_Z``
     - :math:`Z`
     - 2
     - Shops visited per consumer per period


Economy Statistics
------------------

BAM Engine tracks several economy-wide metrics each period:

- **Average market price** (:math:`\bar{P}_t`): exponentially smoothed average
  of all firm prices, weighted by production
- **Unemployment rate**: fraction of households without an employer, measured
  after the production phase
- **Inflation rate**: year-over-year change in average market price
  (:math:`\pi_t = (\bar{P}_t - \bar{P}_{t-4}) / \bar{P}_{t-4}`)

These are stored in ``sim.ec`` (the :class:`~bamengine.economy.Economy` object)
and accessible as time series via ``sim.ec.avg_mkt_price_history``,
``sim.ec.unemp_rate_history``, and ``sim.ec.inflation_history``.


Further Reading
---------------

- **Original reference**: Delli Gatti, D., Gaffeo, E., Gallegati, M., Giulioni,
  G., & Palestrini, A. (2011). *Macroeconomics from the Bottom-up*. Springer.
  Chapter 3.
- **Extensions**: The Growth+ (R&D), buffer-stock consumption, and taxation
  extensions add endogenous productivity, heterogeneous saving behavior, and
  fiscal policy. See :doc:`extensions`.
- **Validation**: BAM Engine includes a validation framework for comparing
  simulation output to the reference results in the book. See :doc:`validation`.
- **Configuration**: For the full parameter reference with valid ranges and
  effects, see :doc:`configuration`.

.. seealso::

   :doc:`overview` for the software architecture, :doc:`pipelines` for the
   event execution order, :doc:`extensions` for model extensions.
