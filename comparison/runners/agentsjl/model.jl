"""
model.jl - Agents.jl v7 implementation of the baseline BAM model: agent types,
economy-state properties, and `build_model`.

This is the structural scaffold (Task 2 of the port). It establishes the
Agents.jl v7.0.3 idioms reused by every later phase task:

  * `@agent struct Firm(NoSpaceAgent) ... end` + `@multiagent BAMAgent(Firm, Household, Bank)`
    merge the three agent variants into one type-stable sum type. Retrieve a
    variant with `variantof(a)` (the type) and `variant(a)` (the enclosed
    instance whose fields are read/mutated).
  * `BAMProperties` is a concrete mutable struct (NOT a heterogeneous Dict) so
    that `model.field` access is type-stable.
  * `build_model` constructs a no-space `StandardABM` with `rng=Xoshiro(seed)`
    and a no-op `model_step! = bam_step!` (the real pipeline is wired in Task 9).

Field names and initial values mirror the Mesa reference in
`comparison/runners/mesa/agents.py` and `comparison/runners/mesa/model.py`.

Relationship encoding (type stability):
  * A Household's employer is an integer agent id, `employer_id`. The sentinel
    `NO_AGENT = 0` means "unemployed / none" (Agents.jl ids start at 1, so 0 is
    safe). The same sentinel is reused for `employer_prev_id` and
    `largest_prod_prev_id`.
  * Loans are a single loan-book vector held in `BAMProperties.loans`; each
    `Loan` records its `borrower_id` and `lender_id`. Matching logic is filled
    in by later tasks; here only the fields are defined.
"""

using Agents
using Random

# Sentinel for "no agent" (unemployed worker, no previous employer, no loyalty
# target, etc.). Agents.jl assigns ids starting at 1, so 0 is always free.
const NO_AGENT = 0

# ---------------------------------------------------------------------------
# Loan record (immutable, type-stable)
# ---------------------------------------------------------------------------
"""
    Loan(principal, rate, borrower_id, lender_id)

A single loan in the economy's loan book. Mirrors the Mesa `Loan` NamedTuple
(`principal`, `rate`, `lender`) but stores integer agent ids instead of object
references, and additionally records the `borrower_id` because the loan book is
a single shared vector in `BAMProperties` rather than a per-firm list.

`interest` = `principal * rate`; `debt` = `principal * (1 + rate)`.
"""
struct Loan
    principal::Float64
    rate::Float64
    borrower_id::Int
    lender_id::Int
end

interest(l::Loan) = l.principal * l.rate
debt(l::Loan) = l.principal * (1.0 + l.rate)

# ---------------------------------------------------------------------------
# Agent variants (columns = the Mesa agent attributes)
# ---------------------------------------------------------------------------

"""
    Firm

Firm agent (Producer + Employer + Borrower roles). Field names and initial
values match the Mesa `Firm.__init__`. Per-round scratch ids/queues
(`loan_apps`, `employee_ids`) default to empty vectors; the loan book itself
lives in `BAMProperties.loans`.
"""
@agent struct Firm(NoSpaceAgent)
    # Producer
    production::Float64 = 0.0
    production_prev::Float64
    inventory::Float64 = 0.0
    expected_demand::Float64 = 1.0
    desired_production::Float64 = 0.0
    labor_productivity::Float64
    breakeven_price::Float64
    price::Float64
    # Employer
    desired_labor::Int = 0
    current_labor::Int = 0
    wage_offer::Float64
    wage_bill::Float64 = 0.0
    n_vacancies::Int = 0
    total_funds::Float64
    # Borrower
    net_worth::Float64
    credit_demand::Float64 = 0.0
    projected_fragility::Float64 = 0.0
    gross_profit::Float64 = 0.0
    net_profit::Float64 = 0.0
    retained_profit::Float64 = 0.0
    # Scratch for market rounds (populated during the credit market phase).
    # Ranked bank ids the firm applies to, consumed per round.
    loan_apps::Vector{Int} = Int[]
    # Ids of currently employed households (deterministic insertion order).
    employee_ids::Vector{Int} = Int[]
end

"""
    Household

Household agent (Worker + Consumer + Shareholder roles). Field names and initial
values match the Mesa `Household.__init__`. Object references in the Mesa port
(`employer`, `employer_prev`, `largest_prod_prev`) are encoded as integer agent
ids with the `NO_AGENT == 0` sentinel.
"""
@agent struct Household(NoSpaceAgent)
    # Worker
    employer_id::Int = NO_AGENT        # firm id, or NO_AGENT (unemployed)
    employer_prev_id::Int = NO_AGENT
    wage::Float64 = 0.0
    periods_left::Int = 0
    contract_expired::Bool = false
    fired::Bool = false
    # Consumer
    income::Float64 = 0.0
    savings::Float64
    income_to_spend::Float64 = 0.0
    propensity::Float64 = 0.0
    largest_prod_prev_id::Int = NO_AGENT   # firm id, or NO_AGENT (none visited)
    job_apps::Vector{Int} = Int[]          # ranked firm ids
    shop_visits::Vector{Int} = Int[]       # price-sorted firm ids
    # Shareholder
    dividends::Float64 = 0.0
end

"""
    Bank

Bank agent (Lender role). Field names and initial values match the Mesa
`Bank.__init__`.
"""
@agent struct Bank(NoSpaceAgent)
    equity_base::Float64
    credit_supply::Float64 = 0.0
    interest_rate::Float64 = 0.0
    opex_shock::Float64 = 0.0
end

# Merge the three variants into a single type-stable sum type. Retrieve a
# variant with `variantof(a)` and the enclosed instance with `variant(a)`.
@multiagent BAMAgent(Firm, Household, Bank)

# ---------------------------------------------------------------------------
# Economy-state properties (type-stable mutable struct, NOT a Dict)
# ---------------------------------------------------------------------------

"""
    BAMProperties

Model-level economy state. A concrete mutable struct (not a heterogeneous Dict)
so that `model.field` access is type-stable. Mirrors the economy state set in
the Mesa `BamModel.__init__`.

Fields:
  * `params` - the canonical economic parameter dict (key => value).
  * `eps` - numerical floor (Mesa `EPS = 1e-9`).
  * `period` - current period counter.
  * `n_firms`, `n_households`, `n_banks` - population sizes. `n_households` is
    set before firms are added because the Mesa Firm init derives
    `production_prev` from it.
  * `avg_mkt_price`, `min_wage`, `min_wage_rev_period` - market state.
  * `avg_mkt_price_history`, `inflation_history` - per-period histories.
  * `loans` - the single shared loan book (vector of `Loan`).
  * `collapsed` - collapse flag (all firms or all banks gone).
"""
mutable struct BAMProperties
    params::Dict{String,Float64}
    eps::Float64
    period::Int
    n_firms::Int
    n_households::Int
    n_banks::Int
    avg_mkt_price::Float64
    min_wage::Float64
    min_wage_rev_period::Int
    avg_mkt_price_history::Vector{Float64}
    inflation_history::Vector{Float64}
    loans::Vector{Loan}
    collapsed::Bool
end

# ---------------------------------------------------------------------------
# Per-period step (no-op scaffold; the real pipeline is wired in Task 9)
# ---------------------------------------------------------------------------

"""
    bam_step!(model)

Per-period model step. A no-op scaffold for now; later tasks fill in the BAM
phase pipeline (planning, labor, credit, production, goods, revenue,
bankruptcy/entry). Defined as a named function (rather than `dummystep`) so the
`StandardABM` is constructed with a real `model_step!` and Task 9 has a single
clear hook to extend.
"""
function bam_step!(model)
    return nothing
end

# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

"""
    build_model(n_firms, n_households, n_banks, params, seed) -> StandardABM

Construct the baseline BAM `StandardABM` (no space) with `rng = Xoshiro(seed)`,
the `BAMProperties` economy state, and all firm/household/bank agents added with
the Mesa initial values.

Derived initial values (Mesa `BamModel.__init__`):
  * `production_init = n_households * labor_productivity / n_firms`
  * `wage_offer_init = price_init / 3`
  * `net_worth_init  = production_init * price_init * net_worth_ratio`
  * `min_wage        = wage_offer_init * min_wage_ratio`
  * `avg_mkt_price   = price_init`

`model.n_households` is set (via the properties struct) before firms are added,
matching the Mesa ordering where the Firm init reads `n_households`.

The step function is the no-op `bam_step!` for now; the real per-period
pipeline is wired in Task 9.
"""
function build_model(n_firms::Integer, n_households::Integer, n_banks::Integer,
                     params::AbstractDict, seed::Integer)
    # Normalise params to a concrete Dict{String,Float64} for type stability.
    p = Dict{String,Float64}(string(k) => Float64(v) for (k, v) in params)
    eps = 1e-9

    lp = p["labor_productivity"]
    price_init = p["price_init"]
    production_init = n_households * lp / n_firms
    wage_offer_init = price_init / 3.0
    net_worth_init = production_init * price_init * p["net_worth_ratio"]

    props = BAMProperties(
        p,                              # params
        eps,                            # eps
        0,                              # period
        Int(n_firms),                   # n_firms
        Int(n_households),              # n_households (set before adding firms)
        Int(n_banks),                   # n_banks
        price_init,                     # avg_mkt_price
        wage_offer_init * p["min_wage_ratio"],  # min_wage
        Int(round(p["min_wage_rev_period"])),   # min_wage_rev_period
        Float64[price_init],            # avg_mkt_price_history
        Float64[0.0],                   # inflation_history
        Loan[],                         # loans (empty loan book)
        false,                          # collapsed
    )

    model = StandardABM(
        BAMAgent, nothing;
        model_step! = bam_step!,
        properties = props,
        rng = Xoshiro(seed),
        warn = false,
    )

    # Add firms. `model.n_households` is already set above (Firm init in Mesa
    # reads it to derive production_prev, here passed explicitly as
    # production_init).
    for _ in 1:n_firms
        add_agent!(
            constructor(BAMAgent, Firm), model;
            production_prev = production_init,
            labor_productivity = lp,
            breakeven_price = price_init,
            price = price_init,
            wage_offer = wage_offer_init,
            total_funds = net_worth_init,
            net_worth = net_worth_init,
        )
    end

    # Add households.
    savings_init = p["savings_init"]
    for _ in 1:n_households
        add_agent!(
            constructor(BAMAgent, Household), model;
            savings = savings_init,
        )
    end

    # Add banks.
    equity_base_init = p["equity_base_init"]
    for _ in 1:n_banks
        add_agent!(
            constructor(BAMAgent, Bank), model;
            equity_base = equity_base_init,
        )
    end

    return model
end

# ---------------------------------------------------------------------------
# Convenience iterators by kind (used by later phase tasks)
# ---------------------------------------------------------------------------

"Return an iterator over all `Firm`-variant agents."
firms(model) = (a for a in allagents(model) if variantof(a) === Firm)

"Return an iterator over all `Household`-variant agents."
households(model) = (a for a in allagents(model) if variantof(a) === Household)

"Return an iterator over all `Bank`-variant agents."
banks(model) = (a for a in allagents(model) if variantof(a) === Bank)
