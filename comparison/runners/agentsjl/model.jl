"""
model.jl - Agents.jl v7 implementation of the baseline BAM model: agent types,
economy-state properties, `build_model`, and the planning phase (events 1-6).

Tasks covered so far:
  * Task 2: agent types (`Firm`, `Household`, `Bank`), `BAMProperties`, and
    `build_model`. The scaffold establishes the Agents.jl v7.0.3 idioms reused
    by every later phase task.
  * Task 3: planning-phase event functions (`_event1_zero_production!` through
    `_event6_fire_excess_workers!`) and the `_planning!` dispatcher, plus a
    `model_step!` skeleton (`bam_step!`) that runs only the planning phase for
    now (later tasks fill in the remaining phases).

Agents.jl v7.0.3 idioms:
  * `@agent struct Variant(NoSpaceAgent) ... end` + `@multiagent BAMAgent(...)`
    merge the three variants into one type-stable sum type.
  * Retrieve variant type with `variantof(a)`, enclosed instance with `variant(a)`.
  * `BAMProperties` is a concrete mutable struct (NOT a Dict) so that `model.field`
    access is type-stable.
  * All randomness flows through `abmrng(model)` (the seeded `Xoshiro`); no global
    RNG is ever used.

Relationship encoding (type stability):
  * A Household's employer is stored as an integer agent id (`employer_id`). The
    sentinel `NO_AGENT = 0` means unemployed (Agents.jl ids start at 1).
  * Loans live in a single shared book `BAMProperties.loans`; each `Loan` records
    both `borrower_id` and `lender_id`. Credit-market tasks filter this vector by
    those ids.

Field names and initial values mirror the Mesa reference in
`comparison/runners/mesa/agents.py` and `comparison/runners/mesa/model.py`.
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
# Planning phase - event functions (events 1-6)
# ---------------------------------------------------------------------------

"""
    _event1_zero_production_and_shock!(model)

Event 1 (firms_decide_desired_production): for each firm in agent-iteration
order, zero current production and apply a production shock to set
`desired_production`.

Formula (Mesa `Firm.decide_desired_production`):
  * `production = 0`
  * `shock ~ U(0, h_rho)` per firm (one draw per firm in agent order)
  * `up = (inventory == 0) AND (price >= p_avg)`
  * `dn = (inventory > 0)  AND (price <  p_avg)`
  * `expected_demand = production_prev`
  * if `up`: `expected_demand *= (1 + shock)`
  * if `dn`: `expected_demand *= (1 - shock)`
  * `desired_production = expected_demand`

RNG alignment with Mesa: Mesa iterates `self.firms` (insertion order, same as
agent-creation order); each call to `model.random.uniform(0, h_rho)` consumes
one draw. Here we iterate `allagents(model)` filtering by `Firm` variant, which
yields firms in the same id-ascending (creation) order, and call
`rand(abmrng(model))` once per firm - preserving the same draw sequence.
"""
function _event1_zero_production_and_shock!(model)
    p_avg = model.avg_mkt_price
    h_rho = model.params["h_rho"]
    rng = abmrng(model)
    for a in allagents(model)
        variantof(a) === Firm || continue
        fv = variant(a)
        fv.production = 0.0
        shock = rand(rng) * h_rho   # U(0, h_rho): rand() in [0,1) * h_rho
        up = fv.inventory == 0.0 && fv.price >= p_avg
        dn = fv.inventory >  0.0 && fv.price <  p_avg
        fv.expected_demand = fv.production_prev
        if up
            fv.expected_demand *= 1.0 + shock
        elseif dn
            fv.expected_demand *= 1.0 - shock
        end
        fv.desired_production = fv.expected_demand
    end
end

"""
    _event2_plan_breakeven_price!(model)

Event 2 (firms_plan_breakeven_price): compute each firm's breakeven price from
its wage bill and the interest on its prior-period loans.

Formula (Mesa `Firm.plan_breakeven_price`):
  * `interest = sum(loan.rate * loan.principal for loan in firm's loans)`
  * `breakeven_price = (wage_bill + interest) / max(desired_production, EPS)`

Loan filtering: the shared loan book `model.loans` stores each loan with its
`borrower_id`; filter by `borrower_id == a.id`.

At t=0 (or whenever no loans exist and wage_bill=0), breakeven_price rounds to
approximately 0. That is correct and matches the Mesa port.
"""
function _event2_plan_breakeven_price!(model)
    eps = model.eps
    loans = model.loans
    for a in allagents(model)
        variantof(a) === Firm || continue
        fv = variant(a)
        # Sum interest over this firm's prior-period loans.
        interest = 0.0
        for loan in loans
            if loan.borrower_id == a.id
                interest += loan.principal * loan.rate
            end
        end
        fv.breakeven_price = (fv.wage_bill + interest) / max(fv.desired_production, eps)
    end
end

"""
    _event3_plan_price!(model)

Event 3 (firms_plan_price): adjust each firm's price based on inventory and its
position vs the market average. Conditions are the COMPLEMENT of event 1.

Formula (Mesa `Firm.plan_price`):
  * `shock ~ U(0, h_eta)` per firm (one draw per firm in agent order)
  * `up = (inventory == 0) AND (price < p_avg)`
  * `dn = (inventory > 0)  AND (price >= p_avg)`
  * if `up`: `price *= (1 + shock)`, then `price = max(price, breakeven_price)`
  * if `dn`: `price *= (1 - shock)`, then `price = max(price, breakeven_price)`
  * (no change if neither condition holds)
"""
function _event3_plan_price!(model)
    p_avg = model.avg_mkt_price
    h_eta = model.params["h_eta"]
    rng = abmrng(model)
    for a in allagents(model)
        variantof(a) === Firm || continue
        fv = variant(a)
        shock = rand(rng) * h_eta
        up = fv.inventory == 0.0 && fv.price <  p_avg
        dn = fv.inventory >  0.0 && fv.price >= p_avg
        if up
            fv.price *= 1.0 + shock
            fv.price = max(fv.price, fv.breakeven_price)
        elseif dn
            fv.price *= 1.0 - shock
            fv.price = max(fv.price, fv.breakeven_price)
        end
    end
end

"""
    _event4_decide_desired_labor!(model)

Event 4 (firms_decide_desired_labor): set `desired_labor` as the ceiling of
`desired_production / labor_productivity`. The `ceil` ratchet is load-bearing
(see model reference Flag 1).

Formula (Mesa `Firm.decide_desired_labor`):
  * `desired_labor = ceil(Int, desired_production / max(labor_productivity, EPS))`
"""
function _event4_decide_desired_labor!(model)
    eps = model.eps
    for a in allagents(model)
        variantof(a) === Firm || continue
        fv = variant(a)
        fv.desired_labor = ceil(Int, fv.desired_production / max(fv.labor_productivity, eps))
    end
end

"""
    _event5_decide_vacancies!(model)

Event 5 (firms_decide_vacancies): open vacancies = max(desired_labor -
current_labor, 0).

Formula (Mesa `Firm.decide_vacancies`):
  * `n_vacancies = max(desired_labor - current_labor, 0)`
"""
function _event5_decide_vacancies!(model)
    for a in allagents(model)
        variantof(a) === Firm || continue
        fv = variant(a)
        fv.n_vacancies = max(fv.desired_labor - fv.current_labor, 0)
    end
end

"""
    _event6_fire_excess_workers!(model)

Event 6 (firms_fire_excess_workers): firms with more workers than desired fire
the excess, chosen uniformly at random.

Formula (Mesa `Firm.fire_excess_workers`):
  * `excess = current_labor - desired_labor`; skip if <= 0
  * shuffle `employee_ids` and fire the first `excess` workers
  * per fired worker: set `employer_id = NO_AGENT`, `employer_prev_id = firm.id`,
    `wage = 0`, `periods_left = 0`, `contract_expired = false`, `fired = true`
  * `current_labor -= 1` per fired worker; remove id from `employee_ids`

RNG: one call to `shuffle!(rng, employee_ids)` per firm that needs to fire,
matching Mesa's `model.random.sample(list(self.employees), k)` which draws k
without replacement from the employee list (equivalent to shuffle + take k).
Mesa samples exactly `k` without replacement using `random.sample`; here we
shuffle the full list and take the first `excess` entries, which is the same
distribution.
"""
function _event6_fire_excess_workers!(model)
    rng = abmrng(model)
    for a in allagents(model)
        variantof(a) === Firm || continue
        fv = variant(a)
        excess = fv.current_labor - fv.desired_labor
        excess <= 0 && continue
        # Shuffle employee_ids in place (random permutation = same as Mesa's sample).
        shuffle!(rng, fv.employee_ids)
        k = min(excess, length(fv.employee_ids))
        victims = fv.employee_ids[1:k]
        for hid in victims
            h = model[hid]
            hv = variant(h)
            hv.employer_id      = NO_AGENT
            hv.employer_prev_id = a.id
            hv.wage             = 0.0
            hv.periods_left     = 0
            hv.contract_expired = false
            hv.fired            = true
        end
        # Remove fired ids from the employee list and decrement current_labor.
        victim_set = Set(victims)
        filter!(id -> !(id in victim_set), fv.employee_ids)
        fv.current_labor -= k
    end
end

# ---------------------------------------------------------------------------
# Planning phase dispatcher
# ---------------------------------------------------------------------------

"""
    _planning!(model)

Phase 1: run events 1-6 (planning + pricing phase) in order.

  1. Zero production + production shock -> desired_production
  2. Breakeven price from wage_bill + prior loan interest
  3. Price adjustment vs avg_mkt_price
  4. Desired labor (ceil ratchet)
  5. Vacancies (desired_labor - current_labor)
  6. Fire excess workers (random selection)

Mirrors Mesa `BamModel._planning()`.
"""
function _planning!(model)
    _event1_zero_production_and_shock!(model)
    _event2_plan_breakeven_price!(model)
    _event3_plan_price!(model)
    _event4_decide_desired_labor!(model)
    _event5_decide_vacancies!(model)
    _event6_fire_excess_workers!(model)
end

# ---------------------------------------------------------------------------
# Per-period step (planning-only scaffold; remaining phases added in later tasks)
# ---------------------------------------------------------------------------

"""
    bam_step!(model)

Per-period model step. Currently runs only Phase 1 (planning, events 1-6);
later tasks (T4-T9) will add labor, credit, production, goods, revenue, and
bankruptcy/entry phases.

Defined as a named function (rather than `dummystep`) so the `StandardABM` is
constructed with a real `model_step!` and later tasks have a single clear hook
to extend.
"""
function bam_step!(model)
    model.collapsed && return nothing
    model.period += 1
    _planning!(model)
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
