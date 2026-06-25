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
  * Task 4: labor-market event functions (events 7-12) and the `_labor_market!`
    dispatcher. The matching round (event 11) lives in `markets.jl`. `bam_step!`
    now runs planning + labor market.
  * Task 5: credit-market event functions (events 13-19) and the `_credit_market!`
    dispatcher. The firm-bank matching round (event 18) lives in `markets.jl`.
    Loans are created into the shared loan book `BAMProperties.loans`; the book
    persists across periods and is purged at the start of `_credit_market!` (event
    17 open), matching the Mesa port. `bam_step!` now runs planning + labor +
    credit. The fire-on-gap step (event 19) reuses the event-6 firing pattern.
  * Task 6: production-phase event functions (events 20-24) and `_production!`
    dispatcher; revenue-phase event functions (events 30-32) and `_revenue!`
    dispatcher. `bam_step!` now runs planning + labor + credit + production +
    revenue. Loans are NOT removed in the revenue phase; the loan book is cleared
    at the next credit-market open (event 17 purge), matching the Mesa port
    (Flag 6). Bank equity is updated per-loan: fully-repaid lenders earn interest;
    defaulting lenders take a proportional loss against pre-update net_worth.
  * Task 7: goods-market event functions (events 25-27, 29) and `_goods_market!`
    dispatcher. The sequential shopping loop (event 28) lives in `markets.jl`.
    `bam_step!` now runs planning + labor + credit + production + goods + revenue.
    Consumer propensity uses a tanh-based formula; firm selection uses
    shuffle-and-take sampling (same pattern as events 10 and 17) with loyalty
    tracking via integer ids and price-ASC sorting (MergeSort for stability).

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

# Market-matching routines (labor market here; credit/goods added in later tasks).
include(joinpath(@__DIR__, "markets.jl"))

# Sentinel for "no agent" (unemployed worker, no previous employer, no loyalty
# target, etc.). Agents.jl assigns ids starting at 1, so 0 is always free.
const NO_AGENT = 0

# ---------------------------------------------------------------------------
# O(k) uniform without-replacement sampler
# ---------------------------------------------------------------------------

"""
    _sample_k_from_range!(rng, lo, hi, k) -> Vector{Int}

Draw `k` distinct integers from `lo:hi` uniformly without replacement in O(k)
expected time via rejection sampling. Returns a freshly-allocated `Vector{Int}`
of length `min(k, hi-lo+1)`.

k is always small (max_M, max_Z, max_H <= 5 in the baseline calibration), so
the expected number of rejections is negligible even for the largest populations
used in benchmarks (10 000+ firms). This replaces the O(F) `copy(pool); shuffle!`
pattern used in events 10 and 27 (and the O(B) pattern in event 17 for banks).

All randomness flows through `rng` (always `abmrng(model)`).
"""
function _sample_k_from_range!(rng::AbstractRNG, lo::Int, hi::Int, k::Int)
    n = hi - lo + 1
    k = min(k, n)
    k == 0 && return Int[]
    result = Vector{Int}(undef, k)
    seen   = Set{Int}()
    sizehint!(seen, k)
    i = 0
    while i < k
        x = rand(rng, lo:hi)
        if x ∉ seen
            push!(seen, x)
            i += 1
            result[i] = x
        end
    end
    return result
end

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

Collection fields (populated when `collect == true`):
  * `collect` - flag: whether to collect per-period data.
  * `c_unemployment` - per-period unemployment rate (1 - employed/n_households).
  * `c_avg_employed_wage` - per-period average wage of employed workers.
  * `c_total_production` - per-period total production (sum over firms).
  * `c_total_vacancies` - per-period total open vacancies (sum over firms).
  * `c_inflation` - per-period price inflation (YoY from inflation_history).
  * `c_production_final` - per-firm production snapshot at the LAST period
    (overwritten each period; length == n_firms after the run completes).
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
    # Collection fields (Task 9: per-period series)
    collect::Bool
    c_unemployment::Vector{Float64}
    c_avg_employed_wage::Vector{Float64}
    c_total_production::Vector{Float64}
    c_total_vacancies::Vector{Float64}
    c_inflation::Vector{Float64}
    c_production_final::Vector{Float64}
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

RNG alignment with Mesa: Mesa iterates `self.firms` (insertion order = id order).
Here we iterate `allagents(model)` in Dict (hash) order, which preserves the
original gate-passing draw sequence. This event calls `rand(abmrng(model))` once
per firm regardless of visit order, so the per-firm draw cadence matches Mesa's;
the draws are i.i.d., so visit order does not affect the distribution (the gate
confirms equivalence).
"""
function _event1_zero_production_and_shock!(model)
    p_avg = model.avg_mkt_price
    h_rho = model.params["h_rho"]
    rng = abmrng(model)
    # Iterate in allagents (hash) order to preserve the original RNG draw sequence.
    # One rand() per firm regardless of visit order; hash order matches the original code.
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
`borrower_id`; filter by `borrower_id == fid` (the firm's integer id).

At t=0 (or whenever no loans exist and wage_bill=0), breakeven_price rounds to
approximately 0. That is correct and matches the Mesa port.
"""
function _event2_plan_breakeven_price!(model)
    eps = model.eps
    loans = model.loans
    for fid in 1:model.n_firms
        a  = model[fid]
        fv = variant(a)
        # Sum interest over this firm's prior-period loans.
        interest = 0.0
        for loan in loans
            if loan.borrower_id == fid
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
    # Iterate in allagents (hash) order to preserve the original RNG draw sequence.
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
    for fid in 1:model.n_firms
        fv = variant(model[fid])
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
    for fid in 1:model.n_firms
        fv = variant(model[fid])
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
    # Iterate in allagents (hash) order to preserve the original RNG draw sequence.
    # Each firm with excess workers consumes one shuffle; hash order matches original.
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
            hv = variant(model[hid])
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
     Collection point (when collect=true): total_vacancies after event 5,
     matching Mesa `BamModel._planning` which collects before event 6.
  6. Fire excess workers (random selection)

Mirrors Mesa `BamModel._planning()`.
"""
function _planning!(model)
    _event1_zero_production_and_shock!(model)
    _event2_plan_breakeven_price!(model)
    _event3_plan_price!(model)
    _event4_decide_desired_labor!(model)
    _event5_decide_vacancies!(model)
    # Event 5 collection: total vacancies after decide_vacancies (Mesa point).
    if model.collect
        total_vac = 0
        for fid in 1:model.n_firms
            total_vac += variant(model[fid]).n_vacancies
        end
        push!(model.c_total_vacancies, Float64(total_vac))
    end
    _event6_fire_excess_workers!(model)
    return nothing
end

# ---------------------------------------------------------------------------
# Labor market phase - event functions (events 7-12)
# ---------------------------------------------------------------------------

"""
    _event7_calc_inflation!(model)

Event 7 (calc_inflation_rate): year-over-year inflation from the price history,
appended to `inflation_history`.

Formula (Mesa `BamModel._calc_inflation`):
  * if `length(avg_mkt_price_history) <= 4`: append `0.0`
  * else `p_now = hist[end]`, `p_prev = hist[end-4]`
    (the value 5 entries back, matching Python's `hist[-5]`)
  * append `0.0` if `p_prev <= 0` else `(p_now - p_prev) / p_prev`

No RNG.
"""
function _event7_calc_inflation!(model)
    hist = model.avg_mkt_price_history
    if length(hist) <= 4
        push!(model.inflation_history, 0.0)
        return nothing
    end
    p_now = hist[end]
    p_prev = hist[end-4]   # Python hist[-5]
    if p_prev <= 0.0
        push!(model.inflation_history, 0.0)
    else
        push!(model.inflation_history, (p_now - p_prev) / p_prev)
    end
    return nothing
end

"""
    _event8_adjust_min_wage!(model)

Event 8 (adjust_minimum_wage): periodically index the minimum wage to inflation.

Formula (Mesa `BamModel._adjust_min_wage`):
  * `m = min_wage_rev_period`; `hist_len = length(avg_mkt_price_history)`
  * skip if `hist_len <= m`
  * skip unless `(hist_len - 1) % m == 0`
  * `inflation = inflation_history[end]`; `min_wage *= (1 + inflation)`
    (bidirectional - can fall when inflation is negative)
  * bump every EMPLOYED worker whose wage is below the new floor up to it.

"Employed" is derived as `employer_id != NO_AGENT`. No RNG.
"""
function _event8_adjust_min_wage!(model)
    m = model.min_wage_rev_period
    hist_len = length(model.avg_mkt_price_history)
    hist_len <= m && return nothing
    (hist_len - 1) % m != 0 && return nothing
    inflation = model.inflation_history[end]
    model.min_wage *= 1.0 + inflation
    new_floor = model.min_wage
    for h in households(model)
        hv = variant(h)
        if hv.employer_id != NO_AGENT && hv.wage < new_floor
            hv.wage = new_floor
        end
    end
    return nothing
end

"""
    _event9_decide_wage_offer!(model)

Event 9 (firms_decide_wage_offer): set each firm's posted wage offer with a
random markup, floored at the minimum wage.

Formula (Mesa `Firm.decide_wage_offer`):
  * `shock ~ U(0, h_xi)` per firm, but ONLY when `n_vacancies > 0`; otherwise
    `shock = 0.0` and NO draw is consumed
  * `wage_offer *= (1 + shock)`
  * `wage_offer = max(wage_offer, min_wage)`

RNG alignment with Mesa: one `rand(rng)` per firm WITH vacancies, matching
Mesa's `self.firms` iteration where the `uniform(0, h_xi)` draw is taken only in
the `n_vacancies > 0` branch. We iterate `allagents(model)` in Dict (hash) order,
which preserves the original gate-passing draw sequence; the draws are i.i.d., so
visit order does not affect the distribution. Firms without vacancies consume no
draw in either implementation.
"""
function _event9_decide_wage_offer!(model)
    h_xi = model.params["h_xi"]
    min_wage = model.min_wage
    rng = abmrng(model)
    # Iterate in allagents (hash) order to preserve the original RNG draw sequence.
    # Only firms WITH vacancies consume a draw; hash order matches the original code.
    for a in allagents(model)
        variantof(a) === Firm || continue
        fv = variant(a)
        shock = fv.n_vacancies > 0 ? rand(rng) * h_xi : 0.0
        fv.wage_offer *= 1.0 + shock
        fv.wage_offer = max(fv.wage_offer, min_wage)
    end
    return nothing
end

"""
    _event10_decide_firms_to_apply!(model)

Event 10 (workers_decide_firms_to_apply): each UNEMPLOYED worker builds a ranked
application queue.

Formula (Mesa `Household.decide_firms_to_apply`):
  * skip employed workers (`employer_id != NO_AGENT`)
  * pool = ALL firm ids; `M_eff = min(max_M, |pool|)`; sample `M_eff` firms
    WITHOUT replacement
  * sort the sample by `wage_offer` DESC
  * loyalty: if `contract_expired AND !fired AND employer_prev_id` is a firm in
    the pool, move `employer_prev_id` to the FRONT (dropping the last entry if
    it was not already in the sample, to keep length `M_eff`)
  * store as `job_apps`; reset `contract_expired = false`, `fired = false`

RNG: one O(k) without-replacement draw per unemployed worker via
`_sample_k_from_range!`. Firms are added first in `build_model`, so firm ids
are exactly the contiguous range `1:n_firms` (stable: bankruptcy replaces in
place, never removes). Drawing from this range produces the same uniform k-subset
distribution as the previous `copy(pool); shuffle!` approach, and consumes the
same number of `rand(rng, ...)` calls per worker (exactly k random draws in the
uncontested case, versus k*F/2 advances through an internal shuffle). This
eliminates the O(F) copy+shuffle per worker (O(F*W) per period, the confirmed
hotspot) and replaces it with O(k) per worker.
"""
function _event10_decide_firms_to_apply!(model)
    rng   = abmrng(model)
    max_M = Int(round(model.params["max_M"]))
    npool = model.n_firms       # firm ids are exactly 1:n_firms (contiguous, stable)

    # Lookup of firm wage_offer by id for DESC ranking.
    wage_offer_of(fid) = variant(model[fid]).wage_offer

    for h in households(model)
        hv = variant(h)
        hv.employer_id == NO_AGENT || continue   # employed workers skip

        M_eff = min(max_M, npool)
        # O(k) uniform without-replacement draw from 1:n_firms.
        sample = _sample_k_from_range!(rng, 1, npool, M_eff)

        # Sort by wage_offer DESC. Use a STABLE sort so ties keep the sampled
        # order, matching Python's stable `list.sort`.
        sort!(sample; by = wage_offer_of, rev = true, alg = MergeSort)

        # Loyalty: move employer_prev_id to the front if eligible.
        prev = hv.employer_prev_id
        if hv.contract_expired && !hv.fired && prev != NO_AGENT && 1 <= prev <= npool
            idx = findfirst(==(prev), sample)
            if idx !== nothing
                deleteat!(sample, idx)
            elseif length(sample) == M_eff
                # Drop the last to keep the M_eff length.
                sample = sample[1:M_eff-1]
            end
            pushfirst!(sample, prev)
        end

        hv.job_apps = sample
        hv.contract_expired = false
        hv.fired = false
    end
    return nothing
end

"""
    _event12_calc_wage_bill!(model)

Event 12 (firms_calc_wage_bill): each firm's wage bill = sum of its employees'
wages.

Formula (Mesa `Firm.calc_wage_bill`):
  * `wage_bill = sum(employee.wage for employee in employees)`

No RNG.
"""
function _event12_calc_wage_bill!(model)
    for fid in 1:model.n_firms
        fv = variant(model[fid])
        total = 0.0
        for hid in fv.employee_ids
            total += variant(model[hid]).wage
        end
        fv.wage_bill = total
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Labor market phase dispatcher
# ---------------------------------------------------------------------------

"""
    _labor_market!(model)

Phase 2: run events 7-12 (labor market) in order.

  7.  Calc YoY inflation -> inflation_history
      Collection point (when collect=true): inflation after event 7,
      matching Mesa `BamModel._labor_market` which collects before event 8.
  8.  Adjust minimum wage (periodic, indexed to inflation)
  9.  Firms decide wage offer (random markup, min-wage floor)
  10. Unemployed workers decide which firms to apply to (ranked queue)
  11. Labor market matching (x max_M rounds; conflict-resolved hiring)
  12. Firms calc wage bill (sum of employees' wages)

Mirrors Mesa `BamModel._labor_market()`.
"""
function _labor_market!(model)
    _event7_calc_inflation!(model)
    # Event 7 collection: inflation after _calc_inflation (Mesa point).
    if model.collect
        push!(model.c_inflation, model.inflation_history[end])
    end
    _event8_adjust_min_wage!(model)
    _event9_decide_wage_offer!(model)
    _event10_decide_firms_to_apply!(model)
    _run_labor_market!(model)        # event 11 (defined in markets.jl)
    _event12_calc_wage_bill!(model)
    return nothing
end

# ---------------------------------------------------------------------------
# Credit market phase - event functions (events 13-19)
# ---------------------------------------------------------------------------

"""
    _purge_loans!(model)

Event 17 open (loan-book purge): clear the previous period's loans at the START
of the credit market.

The shared loan book `model.loans` PERSISTS across the planning and labor phases
so event 2 (`plan_breakeven_price`) can read last period's per-firm interest.
The Mesa port clears `f.loans = []` for every firm at the very start of
`_credit_market`; here we empty the single shared vector once, which is exactly
equivalent (every loan, regardless of borrower, is dropped). New loans for THIS
period are then pushed during event 18.

No RNG.
"""
function _purge_loans!(model)
    empty!(model.loans)
    return nothing
end

"""
    _event13_decide_credit_supply!(model)

Event 13 (banks_decide_credit_supply): each bank's lending capacity from its
equity base and the capital-requirement coefficient.

Formula (Mesa `Bank.decide_credit_supply`):
  * `credit_supply = max(equity_base / v, 0)`

No RNG.
"""
function _event13_decide_credit_supply!(model)
    v = model.params["v"]
    for b in banks(model)
        bv = variant(b)
        bv.credit_supply = max(bv.equity_base / v, 0.0)
    end
    return nothing
end

"""
    _event14_decide_interest_rate!(model)

Event 14 (banks_decide_interest_rate): each bank draws an operating-expense shock
and sets its POSTED interest rate (used to rank banks in event 17).

Formula (Mesa `Bank.decide_interest_rate`):
  * `opex_shock ~ U(0, h_phi)` per bank (one draw per bank in agent order)
  * `interest_rate = r_bar * (1 + opex_shock)`

This is the POSTED rate (Flag 5): it ranks banks during application preparation.
The CONTRACT rate charged on an actual loan is fragility-scaled and computed in
event 18 (see `_run_credit_market!`).

RNG alignment with Mesa: one `rand(rng)` per bank, matching Mesa's `self.banks`
iteration with one `uniform(0, h_phi)` draw each. `banks(model)` iterates in
Dict (hash) order, not creation order; because every bank consumes exactly one
draw, the total draw count matches Mesa regardless of iteration order.
"""
function _event14_decide_interest_rate!(model)
    h_phi = model.params["h_phi"]
    r_bar = model.params["r_bar"]
    rng = abmrng(model)
    for b in banks(model)
        bv = variant(b)
        bv.opex_shock = rand(rng) * h_phi          # U(0, h_phi)
        bv.interest_rate = r_bar * (1.0 + bv.opex_shock)
    end
    return nothing
end

"""
    _event15_decide_credit_demand!(model)

Event 15 (firms_decide_credit_demand): each firm's external financing need.

Formula (Mesa `Firm.decide_credit_demand`):
  * `credit_demand = max(wage_bill - total_funds, 0)`

No RNG.
"""
function _event15_decide_credit_demand!(model)
    for fid in 1:model.n_firms
        fv = variant(model[fid])
        fv.credit_demand = max(fv.wage_bill - fv.total_funds, 0.0)
    end
    return nothing
end

"""
    _event16_calc_fragility!(model)

Event 16 (firms_calc_fragility): each firm's projected financial fragility,
used to rank applicants at each bank during event 18.

Formula (Mesa `Firm.calc_fragility`):
  * if `net_worth > 0`: `projected_fragility = credit_demand / net_worth`
  * else: `projected_fragility = max_leverage`

No RNG.
"""
function _event16_calc_fragility!(model)
    max_leverage = model.params["max_leverage"]
    for fid in 1:model.n_firms
        fv = variant(model[fid])
        if fv.net_worth > 0.0
            fv.projected_fragility = fv.credit_demand / fv.net_worth
        else
            fv.projected_fragility = max_leverage
        end
    end
    return nothing
end

"""
    _event17_prepare_loan_applications!(model)

Event 17 (firms_prepare_loan_applications): each firm with positive credit demand
samples up to `max_H` banks that still have supply and ranks them by POSTED
interest rate ASC (cheapest first).

Formula (Mesa `Firm.prepare_loan_applications`):
  * if `credit_demand <= 0`: `loan_apps = []`
  * `lenders = [bank ids with credit_supply > 0]`; `H_eff = min(max_H, |lenders|)`
  * if `H_eff == 0`: `loan_apps = []`
  * sample `H_eff` lenders WITHOUT replacement; sort by `interest_rate` ASC
  * store as `loan_apps` (consumed one per round in event 18)

RNG: firm-only loops now iterate over `1:n_firms` directly (O(F), not O(F+W+B)).
Bank ids are `n_firms+n_households+1 : n_firms+n_households+n_banks` (banks are
added last in `build_model`, stable). Eligible banks (credit_supply > 0) are
collected into a small sorted vector; then for each applying firm an O(k)
without-replacement draw via `_sample_k_from_range!` over the eligible-bank
index range replaces the O(B) `copy(lenders); shuffle!`. H_eff is at most
max_H (typically 2-3), so the rejection sampler is very fast.
"""
function _event17_prepare_loan_applications!(model)
    rng   = abmrng(model)
    max_H = Int(round(model.params["max_H"]))

    # Eligible lenders: bank ids with positive supply, sorted ascending by id.
    # Bank ids are n_firms+n_households+1 : n_firms+n_households+n_banks
    # (banks added last in build_model); we still filter by credit_supply > 0
    # so the eligible set can be smaller.
    bank_lo = model.n_firms + model.n_households + 1
    bank_hi = model.n_firms + model.n_households + model.n_banks
    lenders = sort!([id for id in bank_lo:bank_hi
                     if variant(model[id]).credit_supply > 0.0])
    n_lenders = length(lenders)

    # Lookup of posted interest rate by bank id for ASC ranking.
    rate_of(bid) = variant(model[bid]).interest_rate

    # Iterate firms by direct id range (O(F), no allagents filter scan).
    for fid in 1:model.n_firms
        fv = variant(model[fid])
        if fv.credit_demand <= 0.0
            fv.loan_apps = Int[]
            continue
        end
        H_eff = min(max_H, n_lenders)
        if H_eff == 0
            fv.loan_apps = Int[]
            continue
        end
        # O(k) draw of H_eff distinct indices into the `lenders` vector, then
        # map to actual bank ids. Drawing indices (1:n_lenders) and indexing
        # into `lenders` is equivalent to drawing from the lenders vector
        # directly, and avoids an O(n_lenders) copy+shuffle.
        idx_sample = _sample_k_from_range!(rng, 1, n_lenders, H_eff)
        sample = lenders[idx_sample]
        # Sort by posted interest_rate ASC. STABLE sort so ties keep the sampled
        # order, matching Python's stable `list.sort`.
        sort!(sample; by = rate_of, alg = MergeSort)
        fv.loan_apps = sample
    end
    return nothing
end

"""
    _event19_fire_workers_for_gap!(model)

Event 19 (firms_fire_workers_for_gap): firms whose wage bill still exceeds their
available funds (after any loans) fire workers until the gap is covered.

Formula (Mesa `Firm.fire_workers_for_gap`):
  * skip if `wage_bill <= total_funds`
  * `gap = wage_bill - total_funds`
  * shuffle the firm's employees; fire them one at a time, accumulating their
    wages, until the cumulative fired wage `>= gap` (then stop)
  * per fired worker: `employer_id = NO_AGENT`, `employer_prev_id = firm.id`,
    `wage = 0`, `periods_left = 0`, `contract_expired = false`, `fired = true`;
    `current_labor -= 1`; remove id from `employee_ids`; `wage_bill -= wage`

RNG alignment with Mesa: one `shuffle!(rng, employee_ids)` per firm WITH a gap,
matching Mesa's `model.random.shuffle(employees_list)`. We iterate `1:n_firms`
(id-ascending); each firm with a gap consumes exactly one shuffle draw, matching
Mesa's draw cadence. Reuses the event-6 firing field pattern.
"""
function _event19_fire_workers_for_gap!(model)
    rng = abmrng(model)
    # Iterate in allagents (hash) order to preserve the original RNG draw sequence.
    # Each firm with a gap consumes one shuffle; hash order matches original.
    for a in allagents(model)
        variantof(a) === Firm || continue
        fv = variant(a)
        fv.wage_bill <= fv.total_funds && continue
        gap = fv.wage_bill - fv.total_funds
        # Shuffle a copy of the employee ids (Mesa shuffles a list copy).
        order = shuffle!(rng, copy(fv.employee_ids))
        cumulative = 0.0
        fired_ids = Int[]
        for hid in order
            hv = variant(model[hid])
            w = hv.wage
            hv.employer_id      = NO_AGENT
            hv.employer_prev_id = a.id
            hv.wage             = 0.0
            hv.periods_left     = 0
            hv.contract_expired = false
            hv.fired            = true
            fv.current_labor   -= 1
            fv.wage_bill       -= w
            cumulative         += w
            push!(fired_ids, hid)
            cumulative >= gap && break
        end
        # Remove the fired ids from the employee list (preserving order of the rest).
        if !isempty(fired_ids)
            fired_set = Set(fired_ids)
            filter!(id -> !(id in fired_set), fv.employee_ids)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Credit market phase dispatcher
# ---------------------------------------------------------------------------

"""
    _credit_market!(model)

Phase 3: run events 13-19 (credit market) in order.

  17a. Purge previous-period loans (retained through planning/labor for breakeven)
  13.  Banks decide credit supply (equity_base / v)
  14.  Banks decide POSTED interest rate (r_bar * (1 + opex_shock))
  15.  Firms decide credit demand (max(wage_bill - total_funds, 0))
  16.  Firms calc projected fragility (credit_demand / net_worth, or max_leverage)
  17.  Firms prepare loan applications (sample max_H lenders, rank by posted rate)
  18.  Credit market matching (x max_H rounds; fragility ASC; partial at boundary;
       contract rate fragility-scaled; loans pushed to shared book)
  19.  Firms fire workers for any remaining wage-bill gap

Mirrors Mesa `BamModel._credit_market()`. The loan-book purge happens first
(event 17 open), so loans created here are this period's only.
"""
function _credit_market!(model)
    _purge_loans!(model)                       # event 17 open: clear prior loans
    _event13_decide_credit_supply!(model)
    _event14_decide_interest_rate!(model)
    _event15_decide_credit_demand!(model)
    _event16_calc_fragility!(model)
    _event17_prepare_loan_applications!(model)
    _run_credit_market!(model)                 # event 18 (defined in markets.jl)
    _event19_fire_workers_for_gap!(model)
    return nothing
end

# ---------------------------------------------------------------------------
# Production phase - event functions (events 20-24)
# ---------------------------------------------------------------------------

"""
    _event20_pay_wages!(model)

Event 20 (firms_pay_wages): each firm deducts its wage bill from total funds.

Formula (Mesa `Firm.pay_wages`):
  * `total_funds -= wage_bill`

No RNG.
"""
function _event20_pay_wages!(model)
    for fid in 1:model.n_firms
        fv = variant(model[fid])
        fv.total_funds -= fv.wage_bill
    end
    return nothing
end

"""
    _event21_receive_wage!(model)

Event 21 (workers_receive_wage): each employed household adds its wage to income.

Formula (Mesa `Household.receive_wage`):
  * if employed (`employer_id != NO_AGENT`): `income += wage`

No RNG.
"""
function _event21_receive_wage!(model)
    for h in households(model)
        hv = variant(h)
        if hv.employer_id != NO_AGENT
            hv.income += hv.wage
        end
    end
    return nothing
end

"""
    _event22_run_production!(model)

Event 22 (firms_run_production): each firm produces output, overwrites inventory,
and updates production_prev.

Formula (Mesa `Firm.run_production`):
  * `production = labor_productivity * current_labor`
  * `production_prev = production`   (unconditional every period; Flag 2)
  * `inventory = production`         (OVERWRITE, not accumulate; Flag 4)

No RNG.
"""
function _event22_run_production!(model)
    for fid in 1:model.n_firms
        fv = variant(model[fid])
        fv.production = fv.labor_productivity * fv.current_labor
        fv.production_prev = fv.production
        fv.inventory = fv.production
    end
    return nothing
end

"""
    _event23_update_avg_mkt_price!(model)

Event 23 (update_avg_mkt_price): production-weighted average price over firms with
production >= 1e-3; keep previous avg_mkt_price if result <= 0; append to history.

Formula (Mesa `BamModel._update_avg_mkt_price`):
  * weighted_sum = sum(price * production for firms with production >= 1e-3)
  * total_prod   = sum(production for firms with production >= 1e-3)
  * new_price = weighted_sum / total_prod  if total_prod > 0  else 0.0
  * if new_price > 0: avg_mkt_price = new_price  (else keep previous; Flag 10)
  * append avg_mkt_price to avg_mkt_price_history

No RNG.
"""
function _event23_update_avg_mkt_price!(model)
    total_prod = 0.0
    weighted_sum = 0.0
    for fid in 1:model.n_firms
        fv = variant(model[fid])
        if fv.production >= 1e-3
            weighted_sum += fv.price * fv.production
            total_prod   += fv.production
        end
    end
    new_price = total_prod > 0.0 ? weighted_sum / total_prod : 0.0
    if new_price > 0.0
        model.avg_mkt_price = new_price
    end
    # else: keep previous avg_mkt_price (Flag 10)
    push!(model.avg_mkt_price_history, model.avg_mkt_price)
    return nothing
end

"""
    _event24_update_contracts!(model)

Event 24 (workers_update_contracts): decrement each employed worker's
`periods_left`; workers whose contract expires (`periods_left` reaches 0) are
separated from their employer.

Formula (Mesa `Household.update_contract`):
  * skip unemployed workers (`employer_id == NO_AGENT`)
  * `periods_left -= 1`
  * if `periods_left == 0`:
    - `employer_prev_id = employer_id`
    - `employer_id = NO_AGENT`
    - `wage = 0.0`
    - `contract_expired = true`
    - `fired = false`
    - remove this worker id from employer's `employee_ids`
    - `employer.current_labor -= 1`

Note: wage_bill is NOT recomputed here; the next period's event 12 handles it.
No RNG. (`contract_expired=1, fired=0` => worker is loyal next period.)
"""
function _event24_update_contracts!(model)
    for h in households(model)
        hv = variant(h)
        hv.employer_id == NO_AGENT && continue
        hv.periods_left -= 1
        if hv.periods_left == 0
            employer_id = hv.employer_id
            hv.employer_prev_id = employer_id
            hv.employer_id      = NO_AGENT
            hv.wage             = 0.0
            hv.contract_expired = true
            hv.fired            = false
            # Update the firm's employee roster and labor count.
            fv = variant(model[employer_id])
            filter!(id -> id != h.id, fv.employee_ids)
            fv.current_labor -= 1
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Production phase dispatcher
# ---------------------------------------------------------------------------

"""
    _production!(model)

Phase 4: run events 20-24 (production phase) in order.

  20. Firms pay wages (total_funds -= wage_bill)
  21. Workers receive wage (employed workers: income += wage)
  22. Firms run production (labor_productivity * current_labor; overwrite inventory)
      Collection point (when collect=true): unemployment, avg_employed_wage,
      total_production, and production_final (per-firm snapshot) after event 22,
      matching Mesa `BamModel._production` which collects before event 23.
  23. Update production-weighted average market price (append to history)
  24. Workers update contracts (decrement periods_left; expire if == 0)

Mirrors Mesa `BamModel._production()`.
"""
function _production!(model)
    _event20_pay_wages!(model)
    _event21_receive_wage!(model)
    _event22_run_production!(model)
    # Event 22 collection: employment and production after run_production (Mesa point).
    if model.collect
        employed_count = 0
        wage_sum = 0.0
        for h in households(model)
            hv = variant(h)
            if hv.employer_id != NO_AGENT
                employed_count += 1
                wage_sum += hv.wage
            end
        end
        n_hh = model.n_households
        unemployment = n_hh > 0 ? 1.0 - Float64(employed_count) / n_hh : 1.0
        avg_wage = employed_count > 0 ? wage_sum / employed_count : 0.0
        total_prod = 0.0
        prod_final = Float64[]
        for fid in 1:model.n_firms
            p = variant(model[fid]).production
            total_prod += p
            push!(prod_final, p)
        end
        push!(model.c_unemployment, unemployment)
        push!(model.c_avg_employed_wage, avg_wage)
        push!(model.c_total_production, total_prod)
        # production_final is overwritten each period; last period's snapshot is used.
        model.c_production_final = prod_final
    end
    _event23_update_avg_mkt_price!(model)
    _event24_update_contracts!(model)
    return nothing
end

# ---------------------------------------------------------------------------
# Goods market phase - event functions (events 25-27, 29)
# ---------------------------------------------------------------------------

"""
    _event25_calc_propensity!(model)

Event 25 (consumers_calc_propensity): compute each household's marginal propensity
to consume, based on its savings relative to the economy-wide average.

Formula (Mesa `Household.calc_propensity`):
  * `avg_sav = max(mean(h.savings for all h), eps)`  (raw savings, floor only the mean)
  * per household: `s = max(savings, 0)`;
    `propensity = 1 / (1 + tanh(s / avg_sav)^beta)`

The savings average uses RAW (possibly negative) savings summed across all
households; only the final mean is floored at `eps`. This matches Mesa exactly:
`avg_sav = max(sum(all_savings) / len(all_savings), EPS)`. The individual
`s = max(savings, 0)` inside the propensity formula is correct and unchanged.

No RNG.
"""
function _event25_calc_propensity!(model)
    eps  = model.eps
    beta = model.params["beta"]

    # Average of RAW savings across all households (floor only the final mean).
    # Matches Mesa: avg_sav = max(sum(all_savings) / len(all_savings), EPS)
    total_sav = 0.0
    n_hh = 0
    for h in households(model)
        total_sav += variant(h).savings
        n_hh += 1
    end
    avg_sav = n_hh > 0 ? max(total_sav / n_hh, eps) : eps

    for h in households(model)
        hv = variant(h)
        s  = max(hv.savings, 0.0)
        hv.propensity = 1.0 / (1.0 + tanh(s / avg_sav)^beta)
    end
    return nothing
end

"""
    _event26_decide_income_to_spend!(model)

Event 26 (consumers_decide_income_to_spend): each household splits wealth into
spending budget and residual savings.

Formula (Mesa `Household.decide_income_to_spend`):
  * `wealth = savings + income`
  * `income_to_spend = wealth * propensity`
  * `savings = wealth - income_to_spend`
  * `income = 0.0`

No RNG.
"""
function _event26_decide_income_to_spend!(model)
    for h in households(model)
        hv = variant(h)
        wealth = hv.savings + hv.income
        hv.income_to_spend = wealth * hv.propensity
        hv.savings         = wealth - hv.income_to_spend
        hv.income          = 0.0
    end
    return nothing
end

"""
    _event27_decide_firms_to_visit!(model)

Event 27 (consumers_decide_firms_to_visit): each household samples up to `max_Z`
firms, applies the loyalty rule, sorts by price ASC, and updates its loyalty
target for the next period.

Formula (Mesa `Household.decide_firms_to_visit`):
  * if `income_to_spend <= eps`: `shop_visits = []`; return
  * pool = ALL firm ids, sorted ascending by id (NOT iteration/Dict-hash order);
    `Z = min(max_Z, |pool|)`
  * sample Z firm ids WITHOUT replacement: O(k) draw from `1:n_firms`
  * loyalty (always applied): if `largest_prod_prev_id != NO_AGENT` and that id
    is NOT already in the sample, replace `sample[end]` with it
  * sort by `price` ASC (MergeSort for stability)
  * update loyalty BEFORE shopping: `largest_prod_prev_id = argmax(production)`
    over the selected firm ids (ties broken by first-maximum in sorted-price
    order, matching Julia's `argmax` semantics on an iterator)
  * store the Z ids as `shop_visits`

Note: `consumer_matching` param is NOT stored in `params` (which is
`Dict{String,Float64}`); the canonical model always uses loyalty matching, so
the loyalty rule is applied unconditionally here, matching Mesa's default
`"loyalty"` path.

RNG: one O(k) draw per household with positive budget via `_sample_k_from_range!`.
Firm ids are `1:n_firms` (contiguous, stable). This replaces the O(F) `copy(pool);
shuffle!` per household (the second confirmed O(N^2) hotspot in the profiling
report). The resulting k-subset distribution is identical: uniform over all
k-subsets of `1:n_firms`. The price-ASC sort, loyalty rule, and loyalty-update
logic are unchanged.
"""
function _event27_decide_firms_to_visit!(model)
    eps   = model.eps
    rng   = abmrng(model)
    max_Z = Int(round(model.params["max_Z"]))
    npool = model.n_firms       # firm ids are exactly 1:n_firms (contiguous, stable)

    price_of(fid)      = variant(model[fid]).price
    production_of(fid) = variant(model[fid]).production

    for h in households(model)
        hv = variant(h)
        if hv.income_to_spend <= eps
            hv.shop_visits = Int[]
            continue
        end

        Z = min(max_Z, npool)
        # O(k) uniform without-replacement draw from 1:n_firms.
        sample = _sample_k_from_range!(rng, 1, npool, Z)

        # Loyalty: if the previous best firm is known and not already sampled,
        # replace the last entry with it.
        prev = hv.largest_prod_prev_id
        if prev != NO_AGENT && !(prev in sample)
            sample[end] = prev
        end

        # Sort by price ASC. STABLE sort so ties keep the sampled order,
        # matching Python's stable `list.sort`.
        sort!(sample; by = price_of, alg = MergeSort)

        # Update loyalty: track the firm with the largest production in the
        # selected set (for the NEXT period). Computed BEFORE shopping so the
        # loyalty target reflects the pre-market production snapshot.
        hv.largest_prod_prev_id = sample[argmax(production_of.(sample))]

        hv.shop_visits = sample
    end
    return nothing
end

"""
    _event29_finalize_purchases!(model)

Event 29 (consumers_finalize_purchases): move any unspent budget back into
savings and zero the spending account.

Formula (Mesa `Household.finalize_purchases`):
  * `savings += income_to_spend`
  * `income_to_spend = 0.0`

No RNG.
"""
function _event29_finalize_purchases!(model)
    for h in households(model)
        hv = variant(h)
        hv.savings        += hv.income_to_spend
        hv.income_to_spend = 0.0
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Goods market phase dispatcher
# ---------------------------------------------------------------------------

"""
    _goods_market!(model)

Phase 5: run events 25-29 (goods market) in order.

  25. Consumers calc propensity (tanh-based formula, savings vs avg)
  26. Consumers decide income to spend (split wealth into budget + residual savings)
  27. Consumers decide which firms to visit (sample, loyalty, price-sort)
  28. Goods market sequential shopping loop (shuffle buyers, walk shop_visits)
  29. Consumers finalize purchases (remaining budget -> savings; zero income_to_spend)

Mirrors Mesa `BamModel._goods_market()`.
"""
function _goods_market!(model)
    _event25_calc_propensity!(model)
    _event26_decide_income_to_spend!(model)
    _event27_decide_firms_to_visit!(model)
    _run_goods_market!(model)               # event 28 (defined in markets.jl)
    _event29_finalize_purchases!(model)
    return nothing
end

# ---------------------------------------------------------------------------
# Revenue phase - event functions (events 30-32)
# ---------------------------------------------------------------------------

"""
    _event30_collect_revenue!(model)

Event 30 (firms_collect_revenue): each firm collects revenue from goods sold,
then computes gross profit.

Formula (Mesa `Firm.collect_revenue`):
  * `qty_sold = production - inventory`
  * `revenue = price * qty_sold`
  * `total_funds += revenue`
  * `gross_profit = revenue - wage_bill`

Note: `inventory` is NOT decremented here; it is updated by the goods market
(events 25-29, Task 7). At the start of this task, goods market is not yet
implemented, so revenue collection uses production as the sole measure of
quantity sold (inventory was set to `production` in event 22, so qty_sold = 0
until goods market runs). This is the correct formula per the Mesa source.
No RNG.
"""
function _event30_collect_revenue!(model)
    for fid in 1:model.n_firms
        fv = variant(model[fid])
        qty_sold = fv.production - fv.inventory
        revenue = fv.price * qty_sold
        fv.total_funds += revenue
        fv.gross_profit = revenue - fv.wage_bill
    end
    return nothing
end

"""
    _event31_validate_debt!(model)

Event 31 (firms_validate_debt_commitments): repay loans if solvent; write off
and record losses if not. Update each bank's equity per loan.

Formula (Mesa `Firm.validate_debt`):
  * per firm: filter `model.loans` by `borrower_id == firm.id`
  * `total_debt      = sum(debt(loan)      for each firm loan)`
  * `total_interest  = sum(interest(loan)  for each firm loan)`
  * `total_principal = sum(principal(loan) for each firm loan)`
  * if `total_debt > EPS`:
    - SOLVENT (`total_funds - total_debt >= -EPS`):
        `total_funds -= total_debt`
        per loan: `bank.equity_base += loan.interest`  (Flag 7: interest only on
        fully-repaid loans; lenders earn nothing on defaulted loans)
    - DEFAULT (`total_funds - total_debt < -EPS`):
        `total_funds = 0.0`
        per loan: `frac = principal / max(total_principal, EPS)`
                  `recovery = clamp(frac * net_worth, 0.0, loan.principal)`
                  `loss = loan.principal - recovery`
                  `bank.equity_base -= loss`
        (uses CURRENT pre-update `net_worth`; Flag 7)
  * ALL firms (regardless of debt): `net_profit = gross_profit - total_interest`

Loan settlement timing: loans are NOT removed from `model.loans` here. The loan
book persists so planning-phase event 2 (next period) can read prior interest.
The book is cleared at the start of the next `_credit_market!` (`_purge_loans!`).
This matches Mesa and the model-reference Flag 6.

No RNG.
"""
function _event31_validate_debt!(model)
    eps = model.eps
    loans = model.loans
    for fid in 1:model.n_firms
        fv = variant(model[fid])

        # Gather this firm's loans by filtering the shared book.
        firm_loans = [l for l in loans if l.borrower_id == fid]

        total_debt      = sum(debt(l)      for l in firm_loans; init = 0.0)
        total_interest  = sum(interest(l)  for l in firm_loans; init = 0.0)
        total_principal = sum(l.principal  for l in firm_loans; init = 0.0)

        if total_debt > eps
            if fv.total_funds - total_debt >= -eps
                # Full repayment: deduct debt; each lender earns interest (Flag 7).
                fv.total_funds -= total_debt
                for l in firm_loans
                    bv = variant(model[l.lender_id])
                    bv.equity_base += interest(l)
                end
            else
                # Default: zero out cash; proportional recovery against pre-update NW.
                nw = fv.net_worth   # pre-update net_worth (Flag 7)
                fv.total_funds = 0.0
                for l in firm_loans
                    frac     = l.principal / max(total_principal, eps)
                    recovery = clamp(frac * nw, 0.0, l.principal)
                    loss     = l.principal - recovery
                    bv = variant(model[l.lender_id])
                    bv.equity_base -= loss
                end
            end
        end

        # Net profit computed for ALL firms (including those with no debt).
        fv.net_profit = fv.gross_profit - total_interest
    end
    return nothing
end

"""
    _event32_pay_dividends!(model)

Event 32 (firms_pay_dividends): distribute dividends from profitable firms to
all households equally (Flag 8).

Formula (Mesa `BamModel._pay_dividends`):
  * per firm: `retained = net_profit`
    if `net_profit > 0`: `retained *= (1 - delta)`
    `dividends_paid = net_profit - retained`
    `total_funds -= dividends_paid`
    `retained_profit = retained`
  * `total_dividends = sum(dividends_paid over all firms)`
  * `div_per_hh = total_dividends / n_households`
  * per household: `savings += div_per_hh`; `dividends = div_per_hh`

Note: `net_worth` is NOT updated here (event 33 in Task 8 handles that).
No RNG.
"""
function _event32_pay_dividends!(model)
    delta = model.params["delta"]
    total_dividends = 0.0
    for fid in 1:model.n_firms
        fv = variant(model[fid])
        retained = fv.net_profit
        if fv.net_profit > 0.0
            retained *= 1.0 - delta
        end
        dividends_paid = fv.net_profit - retained
        fv.total_funds    -= dividends_paid
        fv.retained_profit = retained
        total_dividends   += dividends_paid
    end

    n_hh = model.n_households
    div_per_hh = n_hh > 0 ? total_dividends / n_hh : 0.0
    for h in households(model)
        hv = variant(h)
        hv.savings   += div_per_hh
        hv.dividends  = div_per_hh
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Revenue phase dispatcher
# ---------------------------------------------------------------------------

"""
    _revenue!(model)

Phase 6: run events 30-32 (revenue phase) in order.

  30. Firms collect revenue (qty_sold = production - inventory; gross_profit)
  31. Firms validate debt commitments (loan repayment or default; bank equity update)
  32. Firms pay dividends (profitable firms pay delta fraction; households receive share)

Mirrors Mesa `BamModel._revenue()`.
"""
function _revenue!(model)
    _event30_collect_revenue!(model)
    _event31_validate_debt!(model)
    _event32_pay_dividends!(model)
    return nothing
end

# ---------------------------------------------------------------------------
# Bankruptcy + entry phase - event functions (events 33-37)
# ---------------------------------------------------------------------------

"""
    trim_mean(values; pct=0.05) -> Float64

Compute the trimmed mean of `values`, removing `pct` fraction from each tail.
Mirrors the Mesa `trim_mean` helper (model.py ~line 17).

  * Sort values.
  * `k = floor(Int, n * pct)` items removed from each end.
  * If `n - 2k <= 0`, use the full sorted array (nothing trimmed).
  * Return the mean of the trimmed slice, or 0.0 if the input is empty.
"""
function trim_mean(values::AbstractVector{<:Real}; pct::Float64 = 0.05)::Float64
    isempty(values) && return 0.0
    sorted = sort(values)
    n = length(sorted)
    k = floor(Int, n * pct)
    trimmed = (n - 2 * k > 0) ? sorted[k+1 : n-k] : sorted
    return sum(trimmed) / length(trimmed)
end

"""
    _event33_update_net_worth!(model)

Event 33 (firms_update_net_worth): add `retained_profit` to each firm's
`net_worth`; clamp `total_funds` to `max(net_worth, 0)`.

Formula (Mesa `Firm.update_net_worth`):
  * `net_worth += retained_profit`
  * `total_funds = max(net_worth, 0)`

No RNG.
"""
function _event33_update_net_worth!(model)
    for fid in 1:model.n_firms
        fv = variant(model[fid])
        fv.net_worth   += fv.retained_profit
        fv.total_funds  = max(fv.net_worth, 0.0)
    end
    return nothing
end

"""
    _event34_mark_bankrupt_firms!(model)

Event 34 (mark_bankrupt_firms): identify insolvent or ghost firms; fire all
their workers; purge their loans from the shared book.

Bankruptcy conditions (Mesa `_mark_bankrupt_and_replace`):
  * `net_worth < EPS` (insolvency), OR
  * `production_prev <= EPS` (ghost firm: never produced)

Per bankrupt firm:
  * For each employed worker: `employer_id = NO_AGENT`,
    `employer_prev_id = NO_AGENT`, `wage = 0`, `periods_left = 0`,
    `contract_expired = false`, `fired = false`
    (bankruptcy CLEARS the loyalty link, per Mesa event 34; different from
    normal firing which sets `employer_prev_id = firm.id`)
  * `current_labor = 0`, `wage_bill = 0`, `employee_ids = []`

Loans: filter `model.loans` removing any loan whose `borrower_id` is one of
the exiting firms. (A separate pass in event 35 removes loans from exiting
banks, matching Mesa's two-step prune.)

Returns the vector of bankrupt firm ids for use by event 36.

No RNG.
"""
function _event34_mark_bankrupt_firms!(model)
    eps = model.eps
    exiting_firm_ids = Int[]
    for fid in 1:model.n_firms
        fv = variant(model[fid])
        if fv.net_worth < eps || fv.production_prev <= eps
            push!(exiting_firm_ids, fid)
            # Fire all employees: bankruptcy clears the loyalty link.
            for hid in fv.employee_ids
                hv = variant(model[hid])
                hv.employer_id      = NO_AGENT
                hv.employer_prev_id = NO_AGENT   # cleared, not set to firm (event 34)
                hv.wage             = 0.0
                hv.periods_left     = 0
                hv.contract_expired = false
                hv.fired            = false
            end
            fv.employee_ids  = Int[]
            fv.current_labor = 0
            fv.wage_bill     = 0.0
        end
    end
    # Purge bankrupt-firm loans from the shared book.
    if !isempty(exiting_firm_ids)
        exiting_set = Set(exiting_firm_ids)
        filter!(l -> !(l.borrower_id in exiting_set), model.loans)
    end
    return exiting_firm_ids
end

"""
    _event35_mark_bankrupt_banks!(model)

Event 35 (mark_bankrupt_banks): identify insolvent banks; purge their loans
from the shared book (drop loans where `lender_id` is exiting).

Bankruptcy condition:
  * `equity_base < EPS`

Returns the vector of bankrupt bank ids for use by event 37.

No RNG.
"""
function _event35_mark_bankrupt_banks!(model)
    eps = model.eps
    exiting_bank_ids = Int[]
    for b in banks(model)
        bv = variant(b)
        if bv.equity_base < eps
            push!(exiting_bank_ids, b.id)
        end
    end
    # Purge loans issued by bankrupt banks (from surviving firms' perspective).
    if !isempty(exiting_bank_ids)
        exiting_set = Set(exiting_bank_ids)
        filter!(l -> !(l.lender_id in exiting_set), model.loans)
    end
    return exiting_bank_ids
end

"""
    _event36_spawn_replacement_firms!(model, exiting_firm_ids)

Event 36 (spawn_replacement_firms): if all firms are exiting, set
`model.collapsed = true` and return. Otherwise compute survivor statistics
(5%-trimmed means of net_worth, production_prev, employed wages) and reset
each exiting firm in place.

Replacement fields (Mesa `_mark_bankrupt_and_replace` firm-reset block):
  * `net_worth = total_funds = mean_net * new_firm_size_factor`
  * `gross_profit = net_profit = retained_profit = credit_demand =
     projected_fragility = 0.0`
  * `production = production_prev = mean_prod * new_firm_production_factor`
    (> 0 so the replacement is NOT immediately re-ghosted)
  * `inventory = 0.0`, `expected_demand = 0.0`, `desired_production = 0.0`
  * `price = avg_mkt_price * new_firm_price_markup`
  * `current_labor = desired_labor = n_vacancies = 0`
  * `wage_bill = 0.0`
  * `wage_offer = max(mean_wage * new_firm_wage_factor, min_wage)`
  * `loan_apps = []`, `employee_ids = []` (already cleared in event 34)

No RNG.
"""
function _event36_spawn_replacement_firms!(model, exiting_firm_ids::Vector{Int})
    isempty(exiting_firm_ids) && return nothing

    # Collapse check: all firms exiting. n_firms is stable (replace in place).
    if length(exiting_firm_ids) == model.n_firms
        model.collapsed = true
        return nothing
    end

    # Survivor statistics.
    exiting_set = Set(exiting_firm_ids)
    survivor_net_worths  = Float64[]
    survivor_productions = Float64[]
    for fid in 1:model.n_firms
        fid in exiting_set && continue
        fv = variant(model[fid])
        push!(survivor_net_worths,  fv.net_worth)
        push!(survivor_productions, fv.production_prev)
    end

    # Employed wages from all households with employer_id != NO_AGENT and wage > 0.
    employed_wages = Float64[]
    for h in households(model)
        hv = variant(h)
        if hv.employer_id != NO_AGENT && hv.wage > 0.0
            push!(employed_wages, hv.wage)
        end
    end

    mean_net  = trim_mean(survivor_net_worths)
    mean_prod = trim_mean(survivor_productions)
    mean_wage = trim_mean(employed_wages)

    avg_price = model.avg_mkt_price
    min_wage  = model.min_wage
    p = model.params

    nw_val   = mean_net  * p["new_firm_size_factor"]
    prod_val = mean_prod * p["new_firm_production_factor"]
    new_wage = max(mean_wage * p["new_firm_wage_factor"], min_wage)
    new_price = avg_price * p["new_firm_price_markup"]

    for fid in exiting_firm_ids
        fv = variant(model[fid])
        fv.net_worth           = nw_val
        fv.total_funds         = nw_val
        fv.gross_profit        = 0.0
        fv.net_profit          = 0.0
        fv.retained_profit     = 0.0
        fv.credit_demand       = 0.0
        fv.projected_fragility = 0.0
        fv.production          = prod_val
        fv.production_prev     = prod_val
        fv.inventory           = 0.0
        fv.expected_demand     = 0.0
        fv.desired_production  = 0.0
        fv.price               = new_price
        fv.current_labor       = 0
        fv.desired_labor       = 0
        fv.n_vacancies         = 0
        fv.wage_bill           = 0.0
        fv.wage_offer          = new_wage
        fv.loan_apps           = Int[]
        fv.employee_ids        = Int[]
    end
    return nothing
end

"""
    _event37_spawn_replacement_banks!(model, exiting_bank_ids)

Event 37 (spawn_replacement_banks): if no banks survive, set
`model.collapsed = true` and return. Otherwise each exiting bank clones a
RANDOM surviving bank's `equity_base`; `credit_supply` and `interest_rate`
reset to 0.

Mesa: `src = self.random.choice(survivor_banks)` per exiting bank. Here we use
`rand(abmrng(model), survivor_bank_ids)` once per exiting bank in agent-iteration
order, matching the one-draw-per-exit cadence.

RNG: one `rand(rng, collection)` per exiting bank.
"""
function _event37_spawn_replacement_banks!(model, exiting_bank_ids::Vector{Int})
    isempty(exiting_bank_ids) && return nothing

    # Collapse check: no surviving banks. n_banks is stable (replace in place).
    if length(exiting_bank_ids) == model.n_banks
        model.collapsed = true
        return nothing
    end

    exiting_set = Set(exiting_bank_ids)
    bank_lo = model.n_firms + model.n_households + 1
    bank_hi = model.n_firms + model.n_households + model.n_banks
    survivor_bank_ids = [id for id in bank_lo:bank_hi if !(id in exiting_set)]

    rng = abmrng(model)
    for bid in exiting_bank_ids
        src_id = rand(rng, survivor_bank_ids)   # one draw per exiting bank
        src_equity = variant(model[src_id]).equity_base
        bv = variant(model[bid])
        bv.equity_base    = src_equity
        bv.credit_supply  = 0.0
        bv.interest_rate  = 0.0
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Bankruptcy + entry phase dispatcher
# ---------------------------------------------------------------------------

"""
    _bankruptcy_entry!(model)

Phase 7-8: run events 33-37 (bankruptcy detection and firm/bank entry).

  33. Firms update net worth (net_worth += retained_profit; total_funds = max(nw, 0))
  34. Mark bankrupt firms (net_worth < EPS or production_prev <= EPS); fire workers;
      prune their loans; set `exiting_firms`
  35. Mark bankrupt banks (equity_base < EPS); prune their loans; set `exiting_banks`
  36. Spawn replacement firms (collapsed if all firms exit; reset from survivor stats)
  37. Spawn replacement banks (collapsed if all banks exit; clone random survivor's equity)

Mirrors Mesa `BamModel._bankruptcy_entry()`.
"""
function _bankruptcy_entry!(model)
    _event33_update_net_worth!(model)
    exiting_firm_ids = _event34_mark_bankrupt_firms!(model)
    exiting_bank_ids = _event35_mark_bankrupt_banks!(model)
    model.collapsed && return nothing   # early collapse from event 34 or 35 (pre-entry)
    _event36_spawn_replacement_firms!(model, exiting_firm_ids)
    model.collapsed && return nothing   # collapse set by event 36
    _event37_spawn_replacement_banks!(model, exiting_bank_ids)
    return nothing
end

# ---------------------------------------------------------------------------
# Per-period step (all phases: planning + labor + credit + production +
# goods + revenue + bankruptcy/entry)
# ---------------------------------------------------------------------------

"""
    bam_step!(model)

Per-period model step. Runs all 7 phases (events 1-37):
  * Phase 1: planning (events 1-6)
  * Phase 2: labor market (events 7-12)
  * Phase 3: credit market (events 13-19)
  * Phase 4: production (events 20-24)
  * Phase 5: goods market (events 25-29)
  * Phase 6: revenue (events 30-32)
  * Phase 7-8: bankruptcy + entry (events 33-37)

Defined as a named function so the `StandardABM` has a real `model_step!`.
"""
function bam_step!(model)
    model.collapsed && return nothing
    model.period += 1
    _planning!(model)
    _labor_market!(model)
    _credit_market!(model)
    _production!(model)
    _goods_market!(model)
    _revenue!(model)
    _bankruptcy_entry!(model)
    return nothing
end

# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

"""
    build_model(n_firms, n_households, n_banks, params, seed; collect=false) -> StandardABM

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

When `collect=true`, per-period collection vectors are populated during `bam_step!`
at the same points the Mesa model collects them (Task 9).
"""
function build_model(n_firms::Integer, n_households::Integer, n_banks::Integer,
                     params::AbstractDict, seed::Integer; collect::Bool = false)
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
        collect,                        # collect flag
        Float64[],                      # c_unemployment
        Float64[],                      # c_avg_employed_wage
        Float64[],                      # c_total_production
        Float64[],                      # c_total_vacancies
        Float64[],                      # c_inflation
        Float64[],                      # c_production_final
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
