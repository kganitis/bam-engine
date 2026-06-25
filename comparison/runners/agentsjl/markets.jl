"""
markets.jl - market matching routines for the Agents.jl BAM model.

This file holds the labor-market matcher (event 11), the credit-market matcher
(event 18), and the goods-market matcher (event 28, added in Task 7).

Each routine is a faithful translation of the corresponding Mesa function in
`comparison/runners/mesa/markets.py`. The matching ALGORITHM (worker application
order, per-round offer scan, conflict resolution = one shuffle per
firm-with-applicants, vacancy cap, contract length) and the RNG draw points/order
mirror the Mesa port exactly so the Task 10 equivalence gate holds. Per-seed
numeric values differ across languages (different PRNG); the structure does not.

All randomness flows through `abmrng(model)`.
"""

using Agents
using Random

# ---------------------------------------------------------------------------
# Labor market matching (event 11, repeated max_M rounds)
# ---------------------------------------------------------------------------

"""
    _run_labor_market!(model)

Event 11 (labor_market_round x max_M): multi-round labor-market matching.

Translated faithfully from Mesa `run_labor_market` (`markets.py:38`).

Each round:
  * Gather every still-unemployed worker (`employer_id == NO_AGENT`) with a
    non-empty `job_apps` queue. `households(model)` iterates in Dict (hash)
    order, not creation order. Each such worker pops the FRONT of its queue
    (its current top target).
  * Applications to firms with `n_vacancies == 0` are discarded (the head still
    advances - i.e. the target is popped), matching Mesa's `continue`.
  * The remaining applicants are grouped by target firm. The grouping preserves
    first-seen firm order and, within each group, worker arrival order
    (deterministic, like Mesa's insertion-ordered dict).
  * For each firm WITH applicants (in first-seen order), shuffle that firm's
    applicant list once (`shuffle!(rng, applicants)`) and hire the first
    `n_vacancies` of them. This single shuffle-per-firm IS the conflict
    resolution and matches Mesa's `model.random.shuffle(applicants)`.
  * On hire: `employer_id = firm.id`, `wage = firm.wage_offer` (posted, not
    negotiated), `periods_left = theta` (EXACT), `contract_expired = false`,
    `fired = false`, clear the worker's `job_apps`; push the worker id onto
    `firm.employee_ids`, `firm.current_labor += 1`, `firm.n_vacancies -= 1`.

RNG alignment with Mesa: exactly one `shuffle!` per firm-that-has-applicants per
round, in the SAME first-seen firm order Mesa iterates the dict it builds. The
worker scan uses `households(model)`, which iterates in Dict (hash) order (not
creation order); the grouping is still deterministic because the result depends
only on which workers are unemployed and what their `job_apps` queues contain,
not on the visit order itself. No other random draws occur in this event.
"""
function _run_labor_market!(model)
    rng = abmrng(model)
    max_M = Int(round(model.params["max_M"]))
    theta = Int(round(model.params["theta"]))

    for _ in 1:max_M
        # Group still-unemployed workers (with pending apps) by their popped
        # front target. Preserve first-seen firm order + arrival order to match
        # Mesa's insertion-ordered dict.
        firm_order = Int[]                       # firm ids in first-seen order
        applicants = Dict{Int,Vector{Int}}()     # firm id => worker ids (in arrival order)

        for h in households(model)
            hv = variant(h)
            (hv.employer_id != NO_AGENT || isempty(hv.job_apps)) && continue
            target = popfirst!(hv.job_apps)      # pop the front target
            tv = variant(model[target])          # target is a Firm
            tv.n_vacancies == 0 && continue       # discard app to a full firm (head already advanced)
            if !haskey(applicants, target)
                applicants[target] = Int[]
                push!(firm_order, target)
            end
            push!(applicants[target], h.id)
        end

        # Process firms in first-seen order; one shuffle per firm = conflict
        # resolution; accept first n_vacancies applicants.
        for fid in firm_order
            f = model[fid]
            fv = variant(f)
            fv.n_vacancies == 0 && continue
            group = applicants[fid]
            shuffle!(rng, group)                  # random rationing (one shuffle per firm)
            k = min(fv.n_vacancies, length(group))
            for i in 1:k
                hid = group[i]
                hv = variant(model[hid])
                # Hire.
                hv.employer_id      = fid
                hv.wage             = fv.wage_offer
                hv.periods_left     = theta
                hv.contract_expired = false
                hv.fired            = false
                empty!(hv.job_apps)
                push!(fv.employee_ids, hid)
                fv.current_labor += 1
                fv.n_vacancies   -= 1
            end
        end
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Credit market matching (event 18, repeated max_H rounds)
# ---------------------------------------------------------------------------

"""
    _run_credit_market!(model)

Event 18 (credit_market_round x max_H): multi-round firm-bank matching.

Translated faithfully from Mesa `run_credit_market` (`markets.py:79`).

Each round:
  * Gather every firm with `credit_demand > 0` and a non-empty `loan_apps`
    queue. `allagents(model)` iterates in Dict (hash) order, not creation order.
    Each such firm pops the FRONT of its queue (its current top bank) and is
    grouped under that bank. Grouping preserves first-seen bank order and, within
    each group, firm arrival order (deterministic, like Mesa's insertion-ordered
    dict).
  * For each bank WITH applicants (in first-seen order): skip if its
    `credit_supply <= EPS`. Otherwise rank that bank's applicants by
    `projected_fragility` ASC (safer firms first, NO random tie-break - a STABLE
    sort preserves arrival order on ties, matching Mesa's stable `list.sort`).
  * Walk the ranked applicants maintaining a running `granted_total` against the
    bank's `credit_supply`. For each firm:
      - per-loan cap `max_grant = min(credit_demand, net_worth * max_loan_to_net_worth)`
        (or just `credit_demand` if `max_loan_to_net_worth <= 0`); skip if `<= 0`;
      - `amount = max_grant` if it fits, else the remaining supply (PARTIAL loan
        at the boundary, clamped at 0);
      - if `amount > EPS`: the CONTRACT rate (Flag 5) is
        `r_bar * (1 + opex_shock * min(projected_fragility, max_leverage))` -
        fragility-scaled, distinct from the bank's posted rate. Push a `Loan`
        (borrower_id, lender_id, principal=amount, rate) onto the shared loan
        book `model.loans`; `total_funds += amount`; `credit_demand -= amount`;
        `granted_total += amount`.
  * `break` out of the applicant loop once `granted_total >= supply`.
  * After all of a bank's applicants are processed, decrement
    `credit_supply -= granted_total` ONCE.
  * Loans ACCUMULATE across rounds within the period (the book is purged only at
    the start of the credit market, event 17 open).

RNG alignment with Mesa: NO random draws occur in this event in either
implementation. The applicant ranking is deterministic (`projected_fragility`
ASC, stable). We iterate `1:model.n_firms` (id-ascending), which matches Mesa's
insertion/id order; firm iteration order here only affects the `bank_order`
first-seen sequence and the stable-sort tie-break on equal `projected_fragility`,
and the gate passes.
"""
function _run_credit_market!(model)
    eps = model.eps
    max_H = Int(round(model.params["max_H"]))
    max_loan_to_net_worth = model.params["max_loan_to_net_worth"]
    r_bar = model.params["r_bar"]
    max_leverage = model.params["max_leverage"]

    for _ in 1:max_H
        # Group firms (with demand + pending apps) by their popped front bank.
        # Preserve first-seen bank order + arrival order to match Mesa's
        # insertion-ordered dict.
        bank_order = Int[]                       # bank ids in first-seen order
        applicants = Dict{Int,Vector{Int}}()     # bank id => firm ids (arrival order)

        for fid in 1:model.n_firms
            a  = model[fid]
            fv = variant(a)
            (fv.credit_demand <= 0.0 || isempty(fv.loan_apps)) && continue
            bank = popfirst!(fv.loan_apps)       # pop the front bank
            if !haskey(applicants, bank)
                applicants[bank] = Int[]
                push!(bank_order, bank)
            end
            push!(applicants[bank], fid)
        end

        # Process banks in first-seen order.
        for bid in bank_order
            bv = variant(model[bid])
            supply = bv.credit_supply
            supply <= eps && continue

            group = applicants[bid]
            # Rank applicants by projected_fragility ASC (safer first), NO random
            # tie-break. STABLE sort preserves arrival order on ties.
            frag_of(fid) = variant(model[fid]).projected_fragility
            sort!(group; by = frag_of, alg = MergeSort)

            granted_total = 0.0
            for fid in group
                granted_total >= supply && break
                fv = variant(model[fid])
                # Per-loan cap.
                max_grant = if max_loan_to_net_worth > 0.0
                    min(fv.credit_demand, fv.net_worth * max_loan_to_net_worth)
                else
                    fv.credit_demand
                end
                max_grant <= 0.0 && continue
                remaining = supply - granted_total
                amount = (granted_total + max_grant <= supply) ?
                         max_grant : max(remaining, 0.0)
                if amount > eps
                    fragility = min(fv.projected_fragility, max_leverage)
                    rate = r_bar * (1.0 + bv.opex_shock * fragility)
                    push!(model.loans,
                          Loan(amount, rate, fid, bid))   # principal, rate, borrower, lender
                    fv.total_funds   += amount
                    fv.credit_demand -= amount
                    granted_total    += amount
                end
            end
            # Update bank supply once after all applicants processed.
            bv.credit_supply -= granted_total
        end
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Goods market matching (event 28, sequential shopping loop)
# ---------------------------------------------------------------------------

"""
    _run_goods_market!(model)

Event 28 (goods_market_round): sequential goods-market matching.

Translated faithfully from Mesa `run_goods_market` (`markets.py`).

Algorithm:
  * Build a list of households whose `income_to_spend > eps` (active buyers).
  * Shuffle the buyer list ONCE with `abmrng(model)` (Mesa's
    `model.random.shuffle(buyers)`) so every buyer gets a random service order.
  * For each buyer walk its `shop_visits` (a `Vector{Int}` of firm ids sorted by
    price ASC, built in event 27). For each firm in the visit list:
      - break if buyer's budget (`income_to_spend`) <= eps (budget exhausted)
      - continue if firm's inventory <= eps (nothing left to sell)
      - `qty = min(income_to_spend / price, inventory)`
      - `spent = qty * price`
      - decrement buyer's budget and firm's inventory IMMEDIATELY (in-loop)

This single-pass loop matches Mesa's sequential processing. Inventory and budget
updates are eager: a firm that sells out mid-loop appears exhausted to subsequent
buyers in the same pass. No aggregation or reconciliation step is needed.

RNG: exactly one `shuffle!(abmrng(model), buyers)` call, matching Mesa's single
`model.random.shuffle(buyers)`. No other draws occur in this event.

Note: `shop_visits` stores firm ids (integers), not Firm objects. Look up each
firm variant by `variant(model[firm_id])`.
"""
function _run_goods_market!(model)
    eps = model.eps
    rng = abmrng(model)

    # Collect active buyers in Dict (hash) order (allagents iteration order), then shuffle.
    buyers = [h for h in households(model) if variant(h).income_to_spend > eps]
    isempty(buyers) && return nothing

    shuffle!(rng, buyers)   # random service order, one call matching Mesa

    for buyer in buyers
        hv = variant(buyer)
        for firm_id in hv.shop_visits
            hv.income_to_spend <= eps && break          # budget exhausted
            fv = variant(model[firm_id])
            fv.inventory <= eps && continue             # firm is sold out
            qty   = min(hv.income_to_spend / fv.price, fv.inventory)
            spent = qty * fv.price
            hv.income_to_spend -= spent
            fv.inventory       -= qty
        end
    end

    return nothing
end
