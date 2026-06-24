"""
markets.jl - market matching routines for the Agents.jl BAM model.

So far this holds the labor-market matcher (event 11). Later phase tasks add the
credit-market and goods-market matchers here.

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
    non-empty `job_apps` queue, in agent-iteration (creation) order. Each such
    worker pops the FRONT of its queue (its current top target).
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
worker scan order (agent/creation order) also matches Mesa's `self.households`
iteration. No other random draws occur in this event.
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
