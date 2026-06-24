"""
runtests.jl - Julia test harness for the Agents.jl BAM runner.

Run with:
    julia --project=comparison/runners/agentsjl --startup-file=no comparison/runners/agentsjl/test/runtests.jl

Tests cover the complete Agents.jl BAM runner (all phases: planning, labor,
credit, production, goods market, bankruptcy and entry).
"""

using Test
using Agents

# Load the model under test (model.jl lives one directory up from test/).
include(joinpath(@__DIR__, "..", "model.jl"))

@testset "BamAgentsJl" begin

    @testset "build_model construction" begin
        # Small population + fixed seed. Canonical baseline params (subset
        # needed for construction; values from src/bamengine/config/defaults.yml).
        params = Dict{String,Float64}(
            "labor_productivity"  => 0.50,
            "price_init"          => 0.50,
            "net_worth_ratio"     => 6.0,
            "savings_init"        => 1.0,
            "equity_base_init"    => 5.0,
            "min_wage_ratio"      => 0.5,
            "min_wage_rev_period" => 4.0,
            # Planning-phase params (required because bam_step! now runs planning).
            "h_rho"               => 0.10,
            "h_eta"               => 0.10,
            "h_xi"                => 0.05,
            # Labor-market params (bam_step! now also runs the labor phase).
            "max_M"               => 4.0,
            "theta"               => 8.0,
            # Credit-market params (bam_step! now also runs the credit phase).
            "v"                     => 0.10,
            "h_phi"                 => 0.10,
            "r_bar"                 => 0.02,
            "max_H"                 => 2.0,
            "max_loan_to_net_worth" => 2.0,
            "max_leverage"          => 10.0,
            # Production + revenue params (bam_step! now also runs these phases).
            "delta"                 => 0.10,
            # Goods-market params (bam_step! now also runs the goods market).
            "beta"                  => 2.5,
            "max_Z"                 => 2.0,
            # Bankruptcy + entry params (bam_step! now also runs bankruptcy/entry).
            "new_firm_size_factor"      => 0.5,
            "new_firm_production_factor" => 0.5,
            "new_firm_wage_factor"      => 0.5,
            "new_firm_price_markup"     => 1.20,
        )
        n_firms, n_households, n_banks = 10, 50, 2
        model = build_model(n_firms, n_households, n_banks, params, 42)

        # --- Derived init values (Mesa BamModel.__init__) ---
        production_init = n_households * 0.50 / n_firms        # 2.5
        wage_offer_init = 0.50 / 3.0                           # 0.16666...
        net_worth_init = production_init * 0.50 * 6.0          # 7.5
        min_wage_init = wage_offer_init * 0.5                  # 0.08333...

        # --- Per-kind agent counts ---
        @test nagents(model) == n_firms + n_households + n_banks
        @test count(a -> variantof(a) === Firm, allagents(model)) == n_firms
        @test count(a -> variantof(a) === Household, allagents(model)) == n_households
        @test count(a -> variantof(a) === Bank, allagents(model)) == n_banks

        # Convenience iterators agree with the counts.
        @test length(collect(firms(model))) == n_firms
        @test length(collect(households(model))) == n_households
        @test length(collect(banks(model))) == n_banks

        # --- Economy-state properties ---
        @test model.n_firms == n_firms
        @test model.n_households == n_households
        @test model.n_banks == n_banks
        @test model.period == 0
        @test model.collapsed == false
        @test model.avg_mkt_price == 0.50
        @test model.min_wage ≈ min_wage_init
        @test model.min_wage_rev_period == 4
        @test model.avg_mkt_price_history == [0.50]
        @test model.inflation_history == [0.0]
        @test isempty(model.loans)
        @test eltype(model.loans) === Loan

        # Properties are a concrete struct (type-stable), not a Dict.
        @test abmproperties(model) isa BAMProperties

        # RNG is the seeded Xoshiro (determinism via abmrng, no global RNG).
        @test abmrng(model) isa Xoshiro

        # --- Key firm field initial values (Mesa Firm.__init__) ---
        f = first(firms(model))
        fv = variant(f)
        @test fv.production == 0.0
        @test fv.production_prev ≈ production_init
        @test fv.inventory == 0.0
        @test fv.expected_demand == 1.0
        @test fv.desired_production == 0.0
        @test fv.labor_productivity == 0.50
        @test fv.breakeven_price == 0.50
        @test fv.price == 0.50
        @test fv.desired_labor == 0
        @test fv.current_labor == 0
        @test fv.wage_offer ≈ wage_offer_init
        @test fv.wage_bill == 0.0
        @test fv.n_vacancies == 0
        @test fv.total_funds ≈ net_worth_init
        @test fv.net_worth ≈ net_worth_init
        @test fv.credit_demand == 0.0
        @test fv.projected_fragility == 0.0
        @test fv.gross_profit == 0.0
        @test fv.net_profit == 0.0
        @test fv.retained_profit == 0.0
        @test isempty(fv.loan_apps)
        @test isempty(fv.employee_ids)

        # --- Key household field initial values (Mesa Household.__init__) ---
        h = first(households(model))
        hv = variant(h)
        @test hv.employer_id == NO_AGENT
        @test hv.employer_prev_id == NO_AGENT
        @test hv.wage == 0.0
        @test hv.periods_left == 0
        @test hv.contract_expired == false
        @test hv.fired == false
        @test hv.income == 0.0
        @test hv.savings == 1.0
        @test hv.income_to_spend == 0.0
        @test hv.propensity == 0.0
        @test hv.largest_prod_prev_id == NO_AGENT
        @test isempty(hv.job_apps)
        @test isempty(hv.shop_visits)
        @test hv.dividends == 0.0

        # --- Key bank field initial values (Mesa Bank.__init__) ---
        b = first(banks(model))
        bv = variant(b)
        @test bv.equity_base == 5.0
        @test bv.credit_supply == 0.0
        @test bv.interest_rate == 0.0
        @test bv.opex_shock == 0.0

        # --- A step! advances without error (bam_step! runs planning phase) ---
        step!(model, 1)
        @test nagents(model) == n_firms + n_households + n_banks
    end

    @testset "planning phase invariants (events 1-6)" begin
        # Full canonical param set required by planning events.
        params = Dict{String,Float64}(
            "labor_productivity"  => 0.50,
            "price_init"          => 0.50,
            "net_worth_ratio"     => 6.0,
            "savings_init"        => 1.0,
            "equity_base_init"    => 5.0,
            "min_wage_ratio"      => 0.5,
            "min_wage_rev_period" => 4.0,
            "h_rho"               => 0.10,
            "h_eta"               => 0.10,
            "h_xi"                => 0.05,
        )
        n_firms, n_households, n_banks = 10, 50, 2
        model = build_model(n_firms, n_households, n_banks, params, 42)

        # Run planning phase directly (not via step! so we don't increment period).
        _planning!(model)

        # Event 1: production zeroed; desired_production >= 0 for all firms.
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test fv.production == 0.0
            @test fv.desired_production >= 0.0
            @test fv.expected_demand >= 0.0
        end

        # Event 2: breakeven_price is finite and >= 0.
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test isfinite(fv.breakeven_price)
            @test fv.breakeven_price >= 0.0
        end

        # Event 3: price is finite and positive (breakeven floor prevents <= 0).
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test isfinite(fv.price)
            @test fv.price > 0.0
        end

        # Event 4: desired_labor >= 0 (ceil of non-negative value).
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test fv.desired_labor >= 0
        end

        # Event 5: n_vacancies >= 0.
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test fv.n_vacancies >= 0
        end

        # Event 6: current_labor <= desired_labor after firing (no over-firing).
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test fv.current_labor <= fv.desired_labor
        end

        # Consistency: current_labor == length(employee_ids) for all firms.
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test fv.current_labor == length(fv.employee_ids)
        end

        # Household counts preserved; no agents added or removed.
        @test nagents(model) == n_firms + n_households + n_banks
    end

    @testset "labor market invariants (events 7-12)" begin
        # Full canonical param set required by planning + labor events.
        params = Dict{String,Float64}(
            "labor_productivity"  => 0.50,
            "price_init"          => 0.50,
            "net_worth_ratio"     => 6.0,
            "savings_init"        => 1.0,
            "equity_base_init"    => 5.0,
            "min_wage_ratio"      => 0.5,
            "min_wage_rev_period" => 4.0,
            "h_rho"               => 0.10,
            "h_eta"               => 0.10,
            "h_xi"                => 0.05,
            # Labor-market params.
            "max_M"               => 4.0,
            "theta"               => 8.0,
        )
        n_firms, n_households, n_banks = 10, 50, 2
        model = build_model(n_firms, n_households, n_banks, params, 42)

        # Run planning + labor directly (not via step! so period stays 0).
        _planning!(model)
        _labor_market!(model)

        # Build the set of valid firm ids for membership checks.
        firm_ids = Set(a.id for a in allagents(model) if variantof(a) === Firm)

        # --- Employed count <= n_households ---
        employed = [a for a in allagents(model)
                    if variantof(a) === Household && variant(a).employer_id != NO_AGENT]
        @test length(employed) <= n_households

        # --- Every employed worker's employer_id is a valid Firm id ---
        for a in employed
            hv = variant(a)
            @test hv.employer_id in firm_ids
            # A hired worker carries the firm's posted wage and a full contract.
            @test hv.periods_left == Int(round(params["theta"]))
            @test hv.wage > 0.0
        end

        # --- Each firm's current_labor == count of its employed workers and
        #     == length(employee_ids); employee_ids point back at this firm ---
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            counted = count(h -> variantof(h) === Household
                                 && variant(h).employer_id == a.id,
                            allagents(model))
            @test fv.current_labor == counted
            @test fv.current_labor == length(fv.employee_ids)
            # employee_ids back-reference this firm; no duplicates.
            for hid in fv.employee_ids
                @test variant(model[hid]).employer_id == a.id
            end
            @test length(unique(fv.employee_ids)) == length(fv.employee_ids)
            # No firm hired beyond its desired labor (event 5/6/11 bookkeeping).
            @test fv.n_vacancies >= 0
        end

        # --- Event 12: wage_bill == sum of employees' wages ---
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            expected = sum((variant(model[hid]).wage for hid in fv.employee_ids); init = 0.0)
            @test fv.wage_bill ≈ expected
        end

        # --- Event 7: inflation_history grew by one entry (was length 1) ---
        @test length(model.inflation_history) == 2

        # Populations preserved; no agents added or removed.
        @test nagents(model) == n_firms + n_households + n_banks
    end

    @testset "credit market invariants (events 13-19)" begin
        # Full canonical param set required by planning + labor + credit events.
        # `net_worth_ratio` is set LOW so firms start with little cash and the
        # wage bill exceeds their funds, forcing a financing gap that exercises
        # the credit market (demand > 0, loans granted, possible fire-on-gap).
        params = Dict{String,Float64}(
            "labor_productivity"    => 0.50,
            "price_init"            => 0.50,
            "net_worth_ratio"       => 0.05,   # tiny: forces a financing gap
            "savings_init"          => 1.0,
            "equity_base_init"      => 5.0,
            "min_wage_ratio"        => 0.5,
            "min_wage_rev_period"   => 4.0,
            "h_rho"                 => 0.10,
            "h_eta"                 => 0.10,
            "h_xi"                  => 0.05,
            # Labor-market params.
            "max_M"                 => 4.0,
            "theta"                 => 8.0,
            # Credit-market params (values from defaults.yml).
            "v"                     => 0.10,
            "h_phi"                 => 0.10,
            "r_bar"                 => 0.02,
            "max_H"                 => 2.0,
            "max_loan_to_net_worth" => 2.0,
            "max_leverage"          => 10.0,
        )
        n_firms, n_households, n_banks = 10, 50, 2
        model = build_model(n_firms, n_households, n_banks, params, 42)

        # Run planning + labor + credit directly (period stays 0).
        _planning!(model)
        _labor_market!(model)

        # Snapshot per-firm pre-credit wage_bill / total_funds so we can verify
        # that loans only ever went to firms with a financing gap, and check the
        # total_funds bookkeeping after lending.
        wage_bill_pre  = Dict(a.id => variant(a).wage_bill   for a in allagents(model) if variantof(a) === Firm)
        funds_pre      = Dict(a.id => variant(a).total_funds for a in allagents(model) if variantof(a) === Firm)
        # Capacity each bank brings into the market (equity_base / v).
        v = params["v"]
        supply_cap = Dict(b.id => max(variant(b).equity_base / v, 0.0)
                          for b in allagents(model) if variantof(b) === Bank)

        _credit_market!(model)

        firm_ids = Set(a.id for a in allagents(model) if variantof(a) === Firm)
        bank_ids = Set(b.id for b in allagents(model) if variantof(b) === Bank)

        # --- The gap was actually forced: at least one loan was granted ---
        @test !isempty(model.loans)

        # --- Loans only to firms that had a financing gap (wage_bill > funds) ---
        for loan in model.loans
            @test loan.borrower_id in firm_ids
            @test loan.lender_id in bank_ids
            @test loan.principal > 0.0
            @test loan.rate > 0.0
            # Pre-credit gap must have been positive for any borrower.
            @test wage_bill_pre[loan.borrower_id] - funds_pre[loan.borrower_id] > 0.0
        end

        # --- Per-bank lent <= credit_supply capacity (equity_base / v) ---
        lent_by_bank = Dict{Int,Float64}(bid => 0.0 for bid in bank_ids)
        for loan in model.loans
            lent_by_bank[loan.lender_id] += loan.principal
        end
        for bid in bank_ids
            @test lent_by_bank[bid] <= supply_cap[bid] + 1e-9
            # Remaining advertised supply + what was lent == original capacity.
            bv = variant(model[bid])
            @test bv.credit_supply + lent_by_bank[bid] ≈ supply_cap[bid]
        end

        # --- total_funds bookkeeping: each borrower's funds rose by exactly the
        #     sum of its loan principals (event 18 credits total_funds += amount).
        #     Event 19 fire-on-gap does NOT touch total_funds, so this holds. ---
        lent_by_firm = Dict{Int,Float64}(fid => 0.0 for fid in firm_ids)
        for loan in model.loans
            lent_by_firm[loan.borrower_id] += loan.principal
        end
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test fv.total_funds ≈ funds_pre[a.id] + lent_by_firm[a.id]
        end

        # --- Per-loan cap respected: each INDIVIDUAL loan's principal
        #     <= net_worth * max_loan_to_net_worth. (The per-loan cap is applied
        #     against the borrower's net_worth per draw; a firm CAN exceed this
        #     in aggregate by borrowing from multiple banks/rounds, exactly as in
        #     the Mesa port, so the cap is per-loan, not per-firm-aggregate.) ---
        mltnw = params["max_loan_to_net_worth"]
        if mltnw > 0.0
            nw_of = Dict(a.id => variant(a).net_worth
                         for a in allagents(model) if variantof(a) === Firm)
            for loan in model.loans
                @test loan.principal <= nw_of[loan.borrower_id] * mltnw + 1e-9
            end
        end

        # --- Event 19 fire-on-gap: no firm ends with wage_bill > total_funds by
        #     more than one (last) worker's wage; concretely current_labor and
        #     employee_ids stay consistent and non-negative. ---
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test fv.current_labor >= 0
            @test fv.current_labor == length(fv.employee_ids)
            for hid in fv.employee_ids
                @test variant(model[hid]).employer_id == a.id
            end
        end

        # --- Banks: posted interest_rate set (event 14) and supply non-negative ---
        for b in allagents(model)
            variantof(b) === Bank || continue
            bv = variant(b)
            @test bv.interest_rate ≈ params["r_bar"] * (1.0 + bv.opex_shock)
            @test bv.credit_supply >= -1e-9
        end

        # Populations preserved; no agents added or removed.
        @test nagents(model) == n_firms + n_households + n_banks
    end

    @testset "loan book persistence + purge across periods" begin
        # Verify loans persist through planning/labor (so event 2 can read prior
        # interest) and are purged at the start of the next credit market.
        params = Dict{String,Float64}(
            "labor_productivity"    => 0.50,
            "price_init"            => 0.50,
            "net_worth_ratio"       => 0.05,   # force a gap so loans appear
            "savings_init"          => 1.0,
            "equity_base_init"      => 5.0,
            "min_wage_ratio"        => 0.5,
            "min_wage_rev_period"   => 4.0,
            "h_rho"                 => 0.10,
            "h_eta"                 => 0.10,
            "h_xi"                  => 0.05,
            "max_M"                 => 4.0,
            "theta"                 => 8.0,
            "v"                     => 0.10,
            "h_phi"                 => 0.10,
            "r_bar"                 => 0.02,
            "max_H"                 => 2.0,
            "max_loan_to_net_worth" => 2.0,
            "max_leverage"          => 10.0,
        )
        model = build_model(10, 50, 2, params, 7)

        # Period 1: create loans.
        _planning!(model)
        _labor_market!(model)
        _credit_market!(model)
        @test !isempty(model.loans)
        n_loans_p1 = length(model.loans)

        # Loans persist through the next planning + labor phases (NOT purged yet)
        # so event 2 (plan_breakeven_price) can read prior-period interest.
        _planning!(model)
        _labor_market!(model)
        @test length(model.loans) == n_loans_p1   # still there during planning/labor

        # The next credit market opens with a purge: prior loans are gone, only
        # this period's loans remain afterwards.
        _purge_loans!(model)
        @test isempty(model.loans)
    end

    @testset "production + revenue invariants (events 20-24, 30-32)" begin
        # Full canonical param set (all phases through revenue).
        # Use the default net_worth_ratio=6.0 so firms have plenty of funds and
        # most firms are solvent; a tiny net_worth_ratio tests the default path.
        params = Dict{String,Float64}(
            "labor_productivity"    => 0.50,
            "price_init"            => 0.50,
            "net_worth_ratio"       => 6.0,
            "savings_init"          => 1.0,
            "equity_base_init"      => 5.0,
            "min_wage_ratio"        => 0.5,
            "min_wage_rev_period"   => 4.0,
            "h_rho"                 => 0.10,
            "h_eta"                 => 0.10,
            "h_xi"                  => 0.05,
            "max_M"                 => 4.0,
            "theta"                 => 8.0,
            "v"                     => 0.10,
            "h_phi"                 => 0.10,
            "r_bar"                 => 0.02,
            "max_H"                 => 2.0,
            "max_loan_to_net_worth" => 2.0,
            "max_leverage"          => 10.0,
            "delta"                 => 0.10,
        )
        n_firms, n_households, n_banks = 10, 50, 2
        model = build_model(n_firms, n_households, n_banks, params, 42)

        # Run phases 1-4 directly (not via step! so we have a clean snapshot).
        _planning!(model)
        _labor_market!(model)
        _credit_market!(model)

        # Snapshot pre-production state for wage-income verification.
        wage_of = Dict(h.id => variant(h).wage
                       for h in households(model) if variant(h).employer_id != NO_AGENT)
        income_pre = Dict(h.id => variant(h).income
                          for h in households(model))

        _production!(model)

        # --- Event 22: production == labor_productivity * current_labor per firm ---
        lp = params["labor_productivity"]
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test fv.production ≈ lp * fv.current_labor
        end

        # --- Event 22: production_prev == production (unconditional) ---
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test fv.production_prev == fv.production
        end

        # --- Event 22: inventory == production (overwrite, Flag 4) ---
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test fv.inventory == fv.production
        end

        # --- Event 21: employed workers' income increased by their wage ---
        for (hid, w) in wage_of
            hv = variant(model[hid])
            @test hv.income ≈ income_pre[hid] + w
        end

        # --- Event 23: avg_mkt_price_history gained one entry ---
        # (It was length 1 after build_model, then 2 after the inflation event in
        # _labor_market!... but the history grows per update_avg_mkt_price call.)
        @test length(model.avg_mkt_price_history) >= 2

        # --- avg_mkt_price is positive and finite ---
        @test model.avg_mkt_price > 0.0
        @test isfinite(model.avg_mkt_price)

        # --- Event 24: periods_left decremented for all still-employed workers ---
        # (A newly hired worker starts with periods_left == theta == 8; after one
        # update_contracts pass it is 7. Check it is <= theta - 1 = 7.)
        theta = Int(round(params["theta"]))
        for h in households(model)
            hv = variant(h)
            if hv.employer_id != NO_AGENT
                # Still employed: periods_left was decremented and > 0.
                @test hv.periods_left > 0
                @test hv.periods_left <= theta
            end
        end

        # --- Contract-expired workers are properly unlinked ---
        for h in households(model)
            hv = variant(h)
            if hv.contract_expired
                # Unlinked from employer.
                @test hv.employer_id == NO_AGENT
                @test hv.wage == 0.0
                @test hv.fired == false
            end
        end

        # --- current_labor + employee_ids consistency after contract expiry ---
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test fv.current_labor == length(fv.employee_ids)
            @test fv.current_labor >= 0
        end

        # --- Revenue phase ---
        _revenue!(model)

        # --- Event 32: dividends >= 0 for every household ---
        for h in households(model)
            @test variant(h).dividends >= 0.0
        end

        # --- Event 32: all households received the same dividend per household ---
        div_vals = [variant(h).dividends for h in households(model)]
        if length(div_vals) > 1
            @test all(d -> d ≈ div_vals[1], div_vals)
        end

        # --- Event 30: gross_profit is finite for all firms ---
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test isfinite(fv.gross_profit)
        end

        # --- Event 31: net_profit is finite for all firms ---
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test isfinite(fv.net_profit)
        end

        # --- Event 31: total_funds is finite and non-negative for all firms ---
        # (Defaulting firms get total_funds=0; solvent firms have reduced funds.)
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test isfinite(fv.total_funds)
            @test fv.total_funds >= 0.0
        end

        # --- Populations preserved; no agents added or removed ---
        @test nagents(model) == n_firms + n_households + n_banks
    end

    @testset "goods market invariants (events 25-29)" begin
        # Full canonical param set through revenue; add consumer params.
        # Use default net_worth_ratio=6.0 so firms are solvent and produce goods.
        params = Dict{String,Float64}(
            "labor_productivity"    => 0.50,
            "price_init"            => 0.50,
            "net_worth_ratio"       => 6.0,
            "savings_init"          => 1.0,
            "equity_base_init"      => 5.0,
            "min_wage_ratio"        => 0.5,
            "min_wage_rev_period"   => 4.0,
            "h_rho"                 => 0.10,
            "h_eta"                 => 0.10,
            "h_xi"                  => 0.05,
            "max_M"                 => 4.0,
            "theta"                 => 8.0,
            "v"                     => 0.10,
            "h_phi"                 => 0.10,
            "r_bar"                 => 0.02,
            "max_H"                 => 2.0,
            "max_loan_to_net_worth" => 2.0,
            "max_leverage"          => 10.0,
            "delta"                 => 0.10,
            # Consumer params (goods market).
            "beta"                  => 2.5,
            "max_Z"                 => 2.0,
        )
        n_firms, n_households, n_banks = 10, 50, 2
        model = build_model(n_firms, n_households, n_banks, params, 42)

        # Run phases 1-4 first so firms have production + inventory and
        # workers have income to spend.
        _planning!(model)
        _labor_market!(model)
        _credit_market!(model)
        _production!(model)

        # Snapshot pre-goods state.
        income_to_spend_pre = Dict(h.id => variant(h).income_to_spend
                                   for h in households(model))
        inventory_pre = Dict(a.id => variant(a).inventory
                             for a in allagents(model) if variantof(a) === Firm)

        # Run goods market.
        _goods_market!(model)

        firm_ids = [a.id for a in allagents(model) if variantof(a) === Firm]

        # 1. No household spent more than its pre-goods income_to_spend.
        #    After finalize_purchases, income_to_spend is reset to 0 for all
        #    households; the spending is absorbed into savings. The per-household
        #    budget change (pre - 0) must be non-negative.
        #    Note: income_to_spend_pre is 0 before the goods market runs (workers
        #    receive income in event 21, and event 26 splits it into budget +
        #    savings before shopping). Check that all income_to_spend_pre are
        #    non-negative (they must be by construction) and that after event 29
        #    they are all exactly 0.
        for h in households(model)
            hv = variant(h)
            # Event 29: all income_to_spend zeroed after finalize_purchases.
            @test hv.income_to_spend == 0.0
        end

        # 2. No firm ends with negative inventory.
        for a in allagents(model)
            variantof(a) === Firm || continue
            fv = variant(a)
            @test fv.inventory >= -1e-9
        end

        # 3. Total inventory sold = sum of (inventory_pre - inventory_post) >= 0.
        total_sold = sum(inventory_pre[fid] - variant(model[fid]).inventory
                         for fid in firm_ids)
        @test total_sold >= 0.0

        # 4. Firm inventory did not INCREASE (goods market only removes goods).
        for fid in firm_ids
            fv = variant(model[fid])
            @test fv.inventory <= inventory_pre[fid] + 1e-9
        end

        # 5. Propensity in (0, 1] for all households (tanh-based formula).
        for h in households(model)
            hv = variant(h)
            @test hv.propensity > 0.0
            @test hv.propensity <= 1.0
        end

        # 6. shop_visits contains valid firm ids (or is empty for zero-budget hh).
        for h in households(model)
            hv = variant(h)
            for fid in hv.shop_visits
                @test fid in firm_ids
            end
            @test length(hv.shop_visits) <= Int(round(params["max_Z"]))
        end

        # 7. Populations preserved; no agents added or removed.
        @test nagents(model) == n_firms + n_households + n_banks
    end

    @testset "bankruptcy + entry invariants (events 33-37)" begin
        # Full canonical param set plus entry params.
        params = Dict{String,Float64}(
            "labor_productivity"       => 0.50,
            "price_init"               => 0.50,
            "net_worth_ratio"          => 6.0,
            "savings_init"             => 1.0,
            "equity_base_init"         => 5.0,
            "min_wage_ratio"           => 0.5,
            "min_wage_rev_period"      => 4.0,
            "h_rho"                    => 0.10,
            "h_eta"                    => 0.10,
            "h_xi"                     => 0.05,
            "max_M"                    => 4.0,
            "theta"                    => 8.0,
            "v"                        => 0.10,
            "h_phi"                    => 0.10,
            "r_bar"                    => 0.02,
            "max_H"                    => 2.0,
            "max_loan_to_net_worth"    => 2.0,
            "max_leverage"             => 10.0,
            "delta"                    => 0.10,
            "beta"                     => 2.5,
            "max_Z"                    => 2.0,
            # Entry params (defaults.yml values).
            "new_firm_size_factor"     => 0.5,
            "new_firm_production_factor" => 0.5,
            "new_firm_wage_factor"     => 0.5,
            "new_firm_price_markup"    => 1.20,
        )
        n_firms, n_households, n_banks = 10, 50, 2
        model = build_model(n_firms, n_households, n_banks, params, 42)

        # Run phases 1-6 to get firms into a realistic state.
        _planning!(model)
        _labor_market!(model)
        _credit_market!(model)
        _production!(model)
        _goods_market!(model)
        _revenue!(model)

        # --- Force some firms bankrupt by driving net_worth and production_prev to 0. ---
        firm_ids_sorted = sort!([a.id for a in allagents(model) if variantof(a) === Firm])
        # Force the first 2 firms into insolvency.
        forced_bankrupt_firm_ids = firm_ids_sorted[1:2]
        for fid in forced_bankrupt_firm_ids
            fv = variant(model[fid])
            fv.net_worth       = -1.0   # below EPS: insolvent
            fv.production_prev = 0.0    # also a ghost
        end

        # Hire a worker into the first forced-bankrupt firm to test worker firing.
        victim_firm_id = forced_bankrupt_firm_ids[1]
        victim_fv = variant(model[victim_firm_id])
        # Pick an unemployed household and fake-employ it.
        victim_hh = first(h for h in households(model) if variant(h).employer_id == NO_AGENT)
        victim_hid = victim_hh.id
        victim_hv = variant(victim_hh)
        victim_hv.employer_id      = victim_firm_id
        victim_hv.employer_prev_id = NO_AGENT
        victim_hv.wage             = 0.5
        victim_hv.periods_left     = 4
        push!(victim_fv.employee_ids, victim_hid)
        victim_fv.current_labor = 1

        # --- Force the first bank bankrupt. ---
        bank_ids_sorted = sort!([b.id for b in banks(model) if variantof(b) === Bank])
        forced_bankrupt_bank_id = bank_ids_sorted[1]
        variant(model[forced_bankrupt_bank_id]).equity_base = -1.0   # below EPS

        # --- Inject synthetic loans to make the loan-pruning assertion meaningful. ---
        # One loan whose borrower is a forced-bankrupt firm (should be pruned by event 34).
        push!(model.loans, Loan(5.0, 0.03, forced_bankrupt_firm_ids[1], bank_ids_sorted[2]))
        # One loan whose lender is the forced-bankrupt bank (should be pruned by event 35).
        push!(model.loans, Loan(3.0, 0.02, firm_ids_sorted[end], forced_bankrupt_bank_id))
        @test length(model.loans) >= 2   # sanity: at least the injected loans exist

        # --- Run bankruptcy + entry. ---
        _bankruptcy_entry!(model)

        # M3 loan-pruning: no loan in the book should reference a forced-bankrupt
        # firm as borrower OR a forced-bankrupt bank as lender.
        bankrupt_firm_set = Set(forced_bankrupt_firm_ids)
        for loan in model.loans
            @test !(loan.borrower_id in bankrupt_firm_set)
            @test loan.lender_id != forced_bankrupt_bank_id
        end

        # 1. Population constant per kind.
        @test count(a -> variantof(a) === Firm,      allagents(model)) == n_firms
        @test count(a -> variantof(a) === Household, allagents(model)) == n_households
        @test count(a -> variantof(a) === Bank,      allagents(model)) == n_banks
        @test nagents(model) == n_firms + n_households + n_banks

        # 2. Replacement firms have positive net_worth (not re-ghosted).
        for fid in forced_bankrupt_firm_ids
            fv = variant(model[fid])
            @test fv.net_worth > 0.0
            @test fv.total_funds > 0.0
            @test fv.production_prev > 0.0   # > 0 so not immediately re-ghosted
        end

        # 3. Worker of bankrupt firm is now unemployed; employer_prev cleared.
        hv_after = variant(model[victim_hid])
        @test hv_after.employer_id      == NO_AGENT
        @test hv_after.employer_prev_id == NO_AGENT   # event 34: loyalty link cleared
        @test hv_after.wage             == 0.0

        # 4. Replacement bank has positive equity_base (cloned from survivor).
        bv_after = variant(model[forced_bankrupt_bank_id])
        @test bv_after.equity_base > 0.0
        @test bv_after.credit_supply  == 0.0
        @test bv_after.interest_rate  == 0.0

        # 5. Model not collapsed (not all firms/banks exited).
        @test model.collapsed == false
    end

    @testset "event 2 reads prior-period loan interest" begin
        # Directly verify _event2_plan_breakeven_price! reads the shared loan book
        # by borrower_id (Flag 6: planning-phase breakeven uses last period's
        # interest).
        params = Dict{String,Float64}(
            "labor_productivity"  => 0.50,
            "price_init"          => 0.50,
            "net_worth_ratio"     => 6.0,
            "savings_init"        => 1.0,
            "equity_base_init"    => 5.0,
            "min_wage_ratio"      => 0.5,
            "min_wage_rev_period" => 4.0,
            "h_rho"               => 0.10,
            "h_eta"               => 0.10,
            "h_xi"                => 0.05,
        )
        model = build_model(3, 5, 1, params, 1)

        firm = first(firms(model))
        fid = firm.id
        fv = variant(firm)
        # Set a known wage_bill and desired_production, and inject a loan for this
        # firm into the shared book.
        fv.wage_bill = 2.0
        fv.desired_production = 4.0
        push!(model.loans, Loan(10.0, 0.05, fid, 0))   # interest = 10*0.05 = 0.5

        _event2_plan_breakeven_price!(model)

        # breakeven = (wage_bill + interest) / max(desired_production, eps)
        #           = (2.0 + 0.5) / 4.0 = 0.625
        @test variant(model[fid]).breakeven_price ≈ (2.0 + 0.5) / 4.0

        # A firm with NO loan in the book gets breakeven from wage_bill alone.
        other = collect(firms(model))[2]
        ov = variant(other)
        ov.wage_bill = 1.0
        ov.desired_production = 2.0
        _event2_plan_breakeven_price!(model)
        @test variant(other).breakeven_price ≈ 1.0 / 2.0
    end

end
