"""
runtests.jl - Julia test harness for the Agents.jl BAM runner.

Run with:
    julia --project=comparison/runners/agentsjl --startup-file=no comparison/runners/agentsjl/test/runtests.jl

Tests are added here as the BAM model implementation progresses (Tasks 2-9).
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
            "labor_productivity" => 0.50,
            "price_init"         => 0.50,
            "net_worth_ratio"    => 6.0,
            "savings_init"       => 1.0,
            "equity_base_init"   => 5.0,
            "min_wage_ratio"     => 0.5,
            "min_wage_rev_period" => 4.0,
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

        # --- A no-op step! advances without error (bam_step! for now) ---
        step!(model, 1)
        @test nagents(model) == n_firms + n_households + n_banks
    end

end
