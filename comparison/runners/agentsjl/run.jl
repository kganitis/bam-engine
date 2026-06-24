"""
run.jl - Agents.jl BAM runner for the cross-framework comparison harness.

Usage:
    julia --project=comparison/runners/agentsjl --startup-file=no run.jl <request.json>

Reads a RunRequest JSON file (path in ARGS[1]), runs the Agents.jl BAM model,
and prints a RunResult-shaped JSON as the FINAL line of stdout. The orchestrator
parses the last stdout line as a RunResult (see comparison/orchestrator/contract.py).

Two run modes (driven by collect_outputs):

* collect_outputs=true  (gate runs): one contiguous run of n_periods with
  collection on, producing FULL-LENGTH per-period series (length == n_periods).
  Burn-in is applied centrally by the orchestrator, not here.

* collect_outputs=false (timing runs): an UNTIMED warmup run (warmup_periods,
  no collection) followed by a TIMED steady-state run (n_periods - warmup_periods,
  no collection). The warmup forces Julia JIT compilation so the timed loop
  measures compiled steady-state performance, not first-run overhead.

The six output series match the Mesa and bamengine runners exactly:
  * unemployment      : 1 - employed_count / n_households, per period
  * price_inflation   : inflation_history value, per period (YoY)
  * wage_inflation    : period-over-period growth of average employed wage
  * log_gdp           : log of total production, per period
  * vacancy_rate      : total_vacancies / n_households, per period
  * production_final  : per-firm production snapshot at the last period

Derived from model.jl collection fields populated in bam_step! (Task 9).
"""

using JSON3
using Agents

# Include the model (which includes markets.jl via include inside model.jl).
include(joinpath(@__DIR__, "model.jl"))

# ---------------------------------------------------------------------------
# Schema constants (must match comparison/orchestrator/contract.py)
# ---------------------------------------------------------------------------
const SCHEMA_VERSION = "1.0"
const STATUS_OK      = "ok"
const STATUS_ERROR   = "error"

# ---------------------------------------------------------------------------
# Series derivation (mirrors Mesa run.py build_series and bamengine run.py)
# ---------------------------------------------------------------------------

"""
    build_series(model) -> Dict{String, Vector{Float64}}

Build the six comparison output series from a model run with collect=true.

Series definitions (exact match with Mesa build_series and bamengine build_series):

  unemployment    : model.c_unemployment (collected after event 22)
  price_inflation : model.c_inflation (collected after event 7, from inflation_history)
  wage_inflation  : period-over-period growth of model.c_avg_employed_wage
  log_gdp         : log(model.c_total_production), NaN replaced with 0.0
  vacancy_rate    : model.c_total_vacancies / n_households
  production_final: model.c_production_final (per-firm, last-period snapshot)
"""
function build_series(model)::Dict{String, Vector{Float64}}
    unemployment     = copy(model.c_unemployment)
    avg_wage         = copy(model.c_avg_employed_wage)
    total_production = copy(model.c_total_production)
    total_vacancies  = copy(model.c_total_vacancies)
    price_inflation  = copy(model.c_inflation)
    production_final = copy(model.c_production_final)

    n = length(unemployment)

    # Wage inflation: period-over-period growth of the average employed wage.
    # wage_inflation[1] = 0.0 (no previous period); wage_inflation[i] = avg_wage[i]/avg_wage[i-1] - 1
    # if avg_wage[i-1] != 0, else 0.0. Matches Mesa and bamengine derivations.
    wage_inflation = zeros(Float64, n)
    for i in 2:n
        prev = avg_wage[i-1]
        if prev != 0.0
            wage_inflation[i] = avg_wage[i] / prev - 1.0
        end
    end

    # Log GDP: log of total production; replace non-finite with 0.0.
    log_gdp = zeros(Float64, n)
    for i in 1:n
        g = total_production[i]
        log_gdp[i] = g > 0.0 ? log(g) : 0.0
    end

    # Vacancy rate: total vacancies divided by n_households (labor force).
    n_hh = max(1, model.n_households)
    vacancy_rate = total_vacancies ./ Float64(n_hh)

    # Replace any NaN/Inf in all series with 0.0 (matches numpy nan_to_num).
    _sanitize!(unemployment)
    _sanitize!(price_inflation)
    _sanitize!(wage_inflation)
    _sanitize!(log_gdp)
    _sanitize!(vacancy_rate)
    _sanitize!(production_final)

    return Dict{String, Vector{Float64}}(
        "unemployment"     => unemployment,
        "price_inflation"  => price_inflation,
        "wage_inflation"   => wage_inflation,
        "log_gdp"          => log_gdp,
        "vacancy_rate"     => vacancy_rate,
        "production_final" => production_final,
    )
end

"Replace NaN and Inf in-place with 0.0 (mirrors numpy nan_to_num default)."
function _sanitize!(v::Vector{Float64})
    for i in eachindex(v)
        if !isfinite(v[i])
            v[i] = 0.0
        end
    end
    return v
end

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

function main()
    if length(ARGS) < 1
        error("Usage: julia run.jl <request.json>")
    end

    request_path = ARGS[1]
    req_text     = read(request_path, String)
    req          = JSON3.read(req_text)

    # Resolve framework version: pkgversion(Agents) gives the installed version.
    framework_version = string(pkgversion(Agents))
    language_version  = string(VERSION)

    n_firms      = Int(req.population.n_firms)
    n_households = Int(req.population.n_households)
    n_banks      = Int(req.population.n_banks)
    n_agents     = n_firms + n_households + n_banks
    seed         = Int(req.seed)
    n_periods    = Int(req.n_periods)
    warmup       = Int(req.warmup_periods)

    # Convert model_params JSON object to Dict{String,Float64}, skipping
    # non-numeric entries (e.g. string params like "job_search_method").
    model_params = Dict{String,Float64}()
    for (k, v) in pairs(req.model_params)
        if v isa Number
            model_params[string(k)] = Float64(v)
        end
    end

    try
        outputs = nothing

        if req.collect_outputs
            # Gate run: one contiguous run with collection on.
            t_init0 = time_ns()
            model = build_model(n_firms, n_households, n_banks, model_params, seed;
                                collect = true)
            t_init1 = time_ns()
            init_seconds = (t_init1 - t_init0) / 1e9

            t_run0 = time_ns()
            for _ in 1:n_periods
                bam_step!(model)
                model.collapsed && break
            end
            t_run1 = time_ns()
            run_seconds   = (t_run1 - t_run0) / 1e9
            timed_periods = n_periods

            outputs = build_series(model)
        else
            # Timing run: untimed warmup (compiles + warms JIT) then timed loop.
            warmup_actual  = max(0, warmup)
            timed_periods  = max(1, n_periods - warmup_actual)

            t_init0 = time_ns()
            model = build_model(n_firms, n_households, n_banks, model_params, seed;
                                collect = false)
            t_init1 = time_ns()
            init_seconds = (t_init1 - t_init0) / 1e9

            # Warmup: run untimed to force JIT compilation of all hot paths.
            if warmup_actual > 0
                for _ in 1:warmup_actual
                    bam_step!(model)
                    model.collapsed && break
                end
            end

            # Timed steady-state run.
            t_run0 = time_ns()
            for _ in 1:timed_periods
                bam_step!(model)
                model.collapsed && break
            end
            t_run1 = time_ns()
            run_seconds = (t_run1 - t_run0) / 1e9
        end

        steady     = timed_periods > 0 ? run_seconds / timed_periods : run_seconds
        throughput = run_seconds > 0   ? n_agents * timed_periods / run_seconds : 0.0

        result = Dict{String,Any}(
            "schema_version"    => SCHEMA_VERSION,
            "run_id"            => string(req.run_id),
            "framework"         => "agentsjl",
            "framework_version" => framework_version,
            "language"          => "julia",
            "language_version"  => language_version,
            "status"            => STATUS_OK,
            "error"             => nothing,
            "population"        => Dict{String,Any}(
                "n_firms"        => n_firms,
                "n_households"   => n_households,
                "n_banks"        => n_banks,
                "n_agents_total" => n_agents,
            ),
            "n_periods"         => n_periods,
            "warmup_periods"    => warmup,
            "seed"              => seed,
            "timing"            => Dict{String,Any}(
                "init_seconds"                       => init_seconds,
                "run_seconds"                        => run_seconds,
                "steady_state_per_period_seconds"    => steady,
                "throughput_agent_steps_per_s"       => throughput,
            ),
            "outputs"           => outputs,
        )

        println(JSON3.write(result))

    catch err
        tb  = sprint(showerror, err, catch_backtrace())
        result = Dict{String,Any}(
            "schema_version"    => SCHEMA_VERSION,
            "run_id"            => get(req, :run_id, "unknown"),
            "framework"         => "agentsjl",
            "framework_version" => framework_version,
            "language"          => "julia",
            "language_version"  => language_version,
            "status"            => STATUS_ERROR,
            "error"             => tb,
            "population"        => Dict{String,Any}(
                "n_firms"      => get(req.population, :n_firms,      0),
                "n_households" => get(req.population, :n_households, 0),
                "n_banks"      => get(req.population, :n_banks,      0),
            ),
            "n_periods"         => get(req, :n_periods, 0),
            "warmup_periods"    => get(req, :warmup_periods, 0),
            "seed"              => get(req, :seed, 0),
            "timing"            => Dict{String,Any}(),
            "outputs"           => nothing,
        )
        println(JSON3.write(result))
    end
end

main()
