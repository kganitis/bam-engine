"""
run.jl - Agents.jl BAM runner stub for the cross-framework comparison harness.

Usage:
    julia --project=comparison/runners/agentsjl --startup-file=no run.jl <request.json>

Reads a RunRequest JSON file (path in ARGS[1]), and prints a RunResult-shaped
JSON to the FINAL line of stdout.  The orchestrator parses the last stdout line
as a RunResult (see comparison/orchestrator/contract.py).

This stub always returns status "error" with message
"agentsjl model not yet implemented".  Later tasks replace it with the full
BAM Agents.jl model.
"""

using JSON3

# ---------------------------------------------------------------------------
# Schema constants (must match comparison/orchestrator/contract.py)
# ---------------------------------------------------------------------------
const SCHEMA_VERSION = "1.0"
const STATUS_ERROR   = "error"

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
function main()
    if length(ARGS) < 1
        error("Usage: julia run.jl <request.json>")
    end

    request_path = ARGS[1]
    req_text     = read(request_path, String)
    req          = JSON3.read(req_text)

    # Build a minimal RunResult with error status.
    # All fields from RunResult dataclass must be present so from_json() works.
    result = Dict{String,Any}(
        "schema_version"   => SCHEMA_VERSION,
        "run_id"           => get(req, :run_id, "unknown"),
        "framework"        => "agentsjl",
        "framework_version" => string(pkgversion(@__MODULE__)),
        "language"         => "julia",
        "language_version" => string(VERSION),
        "status"           => STATUS_ERROR,
        "error"            => "agentsjl model not yet implemented",
        "population"       => get(req, :population, Dict()),
        "n_periods"        => get(req, :n_periods, 0),
        "warmup_periods"   => get(req, :warmup_periods, 0),
        "seed"             => get(req, :seed, 0),
        "timing"           => Dict{String,Any}(),
        "outputs"          => nothing,
    )

    # Print the JSON as the LAST line of stdout (orchestrator reads last line).
    println(JSON3.write(result))
end

main()
