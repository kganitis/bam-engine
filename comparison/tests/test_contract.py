from comparison.orchestrator.contract import (
    SCHEMA_VERSION,
    STATUS_OK,
    RunRequest,
    RunResult,
)


def test_request_roundtrip():
    req = RunRequest(
        run_id="bamengine__s100__seed7__rep0",
        framework="bamengine",
        model_params={"markup": 0.1},
        population={"n_firms": 100, "n_households": 500, "n_banks": 10},
        n_periods=1000,
        warmup_periods=100,
        seed=7,
        collect_outputs=True,
        outputs_requested=["unemployment"],
    )
    assert RunRequest.from_json(req.to_json()) == req


def test_result_validate_ok():
    res = RunResult(
        schema_version=SCHEMA_VERSION,
        run_id="x",
        framework="bamengine",
        framework_version="0.9.1",
        language="python",
        language_version="3.13.1",
        status=STATUS_OK,
        error=None,
        population={
            "n_firms": 100,
            "n_households": 500,
            "n_banks": 10,
            "n_agents_total": 610,
        },
        n_periods=1000,
        warmup_periods=100,
        seed=7,
        timing={
            "init_seconds": 0.1,
            "run_seconds": 1.0,
            "steady_state_per_period_seconds": 0.001,
            "throughput_agent_steps_per_s": 1e6,
        },
        outputs=None,
    )
    assert res.validate() == []


def test_result_validate_flags_missing_timing_key():
    res = RunResult(
        schema_version=SCHEMA_VERSION,
        run_id="x",
        framework="bamengine",
        framework_version="0.9.1",
        language="python",
        language_version="3.13.1",
        status=STATUS_OK,
        error=None,
        population={
            "n_firms": 100,
            "n_households": 500,
            "n_banks": 10,
            "n_agents_total": 610,
        },
        n_periods=1000,
        warmup_periods=100,
        seed=7,
        timing={"init_seconds": 0.1},  # missing keys
        outputs=None,
    )
    assert any("steady_state_per_period_seconds" in p for p in res.validate())
