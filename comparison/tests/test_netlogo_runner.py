import os
import subprocess
import sys

from comparison.orchestrator.contract import RunRequest, RunResult
from comparison.orchestrator.params import canonical_params, population_for
from comparison.orchestrator.run import RUNNER_CMD


def _request() -> RunRequest:
    return RunRequest(
        run_id="netlogo__gate__seed0",
        framework="netlogo",
        model_params=canonical_params(),
        population=population_for(100),
        n_periods=60,
        warmup_periods=0,
        seed=0,
        collect_outputs=True,
        outputs_requested=["unemployment"],
    )


def test_netlogo_registered_in_runner_cmd():
    assert "netlogo" in RUNNER_CMD
    assert RUNNER_CMD["netlogo"][1:] == ["-m", "comparison.runners.netlogo.run"]


def test_netlogo_runner_skips_without_toolchain(tmp_path):
    # Run the stub with the CURRENT interpreter (which has no pynetlogo). It must
    # emit a valid, parseable RunResult with status "skipped", never crash.
    path = tmp_path / "req.json"
    path.write_text(_request().to_json())
    proc = subprocess.run(
        [sys.executable, "-m", "comparison.runners.netlogo.run", str(path)],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert proc.returncode == 0, proc.stderr
    res = RunResult.from_json(proc.stdout.strip().splitlines()[-1])
    assert res.framework == "netlogo"
    assert res.status == "skipped"
