"""Smoke test for the Mesa subprocess runner."""

import os
import subprocess
import sys

from comparison.orchestrator.contract import STATUS_OK, RunRequest, RunResult
from comparison.orchestrator.params import canonical_params, population_for


def test_mesa_runner_smoke(tmp_path):
    req = RunRequest(
        run_id="mesa__s100__seed7__rep0",
        framework="mesa",
        model_params=canonical_params(),
        population=population_for(20),
        n_periods=40,
        warmup_periods=5,
        seed=7,
        collect_outputs=True,
        outputs_requested=[
            "unemployment",
            "price_inflation",
            "wage_inflation",
            "log_gdp",
            "vacancy_rate",
            "production_final",
        ],
    )
    path = tmp_path / "req.json"
    path.write_text(req.to_json())
    proc = subprocess.run(
        [sys.executable, "-m", "comparison.runners.mesa.run", str(path)],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert proc.returncode == 0, proc.stderr
    res = RunResult.from_json(proc.stdout.strip().splitlines()[-1])
    assert res.status == STATUS_OK
    assert res.validate() == []
    assert res.timing["steady_state_per_period_seconds"] > 0
    assert len(res.outputs["unemployment"]) == 40
    assert len(res.outputs["production_final"]) == 20
