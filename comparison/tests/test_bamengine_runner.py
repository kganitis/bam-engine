import os
import subprocess
import sys

from comparison.orchestrator.contract import STATUS_OK, RunRequest, RunResult
from comparison.orchestrator.params import canonical_params, population_for


def test_population_ratio():
    assert population_for(100) == {"n_firms": 100, "n_households": 500, "n_banks": 10}
    assert population_for(2000) == {
        "n_firms": 2000,
        "n_households": 10000,
        "n_banks": 200,
    }


def test_canonical_params_nonempty_and_no_population_keys():
    p = canonical_params()
    assert p
    assert not ({"n_firms", "n_households", "n_banks", "n_periods"} & set(p))


def test_bamengine_runner_smoke(tmp_path):
    req = RunRequest(
        run_id="bamengine__s100__seed7__rep0",
        framework="bamengine",
        model_params=canonical_params(),
        population=population_for(100),
        n_periods=60,
        warmup_periods=10,
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
        [sys.executable, "-m", "comparison.runners.bamengine.run", str(path)],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert proc.returncode == 0, proc.stderr
    res = RunResult.from_json(proc.stdout.strip().splitlines()[-1])
    assert res.status == STATUS_OK
    assert res.validate() == []
    assert res.timing["steady_state_per_period_seconds"] > 0
    assert len(res.outputs["unemployment"]) == 60
    assert len(res.outputs["production_final"]) == 100
