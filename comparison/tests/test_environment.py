from comparison.orchestrator.environment import capture_environment, environment_id

REQUIRED = {
    "cpu",
    "n_cores",
    "ram_bytes",
    "os",
    "python_version",
    "bamengine_version",
    "bamengine_commit",
    "thread_env",
    "timestamp",
}


def test_capture_has_required_keys():
    env = capture_environment()
    assert set(env) >= REQUIRED
    assert env["n_cores"] >= 1


def test_environment_id_is_stable_and_short():
    env = capture_environment()
    a, b = environment_id(env), environment_id(env)
    assert a == b
    assert 6 <= len(a) <= 16
