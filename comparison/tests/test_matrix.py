from comparison.orchestrator.matrix import (
    GATE_SEEDS,
    SCALE_FIRMS,
    TIMING_REPS,
    equivalence_jobs,
    timing_jobs,
)


def test_gate_jobs_use_baseline_size_and_20_seeds():
    jobs = equivalence_jobs(["bamengine"])
    assert len(jobs) == len(GATE_SEEDS) == 20
    assert all(j.population["n_firms"] == 100 and j.collect_outputs for j in jobs)


def test_timing_jobs_cover_all_sizes_and_reps():
    jobs = timing_jobs(["bamengine"])
    assert len(jobs) == len(SCALE_FIRMS) * TIMING_REPS
    assert all(not j.collect_outputs for j in jobs)
