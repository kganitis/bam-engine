# Stability Benchmark Results

JSON files in this directory are produced by `benchmarks/bench_seed_stability.py`
and committed to the repository. The `validation-status` CI workflow reads these
files to determine whether the model passes validation (96.2% pass rate across
1000 seeds per scenario).

## Producing results

```bash
PYTHONPATH=src python benchmarks/bench_seed_stability.py
```

## File naming

`{scenario}_{commit_short}_{timestamp}.json`

The workflow reads the most recent file per scenario (by reverse filename sort).
