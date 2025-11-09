# BAM Engine - Benchmarks

Performance benchmarking and profiling tools for BAM Engine.

## Quick Start

```bash
# Run macro-benchmarks (full simulation)
python benchmarks/bench_full_simulation.py

# Run micro-benchmarks (individual events)
python benchmarks/micro/bench_events.py

# Profile with cProfile
python benchmarks/profile_simulation.py

# Analyze profile with snakeviz (requires: pip install snakeviz)
snakeviz simulation_profile.prof
```

## Benchmark Scripts

### Macro-Benchmarks (`bench_full_simulation.py`)

Benchmarks end-to-end performance of full simulation runs with various configurations:

- **Small**: 100 firms, 500 households, 100 periods
- **Medium**: 200 firms, 1,000 households, 100 periods
- **Large**: 500 firms, 2,500 households, 100 periods

Each configuration runs 10 times with different seeds to measure mean and standard deviation.

**Output**:

- Average time per configuration
- Throughput (periods/second)
- Min/Max times
- Summary table

### Micro-Benchmarks (`micro/bench_events.py`)

Benchmarks individual event execution times to identify expensive operations.

- Runs each event 100 times (after 10 warmup runs)
- Measures mean and standard deviation
- Shows top 10 most expensive events

**Note**: Some events may not be benchmarkable individually as they depend on simulation state.

### Profiling (`profile_simulation.py`)

Uses Python's cProfile to identify performance bottlenecks:

- Runs 100 firms, 500 households, 100 periods
- Sorts by cumulative time and total time
- Saves profile to `simulation_profile.prof`

**Analysis**:

```bash
# Terminal output
python benchmarks/profile_simulation.py

# Visual analysis (requires snakeviz)
pip install snakeviz
snakeviz simulation_profile.prof
```

## Performance Regression Tests

Located in `tests/performance/test_regression.py`:

```bash
# Run performance regression tests
pytest tests/performance/ -v

# Skip slow tests
pytest tests/performance/ -v -m "not slow"
```

These tests ensure performance doesn't regress beyond acceptable thresholds.

**Baselines** (established November 9, 2025):

- Small: 0.72s (pytest framework overhead included)
- Medium: 1.20s
- Large: 2.80s

**Threshold**: 15% slower than baseline (allows for test framework variability)

## Benchmark Results

See `BENCHMARK_RESULTS.md` for detailed performance analysis and historical baselines.

## Best Practices

1. **Disable logging**: All benchmarks disable logging (`logging.ERROR`) to measure pure computation time
2. **Multiple runs**: Use `n_runs=10` or more to account for system noise
3. **Warmup**: Run a few iterations before timing to warm up JIT/caches
4. **Consistent environment**: Run benchmarks on same hardware for comparison
5. **Update baselines**: After confirmed performance improvements, update baselines in:
   - `BENCHMARK_RESULTS.md`
   - `tests/performance/test_regression.py`

## Interpreting Results

### Good Performance

- Within 5% of baseline (excellent)
- Within 10% of baseline (good)
- Variance < 5% of mean (consistent)

### Performance Regression

- > 15% slower than baseline (investigate)
- High variance (> 10% of mean) - system noise or non-determinism

### Profiling Hotspots

Common bottlenecks:

- Market queuing loops (`max_M`, `max_H`, `max_Z`)
- NumPy aggregations (`np.add.at`, `np.bincount`)
- Random number generation

See PLAN.md Section 13.7 for optimization strategies.

## Tools

Optional tools for deeper analysis:

```bash
# Visual profiling
pip install snakeviz

# Line-level profiling
pip install line_profiler

# Memory profiling
pip install memory_profiler

# Sampling profiler
pip install py-spy
```

## Notes

- Benchmarks use fixed seeds for reproducibility
- Results are hardware-dependent (times will vary)
- Focus on **relative** performance (new vs old) not absolute times
- Pure benchmark times (without pytest) are ~20-25% faster than regression test baselines
