# Cross-Framework Benchmark Report

Environment: arm (12 cores)

## Fidelity (behavioral-equivalence gate)

- bamengine: PASS (non-blocking reference)
- mesa: PASS

## Steady-state per-period time (median seconds)

| n_agents_total |  bamengine |       mesa |
| -------------: | ---------: | ---------: |
|            610 | 0.00108719 | 0.00163034 |
|           1220 | 0.00225887 | 0.00301161 |
|           3050 | 0.00997742 | 0.00767941 |
|           6100 |  0.0344326 |  0.0155378 |
|          12200 |        nan |  0.0314757 |
|          30500 |        nan |  0.0857822 |

## Notes

- Single-thread pinned. Timing runs serial; gate runs parallel.
- Adaptive cap: see skips.json for sizes dropped per framework.
