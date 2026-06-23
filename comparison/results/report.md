# Cross-Framework Benchmark Report

Environment: arm (12 cores)

## Fidelity (behavioral-equivalence gate)

- bamengine: PASS (non-blocking reference)
- mesa: PASS

## Steady-state per-period time (median seconds)

| n_agents_total |   bamengine |       mesa |
| -------------: | ----------: | ---------: |
|            610 | 0.000683919 | 0.00170402 |
|           1220 | 0.000924449 | 0.00325025 |
|           3050 |  0.00179879 | 0.00833744 |
|           6100 |  0.00312065 |  0.0169489 |
|          12200 |  0.00567981 |  0.0344806 |
|          30500 |   0.0136706 |  0.0953121 |
|          61000 |   0.0288476 |        nan |
|         122000 |   0.0690565 |        nan |

## Notes

- Single-thread pinned. Timing runs serial; gate runs parallel.
- Adaptive cap: see skips.json for sizes dropped per framework.
