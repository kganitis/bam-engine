# Cross-Framework Benchmark Report

Environment: arm (12 cores)

## Fidelity (behavioral-equivalence gate)

- bamengine: PASS (non-blocking reference)
- mesa: PASS

## Steady-state per-period time (median seconds)

| n_agents_total |   bamengine |       mesa |
| -------------: | ----------: | ---------: |
|            610 | 0.000790053 | 0.00170842 |
|           1220 |  0.00107303 | 0.00325749 |
|           3050 |   0.0021604 | 0.00836017 |
|           6100 |  0.00390581 |  0.0168762 |
|          12200 |  0.00723911 |  0.0348351 |
|          30500 |   0.0180633 |  0.0975991 |
|          61000 |   0.0384871 |        nan |
|         122000 |   0.0959153 |        nan |

## Notes

- Single-thread pinned. Timing runs serial; gate runs parallel.
- Adaptive cap: see skips.json for sizes dropped per framework.
