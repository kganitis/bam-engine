# Cross-Framework Benchmark Report

Environment: arm (12 cores)

## Fidelity (behavioral-equivalence gate)

- bamengine: PASS (non-blocking reference)
- mesa: PASS
- mesa_frames: PASS
- agentsjl: PASS

## Steady-state per-period time (median seconds)

| n_agents_total |    agentsjl |   bamengine |       mesa | mesa_frames |
| -------------: | ----------: | ----------: | ---------: | ----------: |
|            610 | 0.000649679 | 0.000718136 |  0.0017986 |   0.0118309 |
|           1220 |  0.00160206 | 0.000856093 | 0.00304463 |    0.010006 |
|           3050 |  0.00806117 |  0.00169687 | 0.00778395 |   0.0230312 |
|           6100 |    0.027675 |  0.00297308 |  0.0157879 |   0.0429527 |
|          12200 |   0.0967138 |  0.00531023 |  0.0323102 |   0.0829817 |
|          30500 |         nan |   0.0126993 |  0.0882926 |         nan |
|          61000 |         nan |   0.0267146 |        nan |         nan |
|         122000 |         nan |   0.0625943 |        nan |         nan |

## Notes

- Single-thread pinned. Timing runs serial; gate runs parallel.
- Adaptive cap: see skips.json for sizes dropped per framework.
