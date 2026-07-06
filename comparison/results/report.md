# Cross-Framework Benchmark Report

Environment: arm (12 cores)

## Fidelity (behavioral-equivalence gate)

- bamengine: PASS (non-blocking reference)
- mesa: PASS
- mesa_frames: PASS
- agentsjl: PASS
- netlogo: FAIL (non-blocking reference)

## Steady-state per-period time (median seconds)

| n_agents_total |    agentsjl |   bamengine |       mesa | mesa_frames |   netlogo |
| -------------: | ----------: | ----------: | ---------: | ----------: | --------: |
|            610 | 0.000308341 |  0.00106509 | 0.00226041 |   0.0118325 | 0.0318986 |
|           1220 | 0.000425985 | 0.000617328 | 0.00321743 |  0.00542641 | 0.0100971 |
|           3050 |  0.00100956 | 0.000962656 | 0.00794804 |   0.0107434 | 0.0416751 |
|           6100 |  0.00216275 |  0.00149424 |  0.0169318 |   0.0190968 |       nan |
|          12200 |  0.00468099 |  0.00254497 |  0.0333929 |   0.0350415 |       nan |
|          30500 |   0.0132845 |  0.00560743 |   0.091484 |   0.0876384 |       nan |
|          61000 |   0.0305484 |   0.0107199 |        nan |         nan |       nan |
|         122000 |   0.0822033 |   0.0224868 |        nan |         nan |       nan |

## Notes

- Single-thread pinned. Timing runs serial; gate runs parallel.
- Adaptive cap: see skips.json for sizes dropped per framework.
