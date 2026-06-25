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
|            610 | 0.000162356 | 0.000751699 | 0.00159998 |  0.00811484 |
|           1220 | 0.000436972 | 0.000607188 | 0.00305794 |  0.00556281 |
|           3050 | 0.000986374 | 0.000944456 | 0.00782697 |   0.0108459 |
|           6100 |  0.00216642 |  0.00145254 |  0.0157361 |   0.0190272 |
|          12200 |  0.00467995 |  0.00242839 |  0.0325397 |   0.0348472 |
|          30500 |   0.0131903 |  0.00538807 |  0.0897447 |   0.0847395 |
|          61000 |   0.0320794 |   0.0105074 |        nan |         nan |
|         122000 |   0.0776806 |   0.0218867 |        nan |         nan |

## Notes

- Single-thread pinned. Timing runs serial; gate runs parallel.
- Adaptive cap: see skips.json for sizes dropped per framework.
