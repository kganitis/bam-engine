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
|            610 | 0.000367128 | 0.000632405 | 0.00160879 |  0.00922023 |
|           1220 | 0.000554917 | 0.000848363 | 0.00303769 |  0.00601375 |
|           3050 |  0.00145566 |  0.00167956 | 0.00774993 |   0.0123413 |
|           6100 |  0.00311091 |  0.00291971 |  0.0156046 |    0.022912 |
|          12200 |  0.00722699 |   0.0053335 |  0.0317142 |    0.043104 |
|          30500 |   0.0187075 |   0.0125447 |  0.0876855 |    0.105179 |
|          61000 |   0.0442757 |   0.0259051 |        nan |         nan |
|         122000 |    0.110145 |   0.0602835 |        nan |         nan |

## Notes

- Single-thread pinned. Timing runs serial; gate runs parallel.
- Adaptive cap: see skips.json for sizes dropped per framework.
