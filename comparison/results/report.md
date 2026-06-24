# Cross-Framework Benchmark Report

Environment: arm (12 cores)

## Fidelity (behavioral-equivalence gate)

- bamengine: PASS (non-blocking reference)
- mesa: PASS
- mesa_frames: PASS

## Steady-state per-period time (median seconds)

| n_agents_total |   bamengine |       mesa | mesa_frames |
| -------------: | ----------: | ---------: | ----------: |
|            610 | 0.000767147 | 0.00183049 |   0.0128854 |
|           1220 | 0.000916907 |  0.0032083 |   0.0110702 |
|           3050 |   0.0017984 | 0.00825832 |   0.0245186 |
|           6100 |  0.00314454 |  0.0168181 |   0.0475281 |
|          12200 |  0.00573866 |   0.034887 |   0.0884752 |
|          30500 |   0.0142159 |   0.100439 |         nan |
|          61000 |     0.03057 |        nan |         nan |
|         122000 |   0.0789422 |        nan |         nan |

## Notes

- Single-thread pinned. Timing runs serial; gate runs parallel.
- Adaptive cap: see skips.json for sizes dropped per framework.
