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
|            610 | 0.000168658 |    0.000673 |  0.0016356 |   0.0091341 |
|           1220 | 0.000426796 | 0.000856278 |  0.0030617 |   0.0055647 |
|           3050 | 0.000984987 |  0.00170461 | 0.00780683 |   0.0107559 |
|           6100 |  0.00211067 |  0.00291123 |  0.0156222 |   0.0191032 |
|          12200 |  0.00460735 |  0.00528846 |  0.0317776 |   0.0351193 |
|          30500 |   0.0129323 |   0.0125842 |  0.0887603 |   0.0858525 |
|          61000 |   0.0302752 |   0.0255235 |        nan |         nan |
|         122000 |   0.0768054 |   0.0641805 |        nan |         nan |

## Notes

- Single-thread pinned. Timing runs serial; gate runs parallel.
- Adaptive cap: see skips.json for sizes dropped per framework.
