---
name: Bug report
about: Report a problem with BAM Engine
title: "[Bug] "
labels: bug
---

## Description

<!-- A clear description of the bug. -->

## Reproduction

<!-- Minimal code to reproduce. Include the seed and parameters so the run is deterministic. -->

```python
import bamengine as bam

sim = bam.Simulation.init(seed=42)
results = sim.run(n_periods=1000)
```

## Expected vs. actual behavior

<!-- What did you expect to happen, and what happened instead? -->

## Environment

- BAM Engine version:
- Python version:
- OS:
- Installed with the fast (numba) extra? (`pip install bamengine[fast]`):
