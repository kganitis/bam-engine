## Summary

<!-- What does this PR do, and why? Link the related issue, e.g. "Closes #123". -->

## Type of change

- [ ] Bug fix
- [ ] New feature / extension
- [ ] Documentation
- [ ] Performance
- [ ] Other:

## Checklist

- [ ] For a substantial or model-behavior change, I opened an issue first to discuss the approach
- [ ] `pytest` passes and coverage stays at 99%
- [ ] `ruff format . && ruff check --fix .` is clean
- [ ] `mypy` is clean
- [ ] New code has tests and NumPy-style docstrings
- [ ] All randomness goes through `sim.rng` (no direct `np.random`)
