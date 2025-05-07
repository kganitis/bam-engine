
## Documentation

1. **Docstrings**: adopt the *NumPy docstring standard* everywhere and mark `out` parameters (`← out`) consistently (your planning docstrings already follow this).
2. **Lint noise**: Ruff config selects only `E, F, W, I`. Consider adding `D` (docstring) once code stabilises.

---

## Performance quick-wins

* `workers_prepare_applications`: use permanent scratch buffers

---

## Later, but keep in mind

* **Parallel execution**: When n\_firms >> 1M, you’ll want *threaded* NumPy or distribute firms across processes. Design data layout (SoA vs AoS) accordingly.
* **Streaming / State IO**: Provide `Scheduler.to_dict()` / `from_dict()` for checkpoint‑restart (important when running hundreds of Monte‑Carlo paths).
* **Interfaces**: Expose a light **`pandas` view** (DataFrame per component) only for *read‑only* downstream analysis; keep core arrays untouched.
* **Policy experiments**: The hook mechanism can evolve into **event bus**; consider `Plugin` objects with `on_event(self, scheduler, tag)` for user‑defined rules.

---

## Small items you can safely defer

| Area                                                 | Why it can wait                                      | Concrete follow‑up when convenient                                                 |
|------------------------------------------------------| ---------------------------------------------------- |------------------------------------------------------------------------------------|
| **Docstrings / API docs**                            | Developer‑facing code is still in flux.              | Finish NumPy‑style docstrings once the full event chain is in.                     |
| **Version bump**                                     | No external users yet.                               | Switch to `0.1.0` the moment you publish the first PyPI wheel.                  |
| **Performance regression**                           | No timing guard yet.                                 | Create a **`pytest‑benchmark`** suite later; fail if step‑time > xms for base size.|
| **Numba JIT**                                        | Profile first‑—CPU is nowhere near a bottleneck yet. | Add `@njit(cache=True)` to pure kernels *after* the rule set stabilises.           |
| **Logging cost in tight loops**                      | Overhead ≈ 0 ms at current sizes.             | Guard heavy `log.debug(...)` blocks with `if log.isEnabledFor(...)`.               |
| **`np.ceil` optimisation in `decide_desired_labor`** | Only 10 µs for 100 k firms.            | Implement the branch‑free floor‑divide trick when you measure >5% runtime there.   |
| **Checkpoint I/O (`snapshot()`)**                    | Not needed until you run long Monte‑Carlo batches.   | Design a simple `dict`/NPZ serialiser later.                                       |
