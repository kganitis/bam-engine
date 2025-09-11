# BAM-Engine roadmap

### TODO Next

* Test coverage
* Readme

---

### Low-priority housekeeping

* Use consistent naming: e.g. when to use `firm` vs `borrower`/`employer`
* How to deal with _EPS values: where to use them, how often they actually occur
* Isolate rng based operations for easier testing
* [DEBUG] First N --> N (DEBUG depth) set by user
* Review project.optional-dependencies and write the minimum required
* Keep the Ruff rule set minimal (`E, F, W, I` plus docstrings later)
* Introduce pytest-benchmark and fail the suite on measurable performance regressions
* Remove dead lint-ignore comments and finalise design diagrams once the API is frozen

---

### ECS Architecture
* Refactor to a true ECS architecture, with `World` object & components query mechanism
* Prototype the systems so that each system by default accepts a before & after hook, or other common logic
* Each system must use and operate on the minimum possible number of vectors
* Review the components structure
  * Need of fields reshuffling, combine components or break them into pieces
  * Based on the systems, so that we optimize speed and memory
  * Remove any shared vector references
  * Introduce an abstract ledger data structure and a job contracts ledger
* Also review which systems belong to which events, general event structure
* Watch vectors lifetime, review which are needed temporarily and which permanently, then optimize
* Dynamic scheduler re-wiring by the user, add/remove systems, change their order
  * Step should just decide which systems are called in what order
 
---

### Performance milestone

* Test removal of pre-allocated buffers and queues, use 2D arrays with standard NumPy operations
* Profile representative sizes to establish baselines
* Remove remaining temporary allocations and other obvious hotspots
  * e.g. reuse permanent scratch buffers in `workers_decide_firms_to_apply` and `firms_prepare_loan_applications`
* Enable threaded NumPy or process-level sharding of firms when array sizes become very large
* Apply targeted `@njit` only after profiling shows sustained benefit

---

### Distribution and research API

* Package the library as a PyPI wheel once the public API stabilises
* Add `Scheduler.to_dict` and `Scheduler.from_dict` for checkpoint and restart workflows
* Provide read-only pandas views for downstream analysis without touching core arrays
* Replace the current hook set with a small plugin event bus (`Plugin.on_event(sim, tag)`)
* Finish NumPy-style docstrings and enable Ruff’s docstring lint rule
* API to capture simulation data at any moment inside a step, without sacrificing performance

---

### Future research extensions

* Implement "growth+" model with R&D and productivity
* Consumption and buffer stock
* Exploration of the parameter space
* Preferential attachment in consumption and the entry mechanism
* Add reinforcement-learning agents for policy search experiments
* Investigate GPU or distributed back-ends once single-node scaling limits are reached
