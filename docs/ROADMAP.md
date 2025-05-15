# BAM-Engine roadmap

## Immediate tasks

* Add a credit-market event integration test
* Raise the test coverage

---

## Incremental event implementation

Implement the next events one by one

* Specify the required components and systems
* Wire the event into the scheduler
* Write unit tests that cover edge cases and buffer reuse
* Write an event-level integration test
* Raise the test coverage

Repeat until the full event chain is implemented

---

## Visualization

* Logging and plotting
  * Add lightweight debug logging
  * Add minimal plotting helper
  * Guard with `log.isEnabledFor`

---

## Low-priority housekeeping

* Guard all heavy logging with level checks
* Keep the Ruff rule set minimal (`E, F, W, I` plus docstrings later)
* Introduce pytest-benchmark and fail the suite on measurable performance regressions
* Remove dead lint-ignore comments and finalise design diagrams once the API is frozen

---

## Performance milestone

* Profile representative sizes to establish baselines
* Remove remaining temporary allocations and other obvious hotspots
  * e.g. reuse permanent scratch buffers in `workers_decide_firms_to_apply` and `firms_prepare_loan_applications`
* Enable threaded NumPy or process-level sharding of firms when array sizes become very large
* Apply targeted `@njit` only after profiling shows sustained benefit

---

## Distribution and research API

* Package the library as a PyPI wheel once the public API stabilises
* Add `Scheduler.to_dict` and `Scheduler.from_dict` for checkpoint and restart workflows
* Provide read-only pandas views for downstream analysis without touching core arrays
* Replace the current hook set with a small plugin event bus (`Plugin.on_event(sched, tag)`)
* Finish NumPy-style docstrings and enable Ruff’s docstring lint rule

---

## Future research extensions

* Add reinforcement-learning agents for policy search experiments
* Investigate GPU or distributed back-ends once single-node scaling limits are reached
