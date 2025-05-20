# BAM-Engine roadmap

### Immediate tasks

* Implement production components
* Implement production systems
* Unit tests
* Scheduler wiring
* Integration tests
* Raise test coverage
* Scheduler tests

---

### Incremental event implementation

Implement the next events one by one

* Specify the required components and systems
* Wire the event into the scheduler
* Write unit tests that cover edge cases and buffer reuse
* Write an event-level integration test
* Raise the test coverage
* Update the scheduler integration test

Repeat until the full event chain is implemented

---

### Visualization

* Logging and plotting
  * Add lightweight debug logging
  * Add minimal plotting helper
  * Guard with `log.isEnabledFor`

---

### Low-priority housekeeping

* Guard all heavy logging with level checks
* Keep the Ruff rule set minimal (`E, F, W, I` plus docstrings later)
* Introduce pytest-benchmark and fail the suite on measurable performance regressions
* Remove dead lint-ignore comments and finalise design diagrams once the API is frozen

---

### Performance milestone

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
* Replace the current hook set with a small plugin event bus (`Plugin.on_event(sched, tag)`)
* Finish NumPy-style docstrings and enable Ruff’s docstring lint rule

---

### Future research extensions

* Add reinforcement-learning agents for policy search experiments
* Investigate GPU or distributed back-ends once single-node scaling limits are reached


Please also implement the Consumer component and the receive_wage function from the original OOP implementation, 
that will make the consumer receive the wage as income (the income will be increased by the wage amount, not replaced).
There won't be a household component, like there's not a firm component.
The components describe behavior/role, not property. 
Make it separate from the firm_pay_wages system.
Finally, write unit tests for all of the systems of production.py.