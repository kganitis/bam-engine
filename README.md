# BAM-Engine (Work in Progress)

A research prototype for my thesis, implementing the BAM model from *Macroeconomics from the Bottom-Up* (Delli Gatti et al., 2011).

### Architecture

* Entity-Component style: all state stored in NumPy arrays
* Event-driven Scheduler: each period runs Planning → Labour market → Credit market → …
* Zero runtime allocation in hot loops

### Technology

* pure Python + NumPy
* pytest for unit and integration tests
* hooks for custom shocks or policy experiments

### Status

* 4 of 8 events implemented and tested:
  * Planning
  * Labour market
  * Credit market
  * Production
  * Goods market


* Remaining events:
  * Revenues
  * Bankruptcy
  * Entry of new agents

### Next Steps

* implement and test the remaining 3 events
* cross-check results with the BAM paper
* profile and optimize critical loops
* explore parallel execution options
* package as a lightweight library for reproducible research
