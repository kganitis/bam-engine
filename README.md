# BAM-Engine (Work in Progress)

A research prototype for my thesis, implementing the BAM model from *Macroeconomics from the Bottom-Up* (Delli Gatti et al., 2011).

### Architecture

* Entity-Component style: all state stored in NumPy arrays
* Event-driven Scheduler: each period runs Planning → Labour market → Credit market → …
* Zero runtime allocation in hot loops

### Technology

* Pure Python + NumPy
* pytest for unit and integration tests
* Hooks for custom shocks or policy experiments

### Status

* 6 of 8 events implemented and tested:
  * Planning
  * Labour market
  * Credit market
  * Production
  * Goods market
  * Revenues

* Remaining events:
  * Bankruptcy
  * Entry of new agents

### Next Steps

* Implement and test the remaining 2 events
* Visualize and cross-check results with the BAM paper
* Refactor architecture to true ECS
* Profile and optimize critical loops
* Explore parallel execution options
* Package as a lightweight library for reproducible research
* Future extensions with RL agents
