Robustness Analysis
===================

*Section 3.10 of Delli Gatti et al. (2011)*

The robustness package tests whether model results are stable across random
seeds, parameter variations, and structural mechanism changes. It consists
of three parts:

1. **Internal Validity** (3.10.1, Part 1) — Multiple random seeds verify
   results are robust to stochastic variation
2. **Sensitivity Analysis** (3.10.1, Part 2) — Parameter sweeps assess how
   input changes alter outputs
3. **Structural Experiments** (3.10.2) — Mechanism toggles test model
   properties (consumer loyalty, entry neutrality)


Quick Start
-----------

**CLI:**

.. code-block:: bash

   # Full analysis (internal validity + sensitivity + structural)
   python -m validation.robustness

   # Individual parts
   python -m validation.robustness --internal-only
   python -m validation.robustness --sensitivity-only
   python -m validation.robustness --structural-only

**Python API:**

.. code-block:: python

   from validation.robustness import (
       run_internal_validity,
       run_sensitivity_analysis,
       run_pa_experiment,
       run_entry_experiment,
   )

   # Internal validity
   iv = run_internal_validity(n_seeds=20, n_periods=1000)

   # Sensitivity analysis
   sa = run_sensitivity_analysis(experiments=["credit_market"])

   # Structural experiments
   pa = run_pa_experiment(n_seeds=20)
   entry = run_entry_experiment(n_seeds=20)

**Growth+ support:**

.. code-block:: python

   from validation.robustness import setup_growth_plus, GROWTH_PLUS_COLLECT_CONFIG

   iv = run_internal_validity(
       setup_hook=setup_growth_plus,
       collect_config=GROWTH_PLUS_COLLECT_CONFIG,
   )


Module Structure
----------------

::

   validation/robustness/
   ├── __init__.py              # Public API exports
   ├── __main__.py              # CLI entry point
   ├── stats.py                 # Statistical tools (HP filter, AR, IRF)
   ├── experiments.py           # Experiment definitions
   ├── internal_validity.py     # Multi-seed analysis pipeline
   ├── sensitivity.py           # Parameter sweep pipeline
   ├── structural.py            # Structural experiment orchestrators
   ├── viz.py                   # Visualizations (Figure 3.9, 3.10)
   └── reporting.py             # Text report formatting

.. toctree::
   :maxdepth: 2
   :hidden:

   internal_validity
   sensitivity
   structural
   stats
   viz
   cli
