Robustness CLI
==============

The robustness analysis can be run from the command line:

.. code-block:: bash

   python -m validation.robustness [OPTIONS]


Usage Examples
--------------

.. code-block:: bash

   # Full analysis (internal validity + sensitivity + structural)
   python -m validation.robustness

   # Individual parts
   python -m validation.robustness --internal-only
   python -m validation.robustness --sensitivity-only
   python -m validation.robustness --structural-only

   # Individual structural experiments
   python -m validation.robustness --pa-experiment
   python -m validation.robustness --entry-experiment

   # Specific sensitivity experiments
   python -m validation.robustness --sensitivity-only --experiments credit_market,contract_length

   # Custom settings
   python -m validation.robustness --seeds 10 --periods 500 --workers 4 --no-plots

   # Use baseline model instead of Growth+ (Growth+ is the default)
   python -m validation.robustness --no-growth-plus


Options
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``--internal-only``
     - Run internal validity analysis only
   * - ``--sensitivity-only``
     - Run sensitivity analysis only
   * - ``--structural-only``
     - Run both structural experiments only
   * - ``--pa-experiment``
     - Run PA (preferential attachment) experiment only
   * - ``--entry-experiment``
     - Run entry neutrality experiment only
   * - ``--experiments NAMES``
     - Comma-separated list of sensitivity experiments
   * - ``--seeds N``
     - Number of random seeds (default: 20)
   * - ``--periods N``
     - Simulation periods (default: 1000)
   * - ``--workers N``
     - Parallel workers (default: 10)
   * - ``--no-plots``
     - Skip plot generation
   * - ``--no-baseline``
     - Skip baseline comparison in PA experiment
   * - ``--no-growth-plus``
     - Use baseline model instead of Growth+


Performance
-----------

Typical execution times (10-core machine):

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15

   * - Configuration
     - Seeds
     - Periods
     - Time
   * - Internal validity
     - 20
     - 1000
     - ~2 min
   * - Single sensitivity experiment
     - 20x5
     - 1000
     - ~10 min
   * - Full sensitivity (5 experiments)
     - 20x28
     - 1000
     - ~50 min
   * - Economy size experiment (10x)
     - 20x7
     - 1000
     - ~30 min
