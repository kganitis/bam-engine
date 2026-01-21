Extension Examples
==================

Examples demonstrating extensions from the original BAM model
(Delli Gatti et al., 2011).

These examples reproduce research extensions from the BAM book:

* **Growth+ Model**: R&D investment and endogenous productivity growth.
  Demonstrates how to create custom roles and events to extend BAM Engine.
* **Consumption and Buffer Shock**: Household consumption shocks and savings buffer dynamics
* **Parameter Space Exploration**: Systematic exploration of model sensitivity to key parameters
* **Preferential Attachment**: Preferential attachment in consumption and firm entry mechanisms

Each example includes detailed explanations of the economic mechanisms and
comparison with results from the original literature.

Note: The Growth+ example is a simplified demonstration focused on teaching
how to create custom extensions. For full validation with target bounds and
statistical annotations, run:

.. code-block:: bash

   python -m validation.scenarios.growth_plus
