Examples
========

Example scripts demonstrating BAM Engine usage patterns.

The examples are organized into three sections:

- **Basic**: Getting started tutorials for new users
- **Advanced**: Custom components and pipeline modifications
- **Extensions**: Scenarios from the BAM literature

These examples are simplified demonstrations focused on teaching core concepts.
For detailed analysis with validation bounds and statistical annotations, see
the ``validation`` package:

.. code-block:: bash

   # Baseline scenario with full validation
   python -m validation.scenarios.baseline

   # Growth+ scenario with full validation
   python -m validation.scenarios.growth_plus
