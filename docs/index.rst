:html_theme.sidebar_secondary.remove:

.. BAM Engine documentation master file

==========
BAM Engine
==========

**Agent-based macroeconomic simulation in Python**

A Python implementation of the BAM (Bottom-Up Adaptive Macroeconomics) model
by Delli Gatti et al. (2011). Simulate households, firms, and banks
interacting across labor, credit, and goods markets; macroeconomic dynamics
emerge from individual agent decisions.

For researchers, students, and practitioners in computational economics.

.. code-block:: bash

   pip install bamengine

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: quickstart
      :link-type: doc
      :class-card: sd-shadow-sm

      Install the package, run your first simulation, and
      explore the results in under five minutes.

   .. grid-item-card:: User Guide
      :link: user_guide/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Learn how the BAM model's agents, markets, and simulation
      pipeline work, and how to configure and extend them.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Complete reference for all components, events,
      operations, and configuration options.

   .. grid-item-card:: Examples
      :link: auto_examples/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Runnable examples covering basic usage, advanced
      customization, and model extensions.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Extensions
      :link: extensions/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Extend the base model with R&D / Growth+,
      buffer-stock consumption, and taxation.

   .. grid-item-card:: Validation
      :link: validation/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Reproduce the book's scenarios and verify model
      behavior with robustness and sensitivity analysis.

   .. grid-item-card:: Calibration
      :link: calibration/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Find optimal parameters through screening,
      grid search, and stability testing.

.. grid:: 2 3 3 6
   :gutter: 2

   .. grid-item::

      :doc:`release_history`

   .. grid-item::

      :doc:`development/index`

   .. grid-item::

      :doc:`glossary`

   .. grid-item::

      `About <https://bamengine.org/about/>`__

   .. grid-item::

      `Community <https://bamengine.org/community/>`__

   .. grid-item::

      `Roadmap <https://bamengine.org/roadmap/>`__

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   installation
   quickstart
   user_guide/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API

   api/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Extensions

   extensions/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Validation

   validation/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Calibration

   calibration/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Release History

   release_history

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Development

   development/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Project

   glossary
