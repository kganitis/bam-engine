:orphan:

:orphan:

.. _examples_gallery:

Examples Gallery
================

This gallery contains examples demonstrating various features and use cases of BAM Engine.

All examples assume BAM Engine is installed. You can install it via:

.. code-block:: bash

   pip install bamengine
   pip install bamengine[pandas]  # For DataFrame export support



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how users can define custom roles and events using the simplified decorator syntax. NO INHERITANCE NEEDED!">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_decorator_usage_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_decorator_usage.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Example demonstrating the @role and @event decorators.</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/decorator_usage

Advanced Examples
=================

Examples showing advanced features like custom events, roles, relationships,
and pipeline configuration.

These examples demonstrate:

* Creating custom events to extend the model
* Defining custom roles to add new agent behaviors
* Creating many-to-many relationships between roles with edge data
* Modifying the event execution pipeline
* Running parameter sweeps and ensemble simulations
* Advanced data collection and analysis techniques



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Basic Examples
==============

Start here if you're new to BAM Engine. These examples demonstrate fundamental
concepts and basic usage patterns.

Examples in this section:

1. **Hello World**: The simplest possible BAM Engine example - initialize, run, and visualize
2. **Configuration**: Customize simulation parameters using keyword arguments
3. **Baseline Scenario**: Reproduce the baseline scenario from section 3.9.1 of the original BAM book

These examples will teach you:

* How to initialize and run a basic simulation
* How to configure simulation parameters via keyword arguments
* How to collect and visualize simulation results
* How to compare different economic scenarios
* How to reproduce scenarios from the BAM literature



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This is the simplest possible BAM Engine example. It shows how to initialize a simulation, run it for a few periods, and display basic results.">

.. only:: html

  .. image:: /auto_examples/basic/images/thumb/sphx_glr_example_hello_world_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_basic_example_hello_world.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hello BAM Engine</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to customize simulation parameters using keyword arguments. You can override any default parameter to explore different economic scenarios.">

.. only:: html

  .. image:: /auto_examples/basic/images/thumb/sphx_glr_example_configuration_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_basic_example_configuration.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Configuring Your Simulation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example reproduces the baseline scenario from section 3.9.1 of the original BAM model book (Delli Gatti et al., 2011). This scenario demonstrates the fundamental dynamics of the model with standard parameter values.">

.. only:: html

  .. image:: /auto_examples/basic/images/thumb/sphx_glr_example_baseline_scenario_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_basic_example_baseline_scenario.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BAM Baseline Scenario (3.9.1)</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Research Examples
=================

Examples demonstrating extensions from the original BAM model
(Delli Gatti et al., 2011).

These examples reproduce research extensions from the BAM book:

* **Growth+ Model**: R&D investment and endogenous productivity growth
* **Consumption and Buffer Shock**: Household consumption shocks and savings buffer dynamics
* **Parameter Space Exploration**: Systematic exploration of model sensitivity to key parameters
* **Preferential Attachment**: Preferential attachment in consumption and firm entry mechanisms

Each example includes detailed explanations of the economic mechanisms and
comparison with results from the original literature.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /auto_examples/advanced/index.rst
   /auto_examples/basic/index.rst
   /auto_examples/research/index.rst



.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
