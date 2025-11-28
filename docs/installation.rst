.. _install:

=======
Install
=======

There are different ways to install BAM Engine:

* :ref:`Install the latest official release <install_latest_release>`. This
  is the best approach for most users. It will provide a stable version
  and pre-built packages are available for all platforms.

* :ref:`Build the package from source <install_from_source>`. This is best
  for users who want the latest features or wish to contribute to the project.


Dependencies
============

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Dependency
     - Version
     - Purpose
   * - Python
     - >= 3.11
     - Core language runtime
   * - numpy
     - >= 1.26
     - Vectorized array operations
   * - PyYAML
     - >= 6.0.2
     - Configuration file parsing
   * - pandas *(optional)*
     - >= 2.0
     - DataFrame export for results


.. _install_latest_release:

Installing the latest release
=============================

BAM Engine is available on `PyPI <https://pypi.org/project/bamengine/>`_.
We recommend using a `virtual environment
<https://docs.python.org/3/tutorial/venv.html>`_ to avoid conflicts with
other packages.

.. code-block:: bash

   pip install -U bamengine

To enable DataFrame export for simulation results, install with pandas support:

.. code-block:: bash

   pip install -U bamengine[pandas]

To verify your installation:

.. code-block:: bash

   python -m pip show bamengine         # show version and location
   python -c "import bamengine; print(bamengine.__version__)"


.. _install_from_source:

Building from source
====================

Building from source is required to work on a contribution or to run the
latest development version.

.. code-block:: bash

   git clone https://github.com/kganitis/bam-engine.git
   cd bam-engine
   pip install -e ".[dev]"

This installs the package in editable mode with all development dependencies
(testing, linting, documentation).

To verify the installation, run the test suite:

.. code-block:: bash

   pytest
