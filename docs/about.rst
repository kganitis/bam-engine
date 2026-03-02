About
=====


History
-------

BAM Engine began as part of an MSc thesis at the University of Piraeus, Greece,
with the goal of creating a modern, extensible Python implementation of the BAM
(Bottom-Up Adaptive Macroeconomics) model from *Macroeconomics from the Bottom-up*
(Delli Gatti et al., 2011).

Development started in late 2024. The initial release (v0.1.0) delivered the
complete core BAM model with an Entity-Component-System architecture, vectorized
NumPy operations, and YAML-configurable event pipelines. Subsequent releases
added validation and calibration frameworks (v0.2.0), the buffer-stock consumption
extension (v0.3.0), and robustness analysis with a rebuilt calibration pipeline
(v0.4.0).


Academic Context
----------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Thesis title
     - [THESIS_TITLE]
   * - Author
     - Kostas Ganitis
   * - Supervisor
     - [SUPERVISOR_NAME]
   * - Department
     - Department of Informatics, University of Piraeus, Greece
   * - Degree
     - MSc in Informatics


Citing BAM Engine
-----------------

If you use BAM Engine in your research, please cite both the software and the
original BAM model.

**BibTeX (software):**

.. code-block:: bibtex

   @software{ganitis2026bamengine,
     title     = {{BAM Engine}: Modular Python Framework for the Agent-based
                  {BAM} Model},
     author    = {Ganitis, Konstantinos},
     year      = {2026},
     url       = {https://github.com/kganitis/bam-engine},
     doi       = {10.5281/zenodo.17610305},
     license   = {MIT}
   }

**BibTeX (original BAM model):**

.. code-block:: bibtex

   @book{delligatti2011macroeconomics,
     title     = {Macroeconomics from the Bottom-up},
     author    = {Delli Gatti, Domenico and Desiderio, Saul and Gaffeo, Edoardo
                  and Cirillo, Pasquale and Gallegati, Mauro},
     year      = {2011},
     publisher = {Springer Milano},
     series    = {New Economic Windows},
     doi       = {10.1007/978-88-470-1971-3},
     isbn      = {978-88-470-1971-3}
   }

**Plain text:**

   Ganitis, K. (2026). *BAM Engine: Modular Python Framework for the Agent-based
   BAM Model* [Computer software]. https://doi.org/10.5281/zenodo.17610305

.. image:: https://zenodo.org/badge/972128676.svg
   :target: https://doi.org/10.5281/zenodo.17610305
   :alt: DOI


License
-------

BAM Engine is distributed under the `MIT License <https://opensource.org/licenses/MIT>`_:

   Copyright (c) 2025 Konstantinos Ganitis

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

See the full `LICENSE <https://github.com/kganitis/bam-engine/blob/main/LICENSE>`_
file for the complete text.


Acknowledgments
---------------

BAM Engine would not exist without the foundational work of the original BAM
model authors:

- Domenico Delli Gatti
- Saul Desiderio
- Edoardo Gaffeo
- Pasquale Cirillo
- Mauro Gallegati

The project was developed at the **Department of Informatics, University of
Piraeus, Greece**.

BAM Engine is built on the Python scientific computing ecosystem, with particular
reliance on `NumPy <https://numpy.org/>`_, `SciPy <https://scipy.org/>`_,
`pandas <https://pandas.pydata.org/>`_, and `Matplotlib <https://matplotlib.org/>`_.
Documentation is powered by `Sphinx <https://www.sphinx-doc.org/>`_ with the
`PyData theme <https://pydata-sphinx-theme.readthedocs.io/>`_. Testing uses
`pytest <https://docs.pytest.org/>`_ and `Hypothesis <https://hypothesis.readthedocs.io/>`_.
