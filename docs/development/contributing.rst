Contributing & Docstrings
==========================

Contributing
------------

.. note::

   This project is currently not accepting external contributions as it is part
   of ongoing thesis work. Once the thesis is submitted, contribution guidelines
   will be published.

For bug reports and feature requests, please open an issue on
`GitHub <https://github.com/kganitis/bam-engine/issues>`_.


Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

When contributions are open:

1. Fork the repository
2. Create a feature branch (``git checkout -b feature/my-feature``)
3. Make changes and add tests
4. Run the full test suite (``pytest``)
5. Run code quality checks (``ruff format . && ruff check --fix . && mypy``)
6. Commit with a descriptive message
7. Push and open a pull request


Docstring Style
~~~~~~~~~~~~~~~~

BAM Engine uses NumPy-style docstrings. See the
`NumPy docstring guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_
for the format specification.

Example:

.. code-block:: python

   def my_function(param1: float, param2: int) -> bool:
       """
       Short description of the function.

       Longer description if needed.

       Parameters
       ----------
       param1 : float
           Description of param1.
       param2 : int
           Description of param2.

       Returns
       -------
       bool
           Description of return value.

       Examples
       --------
       >>> my_function(1.0, 2)
       True
       """
       pass


Release Checklist
-----------------

1. Update version in ``src/bamengine/__init__.py``
2. Update ``release_history.rst``
3. Run full test suite: ``pytest``
4. Run code quality checks: ``ruff format . && ruff check --fix . && mypy``
5. Build docs: ``cd docs && sphinx-build -b html . _build/html``
6. Tag release: ``git tag vX.Y.Z``
7. Build and publish: ``python -m build && twine upload dist/*``
