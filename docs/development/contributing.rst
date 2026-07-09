Contributing & Docstrings
==========================

Contributing
------------

Contributions are welcome: bug fixes, documentation, tests, performance work,
and new extensions. Small, self-contained fixes can go straight to a pull
request. For larger or model-behavior changes (anything that could change the
public API or shift results against the validation targets), please open an
issue on `GitHub <https://github.com/kganitis/bam-engine/issues>`_ first to
align on the approach.

See `CONTRIBUTING.md
<https://github.com/kganitis/bam-engine/blob/main/CONTRIBUTING.md>`_ for the
full guidelines, quality bar, and code of conduct.


Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

1. Fork the repository
2. Create a feature branch off ``main`` (``git checkout -b feature/my-feature``)
3. Make changes and add tests
4. Run the full test suite (``pytest``); keep coverage at 99%
5. Run code quality checks (``ruff format . && ruff check --fix . && mypy``)
6. Ensure all randomness goes through ``sim.rng`` for reproducibility
7. Commit with a descriptive message
8. Push and open a pull request against ``main``, linking the related issue


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
