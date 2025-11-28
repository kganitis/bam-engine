Changelog
=========

All notable changes to BAM Engine are documented here.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

.. note::

   Pre-1.0 releases (0.x.x) may introduce breaking changes between minor versions.

[0.1.1] - 2024-11-14
--------------------

Added
~~~~~

* **CI/CD**: GitHub Actions workflows for automated testing and PyPI publishing

  * Cross-platform testing on Ubuntu, macOS, and Windows
  * Multi-version testing for Python 3.11, 3.12, 3.13
  * Automated PyPI publishing on GitHub releases

Changed
~~~~~~~

* **Breaking**: Dropped Python 3.10 support. Minimum required Python version is now 3.11+

  * Python 3.10 had complex metaclass compatibility issues in decorators that are not worth maintaining
  * Supported versions: Python 3.11, 3.12, 3.13

Fixed
~~~~~

* Windows compatibility in test suite (file handle management, path separators)

[0.1.0] - 2025-11-13
--------------------

This release is feature-complete for the core BAM model but APIs may change in future releases.
Designed for academic research and policy analysis experiments.

Added
~~~~~

**Core ECS Architecture**

* **Role (Component)**: Dataclasses with NumPy arrays for agent state, auto-registration via ``__init_subclass__``
* **Event (System)**: Pure functions wrapped in Event classes, YAML-configurable pipeline execution
* **Relationship**: Many-to-many connections with COO sparse format (e.g., LoanBook for loans)
* **Registry**: Global auto-registration for roles, events, and relationships
* **Pipeline**: YAML-based event ordering with special syntax (repeat, interleave, parameter substitution)
* **Simulation engine**: ``Simulation`` class managing agents, roles, relationships, and event execution

**User-Facing API**

* **Simplified definitions**: ``@role``, ``@event``, ``@relationship`` decorators for easy creation of components
* **Type aliases**: ``Float``, ``Int``, ``Bool``, ``AgentId`` for bam-engine type hints
* **Operations module**: 30+ operations (``ops.add``, ``ops.divide``, etc.) that wrap NumPy operations with safe handling
* **Read methods**: ``sim.get_role()``, ``sim.get_event()``, ``sim.get_relationship()`` with case-insensitive lookup

**BAM Model Implementation**

* **3 agent populations**: Firms, Households, Banks
* **6 agent roles**: Producer, Employer, Borrower, Worker, Consumer, Lender
* **8 event modules**: Planning, labor/credit/goods markets, production, revenue, bankruptcy (39 total events)

**Configuration & Validation**

* **Three-tier configuration**: Package defaults → User YAML → Keyword arguments
* **Centralized validation**: Type checking, range validation, relationship constraints
* **Custom pipeline support**: Load custom event sequences via ``pipeline_path`` parameter
* **Logging configuration**: Global and per-event log levels (DEBUG, INFO, WARNING, ERROR, TRACE)

**Performance & Testing**

* **Vectorized Operations**: NumPy-based computations for efficient large-scale simulations
* **Sparse Relationships**: COO sparse matrix format for memory-efficient relationship storage
* **Benchmarking**: 172 periods/s (small), 96 periods/s (medium), 40 periods/s (large) - 4-9x faster than targets
* **Testing**: Unit tests, integration tests, property-based tests, performance regression tests, covering >99% of codebase
* **Deterministic RNG**: Reproducible simulations with seed control

----

*See the* `GitHub Releases <https://github.com/kganitis/bam-engine/releases>`_ *for complete release history.*
