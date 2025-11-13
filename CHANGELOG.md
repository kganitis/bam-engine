# Changelog

All notable changes to BAM Engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha] - 2025-11-12

**⚠️ Alpha Status**: This release is feature-complete for the core BAM model but APIs may change
in future releases. Suitable for research and experimentation, but not recommended for
production use.

### Added

#### Core ECS Architecture

- **Role (Component)**: Dataclasses with NumPy arrays for agent state, auto-registration via `__init_subclass__`
- **Event (System)**: Pure functions wrapped in Event classes, YAML-configurable pipeline execution
- **Relationship**: Many-to-many connections with COO sparse format (e.g., LoanBook for loans)
- **Registry**: Global auto-registration for roles, events, and relationships
- **Pipeline**: YAML-based event ordering with special syntax (repeat, interleave, parameter substitution)
- **Simulation engine**: `Simulation` class managing agents, roles, relationships, and event execution

#### User-Facing API

- **Simplified definitions**: `@role`, `@event`, `@relationship` decorators for easy creation of components
- **Type aliases**: `Float`, `Int`, `Bool`, `AgentId` for bam-engine type hints
- **Operations module**: 30+ operations (`ops.add`, `ops.divide`, etc.) that wrap NumPy operations with safe handling
- **Read methods**: `sim.get_role()`, `sim.get_event()`, `sim.get_relationship()` with case-insensitive lookup

#### BAM Model Implementation

- **3 agent populations**: Firms, Households, Banks
- **6 agent roles**: Producer, Employer, Borrower, Worker, Consumer, Lender
- **8 event modules**: Planning, labor/credit/goods markets, production, revenue, bankruptcy (39 total events)
- **Deterministic RNG**: Reproducible simulations with seed control

#### Configuration & Validation

- **Three-tier configuration**: Package defaults → User YAML → Keyword arguments
- **Centralized validation**: Type checking, range validation, relationship constraints
- **Custom pipeline support**: Load custom event sequences via `pipeline_path` parameter
- **Logging configuration**: Global and per-event log levels (DEBUG, INFO, WARNING, ERROR, TRACE)

#### Performance & Testing

- **Vectorized Operations**: NumPy-based computations for efficient large-scale simulations
- **Sparse Relationships**: COO sparse matrix format for memory-efficient relationship storage
- **Benchmarking**: 172 periods/s (small), 96 periods/s (medium), 40 periods/s (large) - 4-9x faster than targets
- **Testing**: Unit tests, integration tests, property-based tests, performance regression tests, covering >99% of codebase

---

*Note: This project follows [Semantic Versioning](https://semver.org/). Pre-1.0 releases (0.x.x) may introduce breaking changes between minor versions.*
