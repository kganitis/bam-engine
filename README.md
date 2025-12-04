[![Python](https://img.shields.io/pypi/pyversions/bamengine.svg)](https://pypi.org/project/bamengine/)
[![PyPI version](https://img.shields.io/pypi/v/bamengine.svg?color=blue)](https://pypi.org/project/bamengine/)
[![DOI](https://zenodo.org/badge/972128676.svg)](https://doi.org/10.5281/zenodo.17610305)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[![Tests](https://github.com/kganitis/bam-engine/actions/workflows/test.yml/badge.svg)](https://github.com/kganitis/bam-engine/actions/workflows/test.yml)
[![codecov](https://codecov.io/github/kganitis/bam-engine/graph/badge.svg?token=YIG31U3OR3?color=brightgreen)](https://codecov.io/github/kganitis/bam-engine)
[![Benchmarks](https://img.shields.io/badge/benchmarks-asv-brightgreen)](https://kganitis.github.io/bam-engine/)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: Ruff](https://img.shields.io/badge/linter-ruff-black)](https://github.com/astral-sh/ruff)
[![Type Checked](https://img.shields.io/badge/type%20checked-mypy-black)](http://mypy-lang.org/)

# BAM Engine

**Modular Python Framework for the Agent-based BAM Model**

BAM Engine is a high-performance Python implementation of the BAM model from *Macroeconomics from the Bottom-up* (Delli Gatti et al., 2011, Chapter 3). It provides a modular, extensible agent-based macroeconomic simulation framework built on ECS (Entity-Component-System) architecture with fully vectorized NumPy operations.

> **Note**: This release is feature-complete for the core BAM model but APIs may change in future releases before v1.0.0.

## Features

- **BAM Model**: 3 agent types (firms, households, banks) interacting in 3 markets (labor, credit, consumption goods)
- **High Performance**: Vectorized NumPy operations, 500+ firms in seconds per 1000 periods
- **ECS Architecture**: Modular design separating agent state (Roles) from behavior (Events)
- **Extensible API**: Easy custom roles, events, relationships, and YAML-configurable pipelines
- **Reproducibility**: Deterministic simulations with seedable RNG
- **Well Tested**: 95%+ coverage with unit, integration, property-based, and performance tests

## Quick Start

### Installation

```bash
pip install bamengine
```

**Requirements**: Python 3.11+. NumPy and PyYAML are installed automatically.

### Basic Usage

```python
import bamengine as bam

# Initialize and run simulation
sim = bam.Simulation.init(n_firms=100, n_households=500, seed=42)
results = sim.run(n_periods=100)

# Export to pandas DataFrame
df = results.to_dataframe()
```

## Architecture

BAM Engine uses an **ECS (Entity-Component-System)** architecture:

| Concept | Description |
|---------|-------------|
| **Agents** | Lightweight entities with immutable IDs and types (FIRM, HOUSEHOLD, BANK) |
| **Roles** | Dataclasses storing agent state as NumPy arrays (Producer, Worker, Lender, etc.) |
| **Events** | Pure functions operating on roles, executed in pipeline order |
| **Relationships** | Many-to-many connections with sparse COO format (e.g., LoanBook) |
| **Pipeline** | YAML-configurable event execution with repeat/interleave syntax |

### Agent Roles

- **Firms**: Producer + Employer + Borrower
- **Households**: Worker + Consumer
- **Banks**: Lender

### Event Pipeline

Each period executes 39 events across 8 phases: Planning, Labor Market, Credit Market, Production, Goods Market, Revenue, Bankruptcy, and Entry.

## Documentation

<!-- Full documentation: [https://bam-engine.readthedocs.io](https://bam-engine.readthedocs.io) -->

Documentation is under development. See the [examples/](examples/) directory for usage patterns.

## Changelog

See the [changelog](docs/changelog.rst) for a history of notable changes.

## Development

### Source code

You can check the latest sources with the command:

```bash
git clone https://github.com/kganitis/bam-engine.git
```

### Contributing

This project was developed as part of the final thesis for MSc in Informatics at the University of Piraeus, Athens, Greece. External contributions are not accepted during thesis work.

For bug reports and feature requests, please open an issue on the [issue tracker](https://github.com/kganitis/bam-engine/issues).

### Testing

After installation, you can launch the test suite:

```bash
pip install -e ".[dev]"
pytest
```

See the [development guide](docs/development.rst) for more commands including linting, type checking, and benchmarking.

## Citation

If you use BAM Engine in your research, please cite:

1. **This software** - Use [`CITATION.cff`](CITATION.cff) or GitHub's "Cite this repository"
2. **The original BAM model** - Delli Gatti, D., Desiderio, S., Gaffeo, E., Cirillo, P., & Gallegati, M. (2011). *Macroeconomics from the Bottom-up*. Springer. DOI: [10.1007/978-88-470-1971-3](https://doi.org/10.1007/978-88-470-1971-3)

## License

MIT License - see [LICENSE](LICENSE) for details.
