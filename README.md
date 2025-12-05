[![Python](https://img.shields.io/pypi/pyversions/bamengine.svg)](https://pypi.org/project/bamengine/)
[![PyPI version](https://img.shields.io/pypi/v/bamengine.svg?color=blue)](https://pypi.org/project/bamengine/)
[![DOI](https://zenodo.org/badge/972128676.svg)](https://doi.org/10.5281/zenodo.17610305)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[![Tests](https://github.com/kganitis/bam-engine/actions/workflows/test.yml/badge.svg)](https://github.com/kganitis/bam-engine/actions/workflows/test.yml)
[![Repo-Review](https://github.com/kganitis/bam-engine/actions/workflows/repo-review.yml/badge.svg)](https://github.com/kganitis/bam-engine/actions/workflows/repo-review.yml)
[![codecov](https://codecov.io/github/kganitis/bam-engine/graph/badge.svg?token=YIG31U3OR3?color=brightgreen)](https://codecov.io/github/kganitis/bam-engine)
[![Benchmarks](https://img.shields.io/badge/benchmarks-asv-brightgreen)](https://kganitis.github.io/bam-engine/)

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/kganitis/bam-engine/main.svg)](https://results.pre-commit.ci/latest/github/kganitis/bam-engine/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type Checked](https://img.shields.io/badge/type%20checked-mypy-black)](http://mypy-lang.org/)

# BAM Engine

**Modular Python Framework for the Agent-based BAM Model**

BAM Engine is a high-performance Python implementation of the BAM model from *Macroeconomics from the Bottom-up* (Delli Gatti et al., 2011, Chapter 3). Developed as part of MSc thesis research at the University of Piraeus, it provides a modular, extensible agent-based macroeconomic simulation framework built on ECS (Entity-Component-System) architecture with fully vectorized NumPy operations.

Documentation: [https://bam-engine.readthedocs.io](https://bam-engine.readthedocs.io)

> **Note**: This release is feature-complete for the core BAM model but APIs may change in future releases before v1.0.0.

## Features

- **Complete BAM Model Implementation**: Full BAM model with firms, households, and banks interacting across labor, credit, and goods markets
- **Emergent Macroeconomic Dynamics**: Reproduces all simulation results from the reference BAM model, including business cycles, unemployment dynamics, and inflation patterns
- **Research-Ready Output**: Collect time series data and export to pandas DataFrames for econometric analysis
- **Reproducible Simulations**: Deterministic execution with seed control for exact replication of results
- **YAML Configuration**: Modify model parameters without touching code
- **Extensible**: Add custom agent variables and economic events

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

BAM Engine uses an ECS (Entity-Component-System) architecture: agents are lightweight entities with immutable IDs, state lives in Role components stored as NumPy arrays, and behavior is defined by Event systems executed via a YAML-configurable pipeline. This design separates data from logic, enabling high performance through vectorized operations while maintaining extensibility for custom roles, events, and relationships.

## Changelog

See the [changelog](https://bam-engine.readthedocs.io/en/latest/changelog.html) for a history of notable changes to bam-engine.

## Development

### Important Links

- Official source code repo: [https://github.com/kganitis/bam-engine](https://github.com/kganitis/bam-engine)
- Download releases: [https://pypi.org/project/bamengine/](https://pypi.org/project/bamengine/)
- Issue Tracker: [https://github.com/kganitis/bam-engine/issues](https://github.com/kganitis/bam-engine/issues)

### Source code

You can check the latest sources with the command:

```bash
git clone https://github.com/kganitis/bam-engine.git
```

### Contributing

This project was developed as part of the final thesis for MSc in Informatics at the University of Piraeus, Greece. External contributions are not accepted during thesis work.

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
1. **The original BAM model** - Delli Gatti, D., Desiderio, S., Gaffeo, E., Cirillo, P., & Gallegati, M. (2011). *Macroeconomics from the Bottom-up*. Springer. DOI: [10.1007/978-88-470-1971-3](https://doi.org/10.1007/978-88-470-1971-3)

## License

MIT License - see [LICENSE](LICENSE) for details.
