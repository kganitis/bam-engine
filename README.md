<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/kganitis/bam-engine/main/docs/_static/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/kganitis/bam-engine/main/docs/_static/logo.svg">
    <img alt="BAM Engine" src="https://raw.githubusercontent.com/kganitis/bam-engine/main/docs/_static/logo.svg" width="280">
  </picture>
</p>

<p align="center">
  <strong>A Modular Python Framework for the BAM Agent-Based Macroeconomic Model</strong>
</p>

<p align="center">
  A Python implementation of the BAM (Bottom-Up Adaptive Macroeconomics) model
  from <a href="https://doi.org/10.1007/978-88-470-1971-3"><em>Macroeconomics from the Bottom-up</em></a>
  (Delli Gatti et al., 2011). Instead of assuming markets reach equilibrium,
  BAM Engine simulates individual workers, firms, and banks making decisions
  and interacting across labor, credit, and goods markets, letting
  macroeconomic patterns emerge from the bottom up.
</p>

<p align="center">
  <a href="https://bamengine.org"><strong>bamengine.org</strong></a>
  &bull;
  <a href="https://bam-engine.readthedocs.io"><strong>Documentation</strong></a>
  &bull;
  <a href="https://bam-engine.readthedocs.io/en/latest/quickstart.html"><strong>Getting Started</strong></a>
  &bull;
  <a href="https://bam-engine.readthedocs.io/en/latest/auto_examples/index.html"><strong>Examples</strong></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/bamengine/"><img src="https://img.shields.io/pypi/pyversions/bamengine.svg" alt="Python"></a>
  <a href="https://pypi.org/project/bamengine/"><img src="https://img.shields.io/pypi/v/bamengine.svg?color=blue" alt="PyPI version"></a>
  <a href="https://doi.org/10.5281/zenodo.17610305"><img src="https://zenodo.org/badge/972128676.svg" alt="DOI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
</p>

<p align="center">
  <a href="https://github.com/kganitis/bam-engine/actions/workflows/test.yml"><img src="https://github.com/kganitis/bam-engine/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/kganitis/bam-engine/actions/workflows/repo-review.yml"><img src="https://github.com/kganitis/bam-engine/actions/workflows/repo-review.yml/badge.svg" alt="Repo-Review"></a>
  <a href="https://codecov.io/github/kganitis/bam-engine"><img src="https://codecov.io/github/kganitis/bam-engine/graph/badge.svg?token=YIG31U3OR3?color=brightgreen" alt="codecov"></a>
  <a href="https://kganitis.github.io/bam-engine/"><img src="https://img.shields.io/badge/benchmarks-asv-brightgreen" alt="Benchmarks"></a>
</p>

<p align="center">
  <a href="https://results.pre-commit.ci/latest/github/kganitis/bam-engine/main"><img src="https://results.pre-commit.ci/badge/github/kganitis/bam-engine/main.svg" alt="pre-commit.ci status"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="http://mypy-lang.org/"><img src="https://img.shields.io/badge/type%20checked-mypy-black" alt="Type Checked"></a>
</p>

<p align="center">
  <a href="https://bamengine.org/benchmarks/validation/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://bamengine.org/images/validation-card-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="https://bamengine.org/images/validation-card-light.svg">
      <img alt="Model Validation" src="https://bamengine.org/images/validation-card-light.svg" width="476">
    </picture>
  </a>
</p>

> **Note**: This release is feature-complete for the core BAM model but APIs may change in future releases before v1.0.0.

## Quick Start

```bash
pip install bamengine
```

**Requirements**: Python 3.11+. NumPy and PyYAML are installed automatically.

```python
import bamengine as bam

# Initialize and run simulation
sim = bam.Simulation.init(n_firms=100, n_households=500, seed=42)
results = sim.run(n_periods=100)

# Access results
results["Economy.inflation"]  # 1D array (n_periods,)
results.Producer.price  # 2D array (n_periods, n_firms)

# Export to pandas DataFrame
df = results.to_dataframe()

# Add extensions with one call
from extensions.rnd import RND

sim = bam.Simulation.init(seed=42)
sim.use(RND)
results = sim.run(n_periods=1000)
```

See the [Getting Started guide](https://bam-engine.readthedocs.io/en/latest/quickstart.html) for a complete walkthrough.

## Features

- **Complete BAM Model:** Full Chapter 3 implementation: firms, households, and banks interacting across labor, credit, and goods markets
- **ECS Architecture:** Entity-Component-System design separates data (Roles) from behavior (Events) for clean extensibility
- **Vectorized Performance:** All agent operations use NumPy arrays; no Python loops over agents
- **Built-in Extensions:** R&D / Growth+, buffer-stock consumption, and taxation modules
- **Validation Framework:** Three scenario validators with scoring and robustness analysis
- **Calibration Pipeline:** Morris screening, grid search, and tiered stability testing
- **Easy Configuration:** All parameters configurable without code changes via YAML files

## Architecture

BAM Engine uses an ECS (Entity-Component-System) architecture: agents are lightweight entities, state lives in Role components stored as NumPy arrays, and behavior is defined by Event systems executed via a YAML-configurable pipeline. Custom roles, events, and relationships can be added without modifying core code.

See the [User Guide](https://bam-engine.readthedocs.io/en/latest/user_guide/index.html) for a full walkthrough of the model and its architecture.

## Documentation

Full documentation is available at [bam-engine.readthedocs.io](https://bam-engine.readthedocs.io).

| Section                                                                           | Description                                                                   |
| --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [Getting Started](https://bam-engine.readthedocs.io/en/latest/quickstart.html)    | Installation, first simulation, data collection                               |
| [User Guide](https://bam-engine.readthedocs.io/en/latest/user_guide/index.html)   | Model overview, configuration, custom roles/events, pipelines, best practices |
| [API Reference](https://bam-engine.readthedocs.io/en/latest/api/index.html)       | Complete reference for all components and operations                          |
| [Examples](https://bam-engine.readthedocs.io/en/latest/auto_examples/index.html)  | 16 runnable examples: basic, advanced, and extensions                         |
| [Extensions](https://bam-engine.readthedocs.io/en/latest/extensions/index.html)   | R&D / Growth+, buffer-stock consumption, taxation                             |
| [Validation](https://bam-engine.readthedocs.io/en/latest/validation/index.html)   | Scenario validation, scoring, robustness analysis                             |
| [Calibration](https://bam-engine.readthedocs.io/en/latest/calibration/index.html) | Morris screening, grid search, stability testing                              |

## Development

```bash
git clone https://github.com/kganitis/bam-engine.git
pip install -e ".[dev]"
pytest
ruff format . && ruff check --fix . && mypy
```

This project is part of ongoing MSc thesis research at the University of Piraeus, Greece. Code contributions will be welcome after thesis submission. In the meantime, bug reports, feedback, and documentation suggestions are appreciated via the [issue tracker](https://github.com/kganitis/bam-engine/issues).

See the [Development Guide](https://bam-engine.readthedocs.io/en/latest/development/index.html) for more on testing, linting, benchmarking, and contributing.

## Citation

If you use BAM Engine in your research, please cite:

1. **This software** - Use [`CITATION.cff`](CITATION.cff) or GitHub's "Cite this repository"
1. **The original BAM model** - Delli Gatti, D., Desiderio, S., Gaffeo, E., Cirillo, P., & Gallegati, M. (2011). *Macroeconomics from the Bottom-up*. Springer. DOI: [10.1007/978-88-470-1971-3](https://doi.org/10.1007/978-88-470-1971-3)

## License

MIT License - see [LICENSE](LICENSE) for details.
