# BAM Engine

**Bottom-Up Adaptive Macroeconomics Simulation Framework**

[![PyPI version](https://img.shields.io/pypi/v/bamengine.svg)](https://pypi.org/project/bamengine/)
[![Python](https://img.shields.io/pypi/pyversions/bamengine.svg)](https://pypi.org/project/bamengine/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[![Tests](https://img.shields.io/badge/tests-99%25%20coverage-brightgreen)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checked](https://img.shields.io/badge/type%20checked-mypy-blue)](http://mypy-lang.org/)
[![Linter: Ruff](https://img.shields.io/badge/linter-ruff-orange)](https://github.com/astral-sh/ruff)

BAM Engine is a high-performance Python implementation of the BAM model from *Macroeconomics from the Bottom-up* (Delli Gatti et al., 2011, Chapter 3). It provides a modular, extensible agent-based macroeconomic simulation framework built on ECS (Entity-Component-System) architecture with fully vectorized NumPy operations.

> **Note**: This (**v0.1.0**) release is feature-complete for the core BAM model but APIs may change in future releases before v1.0.0. Designed for academic research and policy analysis experiments.

## Features

### BAM Model

Implementation of 3 agent types (firms, households, banks) interacting in 3 markets (labor, credit, consumption goods) over 8 economic phases per period, capturing emergent macroeconomic dynamics like inflation and unemployment.

### High Performance

Usage of vectorized NumPy operations, allowing the simulation of large economies (500+ firms, 2500+ households) in seconds per 1000 periods on consumer hardware.

### ECS Architecture

BAM's Entity-Component-System design separates agent state (Roles) from behavior (Events) and allows easy customization and extension of the engine.

### API

Friendly to users without deep Python expertise, with:

* Easy access to core ECS components (Roles, Events, Relationships) that allows creation, modification and extension
* Vectorized operations module with safe handling that wraps NumPy functions
* Three-tier configuration system (package defaults -> user YAML -> keyword args)

*Subject to change before v1.0.0.*

### Reproducibility & Testing

* Deterministic simulations with seedable RNG for reproducible results
* Comprehensive test suite (99%+ coverage) with unit, integration, and property-based tests
* Performance regression testing to prevent degradation

## Quick Start

### Installation

```bash
# Install (first time)
pip install bamengine

# Upgrade to latest version
pip install -U bamengine
```

**Dependencies**: NumPy (≥1.26) and PyYAML (≥6.0.2) are automatically installed by pip.

**Requirements**: Python 3.11 or higher.

### Basic Usage

```python
import bamengine as bam

# Initialize simulation with default configuration
sim = bam.Simulation.init(seed=42)

# Run (1000 periods by default)
sim.run()

# Access results
unemployment = sim.ec.unemp_rate_history[-1]
avg_price = sim.ec.price_history[-1]
print(f"Final unemployment: {unemployment:.2%}")
print(f"Final average price: {avg_price:.2f}")
```

## Architecture

BAM Engine uses an **ECS (Entity-Component-System)** architecture for modularity and performance:

* **Agents**: Lightweight entities with immutable IDs and types (FIRM, HOUSEHOLD, BANK)
* **Roles (Components)**: Dataclasses storing agent state as NumPy arrays (Producer, Worker, Lender, etc.)
* **Events (Systems)**: Pure functions operating on roles, executed in pipeline order
* **Relationships**: Many-to-many connections with sparse COO format (e.g., LoanBook for loans)
* **Pipeline**: YAML-configurable event execution with special syntax (repeat, interleave)

### Agent Roles

* **Firms**: Producer + Employer + Borrower
* **Households**: Worker + Consumer
* **Banks**: Lender

### Event Pipeline

Each period executes 39 events across 8 economic phases:

1. **Planning**: Production targets, pricing, labor needs
2. **Labor Market**: Wage setting, job search, hiring (4 rounds)
3. **Credit Market**: Loan supply/demand, matching (2 rounds)
4. **Production**: Wage payments, production execution
5. **Goods Market**: Consumption decisions, shopping (2 rounds)
6. **Revenue**: Revenue collection, debt repayment, dividends
7. **Bankruptcy**: Insolvency detection, exit
8. **Entry**: Replacement firm/bank spawning

## Creating Custom Components

### Custom Role

```python
from bamengine import role
from bamengine.typing import Float, Int

@role
class Inventory:
    """Custom inventory management role."""
    goods_on_hand: Float
    reorder_point: Float
    days_until_delivery: Int
```

### Custom Event

```python
from bamengine import event, ops, Simulation

@event
class CustomPricingEvent:
    """Apply markup pricing to all producers."""

    def execute(self, sim: Simulation) -> None:
        prod = sim.get_role("Producer")
        emp = sim.get_role("Employer")

        # Calculate unit labor cost
        unit_cost = ops.divide(emp.wage_offered, prod.labor_prod)

        # Apply 50% markup
        new_price = ops.multiply(unit_cost, 1.5)

        # Update prices in-place
        ops.assign(prod.price, new_price)
```

### Custom Relationship

```python
from bamengine import relationship, get_role
from bamengine.typing import Float, Int

@relationship(source=get_role("Worker"), target=get_role("Employer"))
class GigEmployment:
    """Many-to-many employment relationship."""
    wage: Float
    contract_duration: Int
    start_period: Int
```

## Configuration

### Three-Tier System

1. **Package defaults** (`src/bamengine/config/defaults.yml`)
2. **User YAML file** (custom configuration)
3. **Keyword arguments** (highest priority)

### Example Configuration

```yaml
# custom_config.yml
n_firms: 200
n_households: 1000
n_banks: 20
seed: 42

# Custom pipeline
pipeline_path: "custom_pipeline.yml"

# Logging configuration
logging:
  default_level: INFO
  events:
    workers_send_one_round: WARNING
    firms_hire_workers: DEBUG
```

Parameters not specified in custom YAML file automatically fall back to package defaults (`src/bamengine/config/defaults.yml`).

```python
sim = bam.Simulation.init(
    config="custom_config.yml",
    # Override YAML
    n_firms=250,
    seed=123
)
```

Parameters not specified in keyword arguments fall back to custom YAML, then package defaults.

## Performance

BAM Engine achieves excellent performance through vectorization:

| Configuration | Firms | Households | Throughput    |
|---------------|-------|------------|---------------|
| Small         | 100   | 500        | 172 periods/s |
| Medium        | 200   | 1,000      | 96 periods/s  |
| Large         | 500   | 2,500      | 40 periods/s  |

**Benchmarks** (1000 periods, Apple M4 Pro, macOS 15.1, Python 3.12):

* Small: 5.8s
* Medium: 10.4s
* Large: 24.5s

Performance scales sub-linearly with agent count due to NumPy vectorization efficiency.

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/kganitis/bam-engine.git
cd bam-engine

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check --fix src/ tests/

# Type checking
mypy src/
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/bamengine

# Skip slow tests
pytest -m "not slow and not regression and not invariants" -v
```

### Benchmarking

```bash
# Run macro-benchmarks (full simulation)
python benchmarks/bench_full_simulation.py

# Profile with cProfile
python benchmarks/profile_simulation.py

# Performance regression tests
pytest tests/performance/ -v
```

## Documentation

Not yet available. Coming soon!

## Project Status

This project was developed as part of the final thesis for MSc in Informatics at the University of Piraeus, Athens, Greece.

## Contributing

This project is currently not accepting external contributions as it is part of ongoing thesis work. Once the thesis is submitted, contribution guidelines will be published.

For bug reports and feature requests, please open an issue on [GitHub](https://github.com/kganitis/bam-engine/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use BAM Engine in your research, please cite both:

1. **This software** - Use the metadata in [`CITATION.cff`](CITATION.cff) or GitHub's "Cite this repository" button
2. **The original BAM model** - Delli Gatti, D., Desiderio, S., Gaffeo, E., Cirillo, P., & Gallegati, M. (2011). The BAM model at work. In Macroeconomics from the Bottom-up (New Economic Windows). Springer Milano. DOI: [10.1007/978-88-470-1971-3](https://doi.org/10.1007/978-88-470-1971-3)

Both citations are included in [`CITATION.cff`](CITATION.cff) for citation management tools.
