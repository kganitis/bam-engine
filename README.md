# BAM Engine

**Bottom-Up Adaptive Macroeconomics Simulation Framework**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

BAM Engine is a high-performance Python implementation of the BAM model from *Macroeconomics from the Bottom-up* (Delli Gatti et al., 2011, Chapter 3). It provides a modular, extensible agent-based macroeconomic simulation framework built on ECS (Entity-Component-System) architecture with fully vectorized NumPy operations.

## Features

**Complete BAM Model**: 3 agent types (firms, households, banks) interacting in 3 markets (labor, credit, consumption goods)

**High Performance**: Fully vectorized NumPy operations enable simulations of large economies (500+ firms, 2500+ households) with minimal overhead

**ECS Architecture**: Modular Entity-Component-System design separates agent state (Roles) from behavior (Events), enabling easy customization and extension of economic mechanisms

**User-Friendly API**: Simplified decorators, NumPy-free operations module, and three-tier configuration system for researchers without deep Python expertise

**Research-Ready**: Deterministic RNG, comprehensive validation, extensive testing suite

**Flexible Configuration**: Three-tier system (BAM defaults → user YAML → kwargs parameters) with customizable event pipeline

## Quick Start

### Installation

```bash
pip install bamengine
```

Or install from source:

```bash
git clone https://github.com/yourusername/bam-engine.git
cd bam-engine
pip install -e ".[dev]"
```

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

### Custom Configuration

```python
# Via keyword arguments
sim = bam.Simulation.init(
    n_firms=200,
    n_households=1000,
    n_banks=20,
    seed=42
)

# Via custom YAML file
sim = bam.Simulation.init(config="my_config.yml", seed=42)

# Step-by-step execution
sim = bam.Simulation.init(seed=42)
for period in range(100):
    sim.step()
    # Analyze state after each period
    print(f"Period {period}: unemployment = {sim.ec.unemp_rate:.2%}")
```

## Architecture

BAM Engine uses an **ECS (Entity-Component-System)** architecture for modularity and performance:

- **Agents**: Lightweight entities with immutable IDs and types (FIRM, HOUSEHOLD, BANK)
- **Roles (Components)**: Dataclasses storing agent state as NumPy arrays (Producer, Worker, Lender, etc.)
- **Events (Systems)**: Pure functions operating on roles, executed in pipeline order
- **Relationships**: Many-to-many connections with sparse COO format (e.g., LoanBook for loans)
- **Pipeline**: YAML-configurable event execution with special syntax (repeat, interleave)

### Agent Roles

- **Firms**: Producer + Employer + Borrower
- **Households**: Worker + Consumer
- **Banks**: Lender

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

Parameters not specified in your YAML file automatically fall back to package defaults (`src/bamengine/config/defaults.yml`).

```yaml
# config.yml
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

```python
sim = bam.Simulation.init(
    config="config.yml",
    n_firms=250,  # Overrides YAML
    seed=123      # Overrides YAML
)
```

## Performance

BAM Engine achieves excellent performance through vectorization:

| Configuration | Firms | Households | Throughput |
|--------------|-------|------------|------------|
| Small | 100 | 500 | 172 periods/s |
| Medium | 200 | 1,000 | 96 periods/s |
| Large | 500 | 2,500 | 40 periods/s |

**Benchmarks** (1000 periods, Apple M4 Pro, macOS 15.1, Python 3.13):

- Small: 5.8s
- Medium: 10.4s
- Large: 24.5s

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

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checked](https://img.shields.io/badge/type%20checked-mypy-blue)](http://mypy-lang.org/)
[![Linter: Ruff](https://img.shields.io/badge/linter-ruff-orange)](https://github.com/astral-sh/ruff)

```bash
# Format code
black src/ tests/

# Lint code
ruff check --fix src/ tests/

# Type checking
mypy src/
```

## Testing

[![Tests](https://img.shields.io/badge/tests-99%25%20coverage-brightgreen)]()

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/bamengine

# Run specific test categories
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest tests/property/       # Property-based tests
pytest -m "not slow"         # Skip slow tests
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

**Version**: 0.1.0

This release is feature-complete for the core BAM model but APIs may change in future releases. Designed for academic research and policy analysis experiments.

## Citation

If you use BAM Engine in your research, please also cite the original BAM model:

```bibtex
@inbook{DelliGatti2011Chapter,
    author    = {Delli Gatti, Domenico and Desiderio, Saul and Gaffeo, Edoardo and Cirillo,
  Pasquale and Gallegati, Mauro},
    title     = {Macroeconomics from the Bottom-up},
    chapter   = {The BAM model at work},
    year      = {2011},
    publisher = {Springer Milano},
    series    = {New Economic Windows},
    volume    = {1},
    doi       = {10.1007/978-88-470-1971-3},
    isbn      = {978-88-470-1971-3}
  }
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/feature-name`)
3. Commit your changes (`git commit -m 'feat: description'`)
4. Push to the branch (`git push origin feat/feature-name`)
5. Open a Pull Request

Please ensure:

- Code is formatted (`black src/ tests/`)
- Linting passes (`ruff check src/ tests/`)
- Type checking passes (`mypy src/`)
- All tests pass (`pytest`)
- Test coverage remains >95%

## Acknowledgments

- **The BAM model**: Delli Gatti, D., Desiderio, S., Gaffeo, E., Cirillo, P., & Gallegati, M. (2011). The BAM model at work. In Macroeconomics from the Bottom-up (New Economic Windows). Springer Milano. [10.1007/978-88-470-1971-3](https://doi.org/10.1007/978-88-470-1971-3)

- **Built with**: NumPy, Python 3.10+

---

**Issues?** Open an issue on [GitHub](https://github.com/kganitis/bam-engine/issues)
