# BAM-Engine (Work in Progress)

A high-performance Python implementation of the BAM model from *Macroeconomics from the Bottom-Up* (Delli Gatti et al., 2011).

---

### Architecture

**Entity–Component–System (ECS)** architecture optimized for large-scale agent-based modeling:

* **Entities** – Economic actors (firms, households, banks) are represented as integer indices, eliminating object overhead
* **Components** – Immutable NumPy dataclass containers in `bamengine.components` hold pure state (e.g., `Producer`, `Worker`, `Lender`) with zero business logic
* **Systems** – Vectorized functions in `bamengine.systems.*` implement economic behavior through efficient in-place mutations
* **Simulation** – `bamengine.Simulation` orchestrates the simulation lifecycle. Basic public API that enables one-liner usage:

```python
import bamengine as be

# Configure and run simulation
sim = be.Simulation.init(n_firms=100, n_households=500, n_banks=10)
sim.run(n_periods=1000)    # or sim.step() for single-period control
```

---

### Technologies

**Minimal, high-performance stack:**

* **Python 3.10+** – Fully type-annotated with `py.typed` for downstream type checking
* **NumPy** – Dense arrays and vectorized operations for maximum computational efficiency
* **PyYAML** – Declarative configuration system (`defaults.yml` + user overrides)

**Zero heavyweight dependencies** – No pandas, no complex frameworks. Pure scientific Python for predictable performance and easy deployment.

---

### Current Status

**Version 0.0.0 – Experimental proof-of-concept**

##### Implemented
* Complete BAM model systems with comprehensive unit tests
* Efficient vectorized kernels handling thousands of agents on standard hardware
* Validated output replicating key results from the original paper
* Extensive logging infrastructure for debugging and analysis

##### Limitations
* Public API restricted to basic `Simulation` interface
* Monolithic simulation design limits extensibility
* Hard-coded logging integration across systems
* Minimal documentation and examples

---

### Roadmap

##### Research
* Replicate **all** figures and empirical results from *Macroeconomics from the Bottom-Up*
* Comprehensive performance profiling and optimization

##### Engineering
* Achieve >95% test coverage across all modules
* Implement structured, configurable logging system
* Refactor monolithic `Simulation` into modular, plugin-based architecture with clean separation of concerns

##### Usage
* Comprehensive API documentation with examples
* End-to-end Jupyter notebook tutorials
* Production-ready packaging and distribution