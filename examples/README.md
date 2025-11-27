# BAM Engine Examples

This gallery contains examples demonstrating various features and use cases of BAM Engine.

## Getting Started

All examples assume BAM Engine is installed. You can install it via:

```bash
# For users
pip install bamengine

# For developers (editable install from project root)
pip install -e .
```

## Example Categories

### Basic Examples

Start here if you're new to BAM Engine. These examples demonstrate fundamental concepts and basic usage patterns.

| Example | Description |
|---------|-------------|
| **Hello World** | Simplest possible example - initialize, run, visualize |
| **Configuration** | Customize parameters using keyword arguments |
| **Baseline Scenario** | Reproduce results from the BAM literature |
| **YAML Configuration** | Configure simulations using YAML files |
| **Results Module** | Collect and analyze simulation results |
| **Type System** | Using Float, Int, Bool, Agent type aliases |
| **Logging** | Configure logging levels globally and per-event |

### Advanced Examples

Examples showing advanced features like custom events, roles, relationships, and pipeline configuration.

| Example | Description |
|---------|-------------|
| **Custom Roles** | Define new agent components with `@role` decorator |
| **Custom Events** | Create custom events with `@event` decorator |
| **Custom Relationships** | Define many-to-many relationships with edge data |
| **Custom Pipeline** | Customize event execution order via YAML |
| **Ops Module** | NumPy-free operations for custom event logic |

### Research Examples

Examples demonstrating extensions from the original BAM model (Delli Gatti et al., 2011).

- **Growth+ Model**: R&D investment and endogenous productivity growth
- **Consumption and Buffer Shock**: Household consumption shocks and savings buffer dynamics
- **Parameter Space Exploration**: Systematic exploration of model sensitivity to key parameters
- **Preferential Attachment**: Preferential attachment in consumption and firm entry mechanisms

## Running the Examples

Each example is a standalone Python script that can be run directly:

```bash
# Basic examples
python examples/basic/example_hello_world.py
python examples/basic/example_yaml_configuration.py
python examples/basic/example_results_module.py

# Advanced examples
python examples/advanced/example_custom_roles.py
python examples/advanced/example_custom_events.py
python examples/advanced/example_ops_module.py
```

## Example Structure

All examples follow the Sphinx Gallery format with `# %%` cell markers:

```python
"""
Title of Example
================

Brief description of what this example demonstrates.
"""

# %%
# Section Title
# -------------
#
# Description of this section.

import bamengine as bam

# Example code here...
```

This format allows examples to be:
- Run as standalone scripts
- Converted to Jupyter notebooks
- Rendered in documentation with Sphinx Gallery

## Learning Path

Recommended order for learning BAM Engine:

1. **Start with basics**: `hello_world` → `configuration` → `baseline_scenario`
2. **Learn data handling**: `results_module` → `yaml_configuration`
3. **Understand types**: `typing_module` → `logging`
4. **Extend the model**: `custom_roles` → `custom_events` → `custom_relationships`
5. **Customize execution**: `custom_pipeline` → `ops_module`
