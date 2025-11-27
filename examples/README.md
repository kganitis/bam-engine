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

- **Basic Simulation**: Run a simple simulation with default parameters
- **Configuration**: Customize simulation parameters via YAML and kwargs
- **Data Collection**: Extract and analyze simulation results

### Advanced Examples
Examples showing advanced features like custom events, roles, relationships, and pipeline configuration.

- **Custom Events**: Create your own events to extend the model
- **Custom Roles**: Add new agent behaviors with custom roles
- **Custom Relationships**: Define many-to-many relationships between roles with edge data
- **Pipeline Modification**: Customize the event execution pipeline
- **Multi-Run Analysis**: Run parameter sweeps and ensemble simulations

### Research Examples
Examples demonstrating extensions from the original BAM model (Delli Gatti et al., 2011).

- **Growth+ Model**: R&D investment and endogenous productivity growth
- **Consumption and Buffer Shock**: Household consumption shocks and savings buffer dynamics
- **Parameter Space Exploration**: Systematic exploration of model sensitivity to key parameters
- **Preferential Attachment**: Preferential attachment in consumption and firm entry mechanisms

## Running the Examples

Each example is a standalone Python script that can be run directly:

```bash
python examples/basic/plot_baseline_scenario.py
```

## Example Structure

Each example follows this structure:

```python
"""
Title of Example
================

Brief description of what this example demonstrates.
"""

import bamengine as bam
import matplotlib.pyplot as plt

# Example code here...
```
