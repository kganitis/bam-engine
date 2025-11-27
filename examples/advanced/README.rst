Advanced Examples
=================

Examples showing advanced features like custom events, roles, relationships,
and pipeline configuration. These examples require familiarity with the basic
concepts covered in the basic examples.

Examples in this section:

1. **Custom Roles**: Define new agent components with the ``@role`` decorator
2. **Custom Events**: Create custom events with the ``@event`` decorator
3. **Custom Relationships**: Define many-to-many relationships with edge data
4. **Custom Pipeline**: Customize the event execution pipeline via YAML
5. **Ops Module**: NumPy-free operations for custom event logic

These examples will teach you:

* How to extend the model with custom agent behaviors using roles
* How to create custom economic events and policies
* How to define relationships between roles with edge-specific data
* How to modify the event execution order
* How to use the ops module for safe array operations
* How to integrate custom components into simulations

Prerequisites
-------------

Before working through these examples, make sure you're comfortable with:

* Basic simulation setup (see basic/example_hello_world.py)
* Configuration via kwargs (see basic/example_configuration.py)
* Results collection (see basic/example_results_module.py)
* Type aliases (see basic/example_typing_module.py)
