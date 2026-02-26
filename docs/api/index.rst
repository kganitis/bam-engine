API
===

This section contains the API reference for BAM Engine.

.. toctree::
   :maxdepth: 2
   :hidden:

   simulation
   core/index
   roles/index
   events/index
   relationships/index
   economy
   config/index
   operations
   types
   logging
   utilities

.. currentmodule:: bamengine

Core Classes
------------

Main classes for running simulations and collecting results.

.. autosummary::
   :nosignatures:

   Simulation
   SimulationResults

ECS Base Classes
----------------

Base classes for creating custom components in the ECS architecture.

.. autosummary::
   :nosignatures:

   ~core.role.Role
   ~core.event.Event
   ~core.relationship.Relationship

Agent Roles
-----------

Built-in role components for the three agent types.
See :doc:`roles/index` for all available roles.

Relationships
-------------

Built-in relationship types for agent connections.
See :doc:`relationships/index` for details.

Core Infrastructure
-------------------

Pipeline, economy state, and agent management infrastructure.
Agent identity types (``Agent``, ``AgentType``) are defined in :mod:`bamengine.core.agent`.

.. autosummary::
   :nosignatures:

   ~core.pipeline.Pipeline
   Economy

Decorators
----------

Simplified syntax for defining custom components.

.. autosummary::
   :nosignatures:

   ~core.decorators.role
   ~core.decorators.event
   ~core.decorators.relationship

Registry Functions
------------------

Functions for retrieving and listing registered components.

.. autosummary::
   :nosignatures:

   ~core.registry.get_role
   ~core.registry.get_event
   ~core.registry.get_relationship
   ~core.registry.list_roles
   ~core.registry.list_events
   ~core.registry.list_relationships

Operations Module
-----------------

NumPy-free operations for writing custom events. See :mod:`bamengine.ops` for the full module documentation.

**Arithmetic**: ``add``, ``subtract``, ``multiply``, ``divide``

**Assignment**: ``assign``

**Comparisons**: ``equal``, ``not_equal``, ``less``, ``less_equal``, ``greater``, ``greater_equal``

**Logical**: ``logical_and``, ``logical_or``, ``logical_not``

**Conditional**: ``where``

**Element-wise**: ``maximum``, ``minimum``, ``clip``

**Aggregation**: ``sum``, ``mean``, ``std``, ``min``, ``max``, ``any``, ``all``

**Array creation**: ``zeros``, ``ones``, ``full``, ``empty``, ``arange``

**Mathematical**: ``log``, ``exp``

**Utilities**: ``unique``, ``bincount``, ``isin``, ``argsort``, ``sort``

**Random**: ``uniform``

Type System
-----------

Type aliases for defining custom roles without NumPy knowledge.
See :doc:`types` for details.

``Float``, ``Int``, ``Bool``, ``AgentId``, ``Rng``

Configuration
-------------

Configuration and validation classes.

.. autosummary::
   :nosignatures:

   ~config.schema.Config

Utilities
---------

Helper functions and utilities.

.. autosummary::
   :nosignatures:

   make_rng
