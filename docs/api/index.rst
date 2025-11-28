API
===

This section contains the API reference for BAM Engine.

.. toctree::
   :maxdepth: 2
   :hidden:

   core
   roles
   relationships
   decorators
   registry
   operations
   types
   config
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

   core.Role
   core.Event
   core.Relationship

Agent Roles
-----------

Built-in role components for the three agent types.

**Firm Roles**

.. autosummary::
   :nosignatures:

   roles.Producer
   roles.Employer
   roles.Borrower

**Household Roles**

.. autosummary::
   :nosignatures:

   roles.Worker
   roles.Consumer

**Bank Roles**

.. autosummary::
   :nosignatures:

   roles.Lender

Relationships
-------------

Built-in relationship types for agent connections.

.. autosummary::
   :nosignatures:

   relationships.LoanBook

Core Infrastructure
-------------------

Pipeline and agent management infrastructure.

.. autosummary::
   :nosignatures:

   core.Pipeline
   Agent
   AgentType
   Economy

Decorators
----------

Simplified syntax for defining custom components.

.. autosummary::
   :nosignatures:

   role
   event
   relationship

Registry Functions
------------------

Functions for retrieving and listing registered components.

.. autosummary::
   :nosignatures:

   get_role
   get_event
   get_relationship
   list_roles
   list_events
   list_relationships

Operations Module
-----------------

NumPy-free operations for writing custom events. See :mod:`bamengine.ops` for the full module documentation.

**Arithmetic**: ``add``, ``subtract``, ``multiply``, ``divide``

**Assignment**: ``assign``

**Comparisons**: ``equal``, ``not_equal``, ``less``, ``less_equal``, ``greater``, ``greater_equal``

**Logical**: ``logical_and``, ``logical_or``, ``logical_not``

**Conditional**: ``where``, ``select``

**Element-wise**: ``maximum``, ``minimum``, ``clip``

**Aggregation**: ``sum``, ``mean``, ``any``, ``all``

**Array creation**: ``zeros``, ``ones``, ``full``, ``empty``, ``arange``

**Mathematical**: ``log``

**Utilities**: ``unique``, ``bincount``, ``isin``, ``argsort``, ``sort``

**Random**: ``uniform``

Type System
-----------

Type aliases for defining custom roles without NumPy knowledge.

.. autosummary::
   :nosignatures:

   Float
   Int
   Bool
   AgentId
   Rng

Configuration
-------------

Configuration and validation classes.

.. autosummary::
   :nosignatures:

   config.Config

Utilities
---------

Helper functions and utilities.

.. autosummary::
   :nosignatures:

   make_rng
