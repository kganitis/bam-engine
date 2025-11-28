API
===

This section contains the API reference for BAM Engine.

.. currentmodule:: bamengine

Core Classes
------------

Main classes for running simulations and collecting results.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Simulation
   SimulationResults

ECS Base Classes
----------------

Base classes for creating custom components in the ECS architecture.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Role
   Event
   Relationship

Agent Roles
-----------

Built-in role components for the three agent types.

**Firm Roles**

.. autosummary::
   :toctree: generated
   :nosignatures:

   roles.Producer
   roles.Employer
   roles.Borrower

**Household Roles**

.. autosummary::
   :toctree: generated
   :nosignatures:

   roles.Worker
   roles.Consumer

**Bank Roles**

.. autosummary::
   :toctree: generated
   :nosignatures:

   roles.Lender

Relationships
-------------

Built-in relationship types for agent connections.

.. autosummary::
   :toctree: generated
   :nosignatures:

   relationships.LoanBook

Core Infrastructure
-------------------

Pipeline and agent management infrastructure.

.. autosummary::
   :toctree: generated
   :nosignatures:

   core.Pipeline
   Agent
   AgentType
   Economy

Decorators
----------

Simplified syntax for defining custom components.

.. autosummary::
   :toctree: generated
   :nosignatures:

   role
   event
   relationship

Registry Functions
------------------

Functions for retrieving and listing registered components.

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_role
   get_event
   get_relationship
   list_roles
   list_events
   list_relationships

Operations Module
-----------------

NumPy-free operations for writing custom events. See :mod:`bamengine.ops` for the full list of operations.

**Arithmetic**: ``add``, ``subtract``, ``multiply``, ``divide``

**Assignment**: ``assign``

**Comparisons**: ``equal``, ``not_equal``, ``less``, ``less_equal``, ``greater``, ``greater_equal``

**Logical**: ``logical_and``, ``logical_or``, ``logical_not``

**Conditional**: ``where``, ``select``

**Element-wise**: ``maximum``, ``minimum``, ``clip``

**Aggregation**: ``sum``, ``mean``, ``any``, ``all``

**Array creation**: ``zeros``, ``ones``, ``full``, ``empty``

**Utilities**: ``unique``, ``bincount``, ``isin``, ``argsort``, ``sort``

**Random**: ``uniform``

Type System
-----------

Type aliases for defining custom roles without NumPy knowledge.

.. autosummary::
   :toctree: generated
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
   :toctree: generated
   :nosignatures:

   config.Config

Utilities
---------

Helper functions and utilities.

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_rng
