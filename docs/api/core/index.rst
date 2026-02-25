Core Infrastructure
===================

.. automodule:: bamengine.core
   :no-members:

The ``bamengine.core`` module provides the fundamental building blocks of the
BAM-ECS architecture.

.. currentmodule:: bamengine.core

Classes
-------

- :class:`~role.Role` — Base class for agent components
- :class:`~event.Event` — Base class for simulation events
- :class:`~relationship.Relationship` — Base class for agent relationships
- :class:`~Pipeline` — Event execution sequence

Decorators
----------

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   role
   event
   relationship

Submodules
----------

.. toctree::
   :maxdepth: 1

   role
   event
   relationship
   pipeline
   decorators
   registry
