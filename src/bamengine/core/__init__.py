"""
Core ECS infrastructure for BAM Engine.

This module provides the fundamental building blocks of the BAM-ECS architecture:
roles (components), events (systems), pipelines, relationships, and the registry
system.

Key Components
--------------
- **Role**: Base class for agent state components
- **Event**: Base class for economic logic (systems)
- **Pipeline**: Manages event execution order
- **Relationship**: Base class for relationships between roles
- **Registry**: Global lookup for roles, events, and relationships

Decorators
----------
- **@role**: Simplified syntax for defining roles
- **@event**: Simplified syntax for defining events
- **@relationship**: Simplified syntax for defining relationships

Examples
--------
Import core classes:

>>> from bamengine.core import Role, Event, Pipeline
>>> from bamengine.core import get_role, get_event, list_roles

Use decorators for simplified syntax:

>>> from bamengine.core import role, event
>>> from bamengine.typing import Float
>>>
>>> @role
... class MyRole:
...     value: Float
>>>
>>> @event
... class MyEvent:
...     def execute(self, sim):
...         pass

Access registry:

>>> from bamengine.core import get_role, list_roles
>>> Producer = get_role("Producer")
>>> all_roles = list_roles()

See Also
--------
:mod:`bamengine.core.role` : Role base class and related functions
:mod:`bamengine.core.event` : Event base class and related functions
:mod:`bamengine.core.relationship` : Relationship base class and related functions
:mod:`bamengine.core.pipeline` : Pipeline management
:mod:`bamengine.core.decorators` : Decorator implementations
:mod:`bamengine.core.registry` : Registry functions
"""

from collections.abc import Callable
from typing import Any, TypeVar

from bamengine.core.agent import Agent, AgentType
from bamengine.core.decorators import event as event_decorator
from bamengine.core.decorators import relationship as relationship_decorator
from bamengine.core.decorators import role as role_decorator
from bamengine.core.event import Event
from bamengine.core.pipeline import Pipeline
from bamengine.core.registry import (
    get_event,
    get_relationship,
    get_role,
    list_events,
    list_relationships,
    list_roles,
)
from bamengine.core.relationship import Relationship
from bamengine.core.role import Role

_T = TypeVar("_T")

# Export decorator functions with their intended names
# These override the submodule names to provide cleaner API
event: Callable[..., Any] = event_decorator
role: Callable[..., Any] = role_decorator
relationship: Callable[..., Any] = relationship_decorator

__all__ = [
    "Agent",
    "AgentType",
    "Event",
    "Pipeline",
    "Relationship",
    "Role",
    "event",
    "get_event",
    "get_relationship",
    "get_role",
    "list_events",
    "list_relationships",
    "list_roles",
    "relationship",
    "role",
]
