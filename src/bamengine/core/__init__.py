"""Core ECS infrastructure for BAM Engine."""

from typing import Any, Callable, TypeVar

from bamengine.core.agent import Agent, AgentType
from bamengine.core.decorators import event as event_decorator, \
    relationship as relationship_decorator
from bamengine.core.decorators import role as role_decorator
from bamengine.core.event import Event
from bamengine.core.pipeline import Pipeline
from bamengine.core.registry import get_event, get_role, list_events, list_roles, \
    get_relationship, list_relationships
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
