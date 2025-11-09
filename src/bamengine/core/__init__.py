"""Core ECS infrastructure for BAM Engine."""

from typing import Any, Callable, TypeVar

from bamengine.core.agent import Agent, AgentType
from bamengine.core.decorators import event as event_decorator
from bamengine.core.decorators import role as role_decorator
from bamengine.core.event import Event
from bamengine.core.pipeline import Pipeline
from bamengine.core.registry import get_event, get_role, list_events, list_roles
from bamengine.core.role import Role

_T = TypeVar("_T")

# Export decorator functions with their intended names
# These override the submodule names to provide cleaner API
event: Callable[..., Any] = event_decorator
role: Callable[..., Any] = role_decorator

__all__ = [
    "Agent",
    "AgentType",
    "Event",
    "Pipeline",
    "Role",
    "event",
    "role",
    "get_event",
    "get_role",
    "list_events",
    "list_roles",
]
