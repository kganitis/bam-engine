"""Core ECS infrastructure for BAM Engine."""

from bamengine.core.agent import Agent, AgentType
from bamengine.core.event import Event
from bamengine.core.registry import get_event, get_role, list_events, list_roles
from bamengine.core.role import Role

__all__ = [
    "Agent",
    "AgentType",
    "Event",
    "Role",
    "get_event",
    "get_role",
    "list_events",
    "list_roles",
]
