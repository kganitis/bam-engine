"""Core ECS infrastructure for BAM Engine."""

from bamengine.core.agent import Agent, AgentType
from bamengine.core.role import Role
from bamengine.core.event import Event
from bamengine.core.registry import role, event, get_role, get_event

__all__ = [
    "Agent",
    "AgentType",
    "Role",
    "Event",
    "role",
    "event",
    "get_role",
    "get_event",
]
