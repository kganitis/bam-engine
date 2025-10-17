"""Agent entity definition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class AgentType(Enum):
    """Types of agents in the BAM model."""

    FIRM = auto()
    HOUSEHOLD = auto()
    BANK = auto()


@dataclass(slots=True, frozen=True)
class Agent:
    """
    Lightweight entity representing an agent in the simulation.

    Agents are just identifiers - all state lives in Role components.
    Being frozen ensures they're immutable and can be safely passed around.

    Parameters
    ----------
    id : int
        Unique identifier for this agent (corresponds to array index).
    agent_type : AgentType
        Type of agent (FIRM, HOUSEHOLD, or BANK).

    Examples
    --------
    >>> firm = Agent(id=0, agent_type=AgentType.FIRM)
    >>> household = Agent(id=5, agent_type=AgentType.HOUSEHOLD)
    """

    id: int
    agent_type: AgentType

    def __post_init__(self) -> None:
        """Validate agent ID is non-negative."""
        if self.id < 0:
            raise ValueError(f"Agent ID must be non-negative, got {self.id}")
