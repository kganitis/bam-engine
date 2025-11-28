"""
Agent entity definition.

This module defines the Agent class and AgentType enum. In BAM-ECS architecture,
agents are lightweight entities containing only an ID and type. All agent state
lives in Role components (e.g., Producer, Worker, Lender).

Design Notes
------------
- Agents are immutable (frozen dataclasses) for safety
- Agent IDs correspond to array indices in Role components
- Agent type distinguishes firms, households, and banks
- Currently not used in the vectorized implementation (state accessed via roles)

Examples
--------
Create agents of different types:

>>> from bamengine.core.agent import Agent, AgentType
>>> firm = Agent(id=0, agent_type=AgentType.FIRM)
>>> household = Agent(id=100, agent_type=AgentType.HOUSEHOLD)
>>> bank = Agent(id=200, agent_type=AgentType.BANK)

Access agent properties:

>>> firm.id
0
>>> firm.agent_type
<AgentType.FIRM: 1>
>>> household.agent_type.name
'HOUSEHOLD'

See Also
--------
:class:`~bamengine.core.Role` : Base class for component state
:class:`AgentType` : Enum of agent types
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class AgentType(Enum):
    """
    Types of agents in the BAM model.

    Attributes
    ----------
    FIRM : int
        Firm agent type (producer, employer, borrower).
    HOUSEHOLD : int
        Household agent type (worker, consumer).
    BANK : int
        Bank agent type (lender).

    Examples
    --------
    >>> from bamengine.core.agent import AgentType
    >>> AgentType.FIRM
    <AgentType.FIRM: 1>
    >>> AgentType.HOUSEHOLD.name
    'HOUSEHOLD'
    >>> AgentType.BANK.value
    3
    """

    FIRM = auto()
    HOUSEHOLD = auto()
    BANK = auto()


@dataclass(slots=True, frozen=True)
class Agent:
    """
    Lightweight entity representing an agent in the simulation.

    Agents are just identifiers - all state lives in Role components.
    Being frozen ensures they're immutable and can be safely passed around.

    Attributes
    ----------
    id : int
        Unique identifier for this agent (corresponds to array index).
    agent_type : AgentType
        Type of agent (FIRM, HOUSEHOLD, or BANK).

    Examples
    --------
    Create agents:

    >>> from bamengine.core.agent import Agent, AgentType
    >>> firm = Agent(id=0, agent_type=AgentType.FIRM)
    >>> household = Agent(id=5, agent_type=AgentType.HOUSEHOLD)
    >>> bank = Agent(id=2, agent_type=AgentType.BANK)

    Access properties:

    >>> firm.id
    0
    >>> firm.agent_type
    <AgentType.FIRM: 1>

    Agents are immutable:

    >>> firm.id = 10  # Raises FrozenInstanceError
    Traceback (most recent call last):
        ...
    dataclasses.FrozenInstanceError: cannot assign to field 'id'

    Notes
    -----
    Currently, agents are not extensively used in the vectorized implementation.
    State is accessed directly via roles (e.g., sim.prod, sim.wrk). The Agent
    class exists for potential future extensions.

    See Also
    --------
    :class:`AgentType` : Enum of agent types
    :class:`~bamengine.core.Role` : Base class for component state
    """

    id: int
    agent_type: AgentType

    def __post_init__(self) -> None:
        """
        Validate agent ID is non-negative.

        Raises
        ------
        ValueError
            If agent ID is negative.
        """
        if self.id < 0:
            raise ValueError(f"Agent ID must be non-negative, got {self.id}")
