"""Tests for Agent entity."""

# noinspection PyPackageRequirements
import pytest

from bamengine.core import Agent, AgentType


def test_agent_creation():
    """Test basic agent creation."""
    firm = Agent(id=0, agent_type=AgentType.FIRM)

    assert firm.id == 0
    assert firm.agent_type == AgentType.FIRM


def test_agent_immutable():
    """Test that agents are immutable (frozen)."""
    agent = Agent(id=5, agent_type=AgentType.HOUSEHOLD)

    with pytest.raises(AttributeError):
        # noinspection PyDataclass
        agent.id = 10


def test_agent_negative_id_raises():
    """Test that negative IDs are rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        Agent(id=-1, agent_type=AgentType.BANK)


def test_agent_types():
    """Test all agent types can be created."""
    firm = Agent(id=0, agent_type=AgentType.FIRM)
    household = Agent(id=1, agent_type=AgentType.HOUSEHOLD)
    bank = Agent(id=2, agent_type=AgentType.BANK)

    assert firm.agent_type == AgentType.FIRM
    assert household.agent_type == AgentType.HOUSEHOLD
    assert bank.agent_type == AgentType.BANK


def test_agent_repr():
    """Test agent string representation."""
    agent = Agent(id=42, agent_type=AgentType.FIRM)
    repr_str = repr(agent)

    assert "42" in repr_str
    assert "FIRM" in repr_str
