"""Tests for Role base class."""

# noinspection PyPackageRequirements
from dataclasses import dataclass

# noinspection PyPackageRequirements
import numpy as np

from bamengine.core import Role
from bamengine.typing import Float1D


@dataclass(slots=True)
class DummyRole(Role):
    """Concrete role for testing."""

    values: Float1D
    flags: np.ndarray


def test_role_creation():
    """Test creating a concrete role."""
    role = DummyRole(
        values=np.array([1.0, 2.0, 3.0]),
        flags=np.array([True, False, True]),
    )

    assert len(role.values) == 3
    assert len(role.flags) == 3


def test_role_slots():
    """Test that role uses slots (no __dict__)."""
    role = DummyRole(
        values=np.zeros(10),
        flags=np.zeros(10, dtype=bool),
    )

    assert not hasattr(role, "__dict__")


def test_role_repr():
    """Test role string representation."""
    role = DummyRole(
        values=np.ones(5),
        flags=np.zeros(5, dtype=bool),
    )

    repr_str = repr(role)
    assert "DummyRole" in repr_str or "fields" in repr_str
