"""Tests for the Extension dataclass."""

from dataclasses import FrozenInstanceError

import pytest

from bamengine import Extension, Float, Int, role


class TestExtensionCreation:
    """Test Extension dataclass construction."""

    def test_create_with_all_fields(self) -> None:
        """Extension can be created with all four fields."""

        @role
        class MyRole:
            value: Float

        class MyEvent:
            pass

        ext = Extension(
            roles={MyRole: "firms"},
            events=[MyEvent],
            relationships=[],
            config_dict={"param": 1.0},
        )
        assert ext.roles == {MyRole: "firms"}
        assert ext.events == [MyEvent]
        assert ext.relationships == []
        assert ext.config_dict == {"param": 1.0}

    def test_create_with_defaults(self) -> None:
        """Extension can be created with all defaults (empty)."""
        ext = Extension()
        assert ext.roles == {}
        assert ext.events == []
        assert ext.relationships == []
        assert ext.config_dict == {}

    def test_frozen_immutable(self) -> None:
        """Extension instances are frozen (immutable)."""
        ext = Extension()
        with pytest.raises(
            (AttributeError, FrozenInstanceError), match="cannot assign"
        ):
            ext.roles = {"new": "value"}  # type: ignore[misc]

    def test_multiple_roles_different_agent_types(self) -> None:
        """Extension can map roles to different agent types."""

        @role
        class FirmRole:
            x: Float

        @role
        class HouseholdRole:
            y: Int

        ext = Extension(
            roles={FirmRole: "firms", HouseholdRole: "households"},
            events=[],
            relationships=[],
            config_dict={},
        )
        assert len(ext.roles) == 2
        assert ext.roles[FirmRole] == "firms"
        assert ext.roles[HouseholdRole] == "households"

    def test_events_only_extension(self) -> None:
        """Extension can have events and config but no roles."""

        class TaxEvent:
            pass

        ext = Extension(
            roles={},
            events=[TaxEvent],
            relationships=[],
            config_dict={"tax_rate": 0.1},
        )
        assert ext.roles == {}
        assert len(ext.events) == 1
