"""Unit tests for event pipeline hooks."""

import pytest

from bamengine.core.decorators import event
from bamengine.core.pipeline import Pipeline
from bamengine.core.registry import get_event_hooks, register_event_hook


class TestEventHookRegistration:
    """Tests for event hook registration via decorator and registry."""

    def test_event_with_after_hook_registers_correctly(self, clean_registry):
        """@event(after=...) registers hook in registry."""

        @event
        class TargetEvent:
            def execute(self, sim):
                pass

        @event(after="target_event")
        class HookedEvent:
            def execute(self, sim):
                pass

        hooks = get_event_hooks()
        assert "hooked_event" in hooks
        assert hooks["hooked_event"]["after"] == "target_event"
        assert hooks["hooked_event"]["before"] is None
        assert hooks["hooked_event"]["replace"] is None

    def test_event_with_before_hook_registers_correctly(self, clean_registry):
        """@event(before=...) registers hook in registry."""

        @event
        class TargetEvent:
            def execute(self, sim):
                pass

        @event(before="target_event")
        class HookedEvent:
            def execute(self, sim):
                pass

        hooks = get_event_hooks()
        assert "hooked_event" in hooks
        assert hooks["hooked_event"]["before"] == "target_event"
        assert hooks["hooked_event"]["after"] is None
        assert hooks["hooked_event"]["replace"] is None

    def test_event_with_replace_hook_registers_correctly(self, clean_registry):
        """@event(replace=...) registers hook in registry."""

        @event
        class TargetEvent:
            def execute(self, sim):
                pass

        @event(replace="target_event")
        class ReplacementEvent:
            def execute(self, sim):
                pass

        hooks = get_event_hooks()
        assert "replacement_event" in hooks
        assert hooks["replacement_event"]["replace"] == "target_event"
        assert hooks["replacement_event"]["after"] is None
        assert hooks["replacement_event"]["before"] is None

    def test_event_with_custom_name_and_hook(self, clean_registry):
        """@event(name=..., after=...) works with custom name."""

        @event
        class TargetEvent:
            def execute(self, sim):
                pass

        @event(name="my_custom_event", after="target_event")
        class HookedEvent:
            def execute(self, sim):
                pass

        hooks = get_event_hooks()
        assert "my_custom_event" in hooks
        assert hooks["my_custom_event"]["after"] == "target_event"

    def test_multiple_hooks_raises_error(self, clean_registry):
        """Cannot specify multiple hook types on same decorator."""
        with pytest.raises(ValueError, match="Only one of"):

            @event(after="event_a", before="event_b")
            class BadEvent:
                def execute(self, sim):
                    pass

    def test_multiple_hooks_raises_error_all_three(self, clean_registry):
        """Cannot specify all three hook types."""
        with pytest.raises(ValueError, match="Only one of"):

            @event(after="a", before="b", replace="c")
            class BadEvent:
                def execute(self, sim):
                    pass

    def test_event_without_hook_does_not_register_hook(self, clean_registry):
        """@event without hook parameters does not register in hooks registry."""

        @event
        class PlainEvent:
            def execute(self, sim):
                pass

        hooks = get_event_hooks()
        assert "plain_event" not in hooks

    def test_register_event_hook_directly(self, clean_registry):
        """register_event_hook() function works directly."""

        @event
        class MyEvent:
            def execute(self, sim):
                pass

        register_event_hook("my_event", after="some_target")

        hooks = get_event_hooks()
        assert "my_event" in hooks
        assert hooks["my_event"]["after"] == "some_target"

    def test_register_event_hook_validation(self, clean_registry):
        """register_event_hook() validates hook types."""
        with pytest.raises(ValueError, match="multiple hook types"):
            register_event_hook("event", after="a", before="b")


class TestPipelineHookApplication:
    """Tests for automatic hook application during pipeline creation."""

    def test_after_hook_inserts_event(self, clean_registry):
        """Event with after hook is inserted correctly in pipeline."""

        @event
        class EventA:
            def execute(self, sim):
                pass

        @event
        class EventB:
            def execute(self, sim):
                pass

        @event(after="event_a")
        class HookedEvent:
            def execute(self, sim):
                pass

        pipeline = Pipeline.from_event_list(["event_a", "event_b"])

        names = [e.name for e in pipeline.events]
        assert names == ["event_a", "hooked_event", "event_b"]

    def test_before_hook_inserts_event(self, clean_registry):
        """Event with before hook is inserted correctly in pipeline."""

        @event
        class EventA:
            def execute(self, sim):
                pass

        @event
        class EventB:
            def execute(self, sim):
                pass

        @event(before="event_b")
        class HookedEvent:
            def execute(self, sim):
                pass

        pipeline = Pipeline.from_event_list(["event_a", "event_b"])

        names = [e.name for e in pipeline.events]
        assert names == ["event_a", "hooked_event", "event_b"]

    def test_replace_hook_replaces_event(self, clean_registry):
        """Event with replace hook replaces target event."""

        @event
        class OriginalEvent:
            def execute(self, sim):
                pass

        @event
        class EventB:
            def execute(self, sim):
                pass

        @event(replace="original_event")
        class ReplacementEvent:
            def execute(self, sim):
                pass

        pipeline = Pipeline.from_event_list(["original_event", "event_b"])

        names = [e.name for e in pipeline.events]
        assert names == ["replacement_event", "event_b"]
        assert "original_event" not in names

    def test_multiple_events_same_target_ordering(self, clean_registry):
        """Multiple events targeting same point maintain registration order."""

        @event
        class TargetEvent:
            def execute(self, sim):
                pass

        @event(after="target_event")
        class HookA:
            def execute(self, sim):
                pass

        @event(after="target_event")
        class HookB:
            def execute(self, sim):
                pass

        pipeline = Pipeline.from_event_list(["target_event"])

        names = [e.name for e in pipeline.events]
        # First registered = closest to target
        assert names == ["target_event", "hook_a", "hook_b"]

    def test_chained_hooks(self, clean_registry):
        """Chained hooks work correctly (B after A, C after B)."""

        @event
        class TargetEvent:
            def execute(self, sim):
                pass

        @event(after="target_event")
        class HookA:
            def execute(self, sim):
                pass

        @event(after="hook_a")
        class HookB:
            def execute(self, sim):
                pass

        pipeline = Pipeline.from_event_list(["target_event"])

        names = [e.name for e in pipeline.events]
        assert names == ["target_event", "hook_a", "hook_b"]

    def test_hook_target_not_in_pipeline_ignored(self, clean_registry):
        """Hooks targeting non-existent events are silently ignored."""

        @event
        class SomeEvent:
            def execute(self, sim):
                pass

        @event(after="nonexistent_event")
        class HookedEvent:
            def execute(self, sim):
                pass

        pipeline = Pipeline.from_event_list(["some_event"])

        names = [e.name for e in pipeline.events]
        assert "hooked_event" not in names
        assert names == ["some_event"]

    def test_apply_hooks_false_skips_hooks(self, clean_registry):
        """Pipeline created with apply_hooks=False skips hook application."""

        @event
        class BaseEvent:
            def execute(self, sim):
                pass

        @event(after="base_event")
        class HookedEvent:
            def execute(self, sim):
                pass

        pipeline = Pipeline.from_event_list(["base_event"], apply_hooks=False)

        names = [e.name for e in pipeline.events]
        assert names == ["base_event"]
        assert "hooked_event" not in names

    def test_event_already_in_pipeline_not_duplicated(self, clean_registry):
        """If hooked event is already in pipeline, it's not inserted again."""

        @event
        class EventA:
            def execute(self, sim):
                pass

        @event(after="event_a")
        class EventB:
            def execute(self, sim):
                pass

        # EventB is both in the list AND has a hook
        pipeline = Pipeline.from_event_list(["event_a", "event_b"])

        names = [e.name for e in pipeline.events]
        # Should not have duplicate event_b
        assert names.count("event_b") == 1

    def test_replace_hook_target_not_in_pipeline_ignored(self, clean_registry):
        """Replace hook for non-existent target is silently ignored."""

        @event
        class SomeEvent:
            def execute(self, sim):
                pass

        @event(replace="nonexistent_event")
        class ReplacementEvent:
            def execute(self, sim):
                pass

        pipeline = Pipeline.from_event_list(["some_event"])

        names = [e.name for e in pipeline.events]
        assert names == ["some_event"]
        assert "replacement_event" not in names


class TestPipelineFromYamlWithHooks:
    """Tests for hook application with Pipeline.from_yaml()."""

    def test_from_yaml_applies_hooks_by_default(self, clean_registry, tmp_path):
        """Pipeline.from_yaml() applies hooks by default."""

        @event
        class EventA:
            def execute(self, sim):
                pass

        @event(after="event_a")
        class HookedEvent:
            def execute(self, sim):
                pass

        yaml_file = tmp_path / "pipeline.yml"
        yaml_file.write_text("events:\n  - event_a\n")

        pipeline = Pipeline.from_yaml(yaml_file)

        names = [e.name for e in pipeline.events]
        assert names == ["event_a", "hooked_event"]

    def test_from_yaml_apply_hooks_false(self, clean_registry, tmp_path):
        """Pipeline.from_yaml(apply_hooks=False) skips hooks."""

        @event
        class EventA:
            def execute(self, sim):
                pass

        @event(after="event_a")
        class HookedEvent:
            def execute(self, sim):
                pass

        yaml_file = tmp_path / "pipeline.yml"
        yaml_file.write_text("events:\n  - event_a\n")

        pipeline = Pipeline.from_yaml(yaml_file, apply_hooks=False)

        names = [e.name for e in pipeline.events]
        assert names == ["event_a"]
        assert "hooked_event" not in names
