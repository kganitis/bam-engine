"""Unit tests for event pipeline hooks."""

import pytest

from bamengine.core.decorators import event
from bamengine.core.pipeline import Pipeline


class TestEventHookRegistration:
    """Tests for event hook metadata stored on class by decorator."""

    def test_event_with_after_hook_sets_class_attribute(self, clean_registry):
        """@event(after=...) stores _hook_after on the class."""

        @event
        class TargetEvent:
            def execute(self, sim):
                pass

        @event(after="target_event")
        class HookedEvent:
            def execute(self, sim):
                pass

        assert HookedEvent._hook_after == "target_event"
        assert HookedEvent._hook_before is None
        assert HookedEvent._hook_replace is None

    def test_event_with_before_hook_sets_class_attribute(self, clean_registry):
        """@event(before=...) stores _hook_before on the class."""

        @event
        class TargetEvent:
            def execute(self, sim):
                pass

        @event(before="target_event")
        class HookedEvent:
            def execute(self, sim):
                pass

        assert HookedEvent._hook_before == "target_event"
        assert HookedEvent._hook_after is None
        assert HookedEvent._hook_replace is None

    def test_event_with_replace_hook_sets_class_attribute(self, clean_registry):
        """@event(replace=...) stores _hook_replace on the class."""

        @event
        class TargetEvent:
            def execute(self, sim):
                pass

        @event(replace="target_event")
        class ReplacementEvent:
            def execute(self, sim):
                pass

        assert ReplacementEvent._hook_replace == "target_event"
        assert ReplacementEvent._hook_after is None
        assert ReplacementEvent._hook_before is None

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

        assert HookedEvent._hook_after == "target_event"
        assert HookedEvent.name == "my_custom_event"

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

    def test_event_without_hook_has_no_hook_attributes(self, clean_registry):
        """@event without hook parameters does not set hook attributes."""

        @event
        class PlainEvent:
            def execute(self, sim):
                pass

        assert not hasattr(PlainEvent, "_hook_after")
        assert not hasattr(PlainEvent, "_hook_before")
        assert not hasattr(PlainEvent, "_hook_replace")


class TestPipelineApplyHooks:
    """Tests for Pipeline.apply_hooks() method."""

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
        pipeline.apply_hooks(HookedEvent)

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
        pipeline.apply_hooks(HookedEvent)

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
        pipeline.apply_hooks(ReplacementEvent)

        names = [e.name for e in pipeline.events]
        assert names == ["replacement_event", "event_b"]
        assert "original_event" not in names

    def test_multiple_events_same_target_ordering(self, clean_registry):
        """Multiple events targeting same point maintain argument order."""

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
        pipeline.apply_hooks(HookA, HookB)

        names = [e.name for e in pipeline.events]
        # First in argument list = closest to target
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
        pipeline.apply_hooks(HookA, HookB)

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
        pipeline.apply_hooks(HookedEvent)

        names = [e.name for e in pipeline.events]
        assert "hooked_event" not in names
        assert names == ["some_event"]

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
        pipeline.apply_hooks(EventB)

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
        pipeline.apply_hooks(ReplacementEvent)

        names = [e.name for e in pipeline.events]
        assert names == ["some_event"]
        assert "replacement_event" not in names

    def test_class_without_hook_metadata_silently_skipped(self, clean_registry):
        """Classes without hook metadata are silently skipped."""

        @event
        class EventA:
            def execute(self, sim):
                pass

        @event
        class PlainEvent:
            def execute(self, sim):
                pass

        pipeline = Pipeline.from_event_list(["event_a"])
        pipeline.apply_hooks(PlainEvent)

        names = [e.name for e in pipeline.events]
        assert names == ["event_a"]

    def test_pipeline_from_event_list_does_not_auto_apply_hooks(self, clean_registry):
        """Pipeline creation does not auto-apply hooks."""

        @event
        class BaseEvent:
            def execute(self, sim):
                pass

        @event(after="base_event")
        class HookedEvent:
            def execute(self, sim):
                pass

        pipeline = Pipeline.from_event_list(["base_event"])

        names = [e.name for e in pipeline.events]
        assert names == ["base_event"]
        assert "hooked_event" not in names


class TestPipelineFromYamlWithHooks:
    """Tests for hook application with Pipeline.from_yaml()."""

    def test_from_yaml_does_not_auto_apply_hooks(self, clean_registry, tmp_path):
        """Pipeline.from_yaml() does not auto-apply hooks."""

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
        assert names == ["event_a"]
        assert "hooked_event" not in names

    def test_from_yaml_with_explicit_apply_hooks(self, clean_registry, tmp_path):
        """Pipeline.from_yaml() + apply_hooks() inserts hooked events."""

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
        pipeline.apply_hooks(HookedEvent)

        names = [e.name for e in pipeline.events]
        assert names == ["event_a", "hooked_event"]


class TestSimulationUseEvents:
    """Tests for Simulation.use_events() integration."""

    def test_use_events_applies_hooks_to_pipeline(self):
        """sim.use_events() applies hooks to the simulation pipeline."""
        import bamengine as bam

        @event(after="firms_pay_dividends")
        class CustomPostDividend:
            def execute(self, sim):
                pass

        sim = bam.Simulation.init(n_firms=5, n_households=10, n_banks=2, seed=0)
        sim.use_events(CustomPostDividend)

        names = [e.name for e in sim.pipeline.events]
        # Should be inserted after firms_pay_dividends
        dividends_idx = names.index("firms_pay_dividends")
        custom_idx = names.index("custom_post_dividend")
        assert custom_idx == dividends_idx + 1

    def test_use_events_with_replace(self):
        """sim.use_events() applies replace hooks correctly."""
        import bamengine as bam

        @event(replace="firms_adjust_price")
        class CustomPricing:
            def execute(self, sim):
                pass

        sim = bam.Simulation.init(n_firms=5, n_households=10, n_banks=2, seed=0)
        sim.use_events(CustomPricing)

        names = [e.name for e in sim.pipeline.events]
        assert "custom_pricing" in names
        assert "firms_adjust_price" not in names
