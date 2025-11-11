"""Example demonstrating the @role and @event decorators.

This example shows how users can define custom roles and events using the
simplified decorator syntax. NO INHERITANCE NEEDED!
"""

import numpy as np

from bamengine import *


# ============================================================================
# Example 1: Define a custom Role using @role decorator (NO INHERITANCE!)
# ============================================================================


@role
class Inventor:  # ← No (Role) inheritance needed!
    """Custom role tracking innovation capacity of agents."""

    innovation_score: Float
    patents: Int
    research_budget: Float


# Alternative: With custom name
@role(name="CustomInventor")
class InventorV2:  # ← No (Role) inheritance needed!
    """Custom role with explicit name."""

    innovation_score: Float
    patents: Int


# Traditional syntax still works if you prefer explicit inheritance
@role
class InventorV3(Role):
    """Traditional syntax with explicit Role inheritance."""

    innovation_score: Float
    patents: Int


# ============================================================================
# Example 2: Define a custom Event using @event decorator (NO INHERITANCE!)
# ============================================================================


@event
class InnovationEvent:  # ← No (Event) inheritance needed!
    """Custom event that processes innovation."""

    def execute(self, sim: Simulation) -> None:
        """Execute innovation logic."""
        # Access custom role (would need to be added to simulation first)
        print(f"Executing {self.name}")  # TODO Unresolved attribute reference 'name' for class 'InnovationEvent'
        # Custom logic here...


# Alternative: With custom name
@event(name="my_innovation_event")
class InnovationEventV2:  # ← No (Event) inheritance needed!
    """Custom event with explicit name."""

    def execute(self, sim: Simulation) -> None:
        """Execute innovation logic."""
        print(f"Executing {self.name}")  # TODO Unresolved attribute reference 'name' for class 'InnovationEventV2'


# ============================================================================
# Example 3: Verify registration
# ============================================================================

if __name__ == "__main__":
    # List all registered roles
    print("Registered Roles:")
    for role_name in list_roles():
        print(f"  - {role_name}")

    print("\nOur custom roles:")
    print(f"  - Inventor: {get_role('Inventor')}")
    print(f"  - CustomInventor: {get_role('CustomInventor')}")

    # List all registered events
    print("\nSome registered Events (first 5):")
    for event_name in sorted(list_events())[:5]:
        print(f"  - {event_name}")

    print("\nOur custom events:")
    print(f"  - innovation_event: {get_event('innovation_event')}")
    print(f"  - my_innovation_event: {get_event('my_innovation_event')}")

    # Instantiate custom role
    print("\nInstantiating Inventor role:")
    inventor = Inventor(
        innovation_score=np.array([0.5, 0.8, 0.3]),
        patents=np.array([2, 5, 1], dtype=np.int64),
        research_budget=np.array([10000.0, 25000.0, 5000.0]),
    )
    print(f"  {inventor}")
    print(f"  Innovation scores: {inventor.innovation_score}")
    print(f"  Patents: {inventor.patents}")

    # Instantiate custom event
    print("\nInstantiating InnovationEvent:")
    event = InnovationEvent()
    print(f"  {event}")
    print(f"  Event name: {event.name}")  # TODO Unresolved attribute reference 'name' for class 'InnovationEvent'

    print("\n✅ All examples completed successfully!")
