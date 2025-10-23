"""Integration tests for Role and Event registration with BAM components."""

import numpy as np

from bamengine.core import Role
from bamengine.core.registry import list_roles, get_role

# Note: These tests require real BAM components to be in the registry.
# If unit tests run first and clear the registry, components won't
# re-register on import (Python caches modules). Solution: these tests
# re-import components to ensure they're loaded, and manually re-register
# if needed.


# ============================================================================
# Role Registration Tests
# ============================================================================


def test_all_roles_registered():
    """Verify all BAM roles are in registry."""
    # Import all roles to ensure they're loaded
    from bamengine.roles import (
        Producer,
        Employer,
        Borrower,
        Worker,
        Consumer,
        Lender,
    )

    roles = list_roles()

    # If registry was cleared by unit tests, manually re-register
    # (Python doesn't re-execute module code on re-import)
    if "Producer" not in roles:
        import bamengine.core.registry as registry_module

        for role_cls in [Producer, Employer, Borrower, Worker, Consumer, Lender]:
            registry_module._ROLE_REGISTRY[role_cls.name] = role_cls

        roles = list_roles()

    # Check all core agent roles
    assert "Producer" in roles
    assert "Employer" in roles
    assert "Borrower" in roles
    assert "Worker" in roles
    assert "Consumer" in roles
    assert "Lender" in roles


def test_role_instantiation_via_registry():
    """Test creating role instances via registry lookup."""
    from bamengine.roles import Employer

    # Ensure Employer is in registry (in case it was cleared)
    if "Employer" not in list_roles():
        import bamengine.core.registry as registry_module

        registry_module._ROLE_REGISTRY[Employer.name] = Employer

    # Get class from registry
    EmployerCls = get_role("Employer")

    # Should get the same class
    assert EmployerCls is Employer

    # Should be able to instantiate
    emp = EmployerCls(
        wage_offer=np.ones(10),
        current_labor=np.zeros(10, dtype=int),
        desired_labor=np.ones(10, dtype=int),
        wage_bill=np.zeros(10),
        n_vacancies=np.zeros(10, dtype=int),
        total_funds=np.ones(10) * 100,
        recv_job_apps_head=np.zeros(10, dtype=int),
        recv_job_apps=np.zeros((10, 5), dtype=int),
    )

    assert len(emp.wage_offer) == 10
    assert emp.name == "Employer"


def test_role_has_slots():
    """Test that imported roles have slots enabled."""
    from bamengine.roles import Producer

    # Should have __slots__ (via @dataclass(slots=True))
    assert hasattr(Producer, "__slots__")

    # Instance should not have __dict__
    prod = Producer(
        production=np.ones(10),
        inventory=np.zeros(10),
        expected_demand=np.ones(10),
        desired_production=np.ones(10),
        labor_productivity=np.ones(10),
        breakeven_price=np.ones(10),
        price=np.ones(10),
    )

    assert not hasattr(prod, "__dict__"), "Instance has __dict__ - slots not working!"


def test_role_is_dataclass():
    """Test that imported roles are dataclasses."""
    from bamengine.roles import Producer

    assert hasattr(Producer, "__dataclass_fields__")
    assert "__init__" in dir(Producer)  # Has generated __init__


def test_role_inheritance():
    """Verify all roles inherit from Role."""
    from bamengine.roles import (
        Producer,
        Worker,
        Lender,
        Borrower,
        Consumer,
        Employer,
    )

    assert issubclass(Producer, Role)
    assert issubclass(Worker, Role)
    assert issubclass(Lender, Role)
    assert issubclass(Borrower, Role)
    assert issubclass(Consumer, Role)
    assert issubclass(Employer, Role)


# ============================================================================
# Event Registration Tests
# ============================================================================
# NOTE: Event registration tests will be added once wrapper events are
# created in Day 9. These tests will mirror the role tests above:
# - test_all_events_registered()
# - test_event_instantiation_via_registry()
# - test_event_has_slots()
# - test_event_is_dataclass()
# - test_event_inheritance()
