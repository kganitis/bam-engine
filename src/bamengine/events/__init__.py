"""Event classes for BAM Engine ECS architecture.

This module contains Event classes that wrap existing system functions.
Events are auto-registered via __init_subclass__ hook and can be
composed into a Pipeline for execution.

Each event module corresponds to a system module:
- planning.py → wraps systems/planning.py
- labor_market.py → wraps systems/labor_market.py
- credit_market.py → wraps systems/credit_market.py
- production.py → wraps systems/production.py
- goods_market.py → wraps systems/goods_market.py
- revenue.py → wraps systems/revenue.py
- bankruptcy.py → wraps systems/bankruptcy.py
"""

__all__ = []
