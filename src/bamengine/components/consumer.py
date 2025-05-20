# src/bamengine/components/consumer.py
from dataclasses import dataclass

from bamengine.typing import Float1D


@dataclass(slots=True)
class Consumer:
    """Dense state for *all* consumers."""

    income: Float1D  # disposable income y_h  (accumulates wages)
