"""Extensions package for BAM Engine.

This package provides optional extensions that add functionality
beyond the core BAM model.

Available extensions:
    - rnd: R&D investment and endogenous productivity growth (Section 3.8)
    - buffer_stock: Buffer-stock consumption with adaptive MPC (Section 3.9.4)
"""

from __future__ import annotations

__all__ = ["rnd", "buffer_stock"]
