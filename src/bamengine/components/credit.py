# src/bamengine/components/credit.py
from dataclasses import dataclass, field
import numpy as np
from bamengine.typing import FloatA, IntA

@dataclass(slots=True)
class LoanBook:
    """
    Edge-list ledger of *active* loans.
    Grows automatically; no per-step allocations in hot loops.

    ---

    **Designing loan ledger system**

    ### You want **per-loan detail** ( *who* ↔ *whom* ↔ *how much* ↔ *rate* ).

    A dense `(N_firms × N_banks)` matrix explodes in RAM and is awkward to update, so the usual remedy is an **edge-list ledger** that sits *once* in memory and is shared by all credit-market systems.

    ---

    ## Data-layout: a flat “COO” edge list

    | field name  | dtype     | meaning                               | remarks                                                  |
    | ----------- | --------- | ------------------------------------- | -------------------------------------------------------- |
    | `firm`      | `int64`   | index *i* of borrowing firm           | 0 … N-1                                                  |
    | `bank`      | `int64`   | index *k* of lending bank             | 0 … K-1                                                  |
    | `principal` | `float64` | original loan amount *L<sub>ik</sub>* | cumulative top-ups simply **append** new rows            |
    | `rate`      | `float64` | contractual *r<sub>ik</sub>*          | immutable after signing                                  |
    | `interest`  | `float64` | *r · L* (cached)                      | lets you sum interests in **O(1)** without recalculation |
    | `debt`      | `float64` | *L × (1 + r)*                         | idem                                                     |

    *All six columns are 1-D NumPy arrays of equal length **M** (number of active loans).*

    ### Why this layout?

    * **Sparse:** you store only existing contracts – memory ∝ *loans*, not *firms × banks*.
      `M ≪ N·K` in almost every macro model.
    * **Vectorisable:** aggregations are `np.bincount` or `np.ufunc.at`, e.g.

      ```python
      firm_debt = np.zeros(N)
      np.add.at(firm_debt, ledger.firm, ledger.debt)
      ```

      → \~ 70 ns per loan on CPython/NumPy.
    * **Append-only write pattern:** adding a loan = `idx = len(firm)` → extend by 1.
      No in-place reshuffle, so you avoid cache-thrashing during the inner loop.
    * **Easy resize-to-fit:** start with capacity `≈ N_firms * H` and `np.resize` (2×) when full, amortised O(1).

    """
    firm:      IntA     = field(default_factory=lambda: np.empty(0, np.int64))
    bank:      IntA     = field(default_factory=lambda: np.empty(0, np.int64))
    principal: FloatA   = field(default_factory=lambda: np.empty(0, np.float64))
    rate:      FloatA   = field(default_factory=lambda: np.empty(0, np.float64))
    interest:  FloatA   = field(default_factory=lambda: np.empty(0, np.float64))
    debt:      FloatA   = field(default_factory=lambda: np.empty(0, np.float64))
    capacity:  int      = 128        # current physical length
    size:      int      = 0          # number of *filled* rows
