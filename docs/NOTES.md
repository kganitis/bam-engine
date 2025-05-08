
## Documentation

1. **Docstrings**: adopt the *NumPy docstring standard* everywhere and mark `out` parameters (`← out`) consistently (your planning docstrings already follow this).
2. **Lint noise**: Ruff config selects only `E, F, W, I`. Consider adding `D` (docstring) once code stabilises.

---

## Performance quick-wins

* `workers_prepare_applications`: use permanent scratch buffers

---

## Later, but keep in mind

* **Parallel execution**: When n\_firms >> 1M, you’ll want *threaded* NumPy or distribute firms across processes. Design data layout (SoA vs AoS) accordingly.
* **Streaming / State IO**: Provide `Scheduler.to_dict()` / `from_dict()` for checkpoint‑restart (important when running hundreds of Monte‑Carlo paths).
* **Interfaces**: Expose a light **`pandas` view** (DataFrame per component) only for *read‑only* downstream analysis; keep core arrays untouched.
* **Policy experiments**: The hook mechanism can evolve into **event bus**; consider `Plugin` objects with `on_event(self, scheduler, tag)` for user‑defined rules.

---

## Small items you can safely defer

| Area                                                 | Why it can wait                                      | Concrete follow‑up when convenient                                                 |
|------------------------------------------------------| ---------------------------------------------------- |------------------------------------------------------------------------------------|
| **Docstrings / API docs**                            | Developer‑facing code is still in flux.              | Finish NumPy‑style docstrings once the full event chain is in.                     |
| **Version bump**                                     | No external users yet.                               | Switch to `0.1.0` the moment you publish the first PyPI wheel.                  |
| **Performance regression**                           | No timing guard yet.                                 | Create a **`pytest‑benchmark`** suite later; fail if step‑time > xms for base size.|
| **Numba JIT**                                        | Profile first‑—CPU is nowhere near a bottleneck yet. | Add `@njit(cache=True)` to pure kernels *after* the rule set stabilises.           |
| **Logging cost in tight loops**                      | Overhead ≈ 0 ms at current sizes.             | Guard heavy `log.debug(...)` blocks with `if log.isEnabledFor(...)`.               |
| **`np.ceil` optimisation in `decide_desired_labor`** | Only 10 µs for 100 k firms.            | Implement the branch‑free floor‑divide trick when you measure >5% runtime there.   |
| **Checkpoint I/O (`snapshot()`)**                    | Not needed until you run long Monte‑Carlo batches.   | Design a simple `dict`/NPZ serialiser later.                                       |


## `sign_loan_with_bank` implementation


Good catch — that *signature* logic hasn’t vanished; it’s just been **pared‑down** to the absolute minimum required for the
next step.

### Where it went

* In the OOP code, `sign_loan_with_bank(…)` **mutates four things** inside the
  firm:

  | purpose                  | field updated               |
  | ------------------------ | --------------------------- |
  | book‑keeping of the deal | `loan_data[bank] = {…}`     |
  | reduce unmet demand      | `credit_demand_B -= amount` |
  | raise cash               | `total_funds += amount`     |
  | logging                  | side‑effect                 |

* In the ECS draft, `banks_grant_loans(…)` **already performs the second point** (it decreases `fp.credit_demand` and
  the bank’s `credit_supply`) – that’s what lets the round‑robin continue
  without reallocations.

* What it does **not** yet store is the **per‑bank, per‑firm loan ledger** and
  the **cash‑injection onto the firm’s balance sheet**.
  I left that out intentionally because:

  1. It isn’t needed until we implement the **profit / balance‑sheet** event
     (or wages if you want workers to be fired immediately).
  2. A sparse “matrix” of loans (many firms × many banks, most entries zero)
     needs a design decision: *dense arrays* (fast but memory‑heavy) vs
     *edge‑list* style (indices + amounts). I didn’t want to pre‑empt that
     choice without your input.

### Two easy ways to restore full fidelity

| option                      | how                                                                                                                                                                              | pros                                   | cons                                                                                              |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Edge list** (recommended) | Add two 1‑D growing arrays on a separate component:<br>`loan_firm_idx`, `loan_bank_idx`, `loan_amount`, `loan_rate`, `loan_interest`…<br>Append rows inside `banks_grant_loans`. | Sparse, flexible, no giant N×K matrix. | Requires tiny helper to filter/aggregate later.                                                   |
| **Dense cube**              | A tensor `(N_firms, N_banks)` for `loan_amount` and one for `rate`.<br> Update by simple indexing.                                                                               | Pure NumPy, very fast.                 | 100 firms × 10 banks ⇒ 1 000 floats ≈ OK, but 1 000 × 100 ⇒ 100 000; can explode if you scale up. |

Both keep the *permanent‑buffer* ethos: allocate once in `Scheduler.init()`.

### Suggested incremental plan

1. **Stay minimal** for the first integration pass:
   Keep `banks_grant_loans` as‑is so you can run the full period loop.
2. **Add the ledger** once we reach the *balance‑sheet / profit* event:
   we’ll pick the edge‑list approach, add two small write helpers, and extend the
   component dataclasses.
3. **Cash injection** (`total_funds += amount`) can be a simple shared view:
   expose a `firm_cash` array inside `FirmCreditPlan` (or reuse `net_worth`
   proxy) and increment it in place right after reducing `credit_demand`.

### Quick patch if you already need the cash field

```python
# in FirmCreditDemand
cash: FloatA                # total_funds in the OOP code
```

And inside `banks_grant_loans` add:

```python
fp.cash[f] += amount
```

That restores the most urgent side‑effect without designing the ledger yet.

---

**TL;DR** – the “sign” side‑effects are deliberately trimmed; we’ll extend them
when we implement accounting. Let me know which ledger style you prefer and
I’ll wire it up.


## Review of the “DTO‑per‑system” component style

*(one component == exactly the fields a **single** system needs)*

| Pillar          | Upsides                                                                                                                                                                                                          | Draw‑backs & guard‑rails                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Performance** | *➕ Cache‑friendly:* each system touches a tight SoA block, nothing extra.<br>*➕ No scatter updates:* every write is contiguous (`out=`).                                                                         | *➖ Fragmentation:* the **same physical array** (`credit_demand`) is now duplicated in three separate dataclasses.  NumPy will happily store three *views* to the same buffer, **but only if** you pass the same object around.  The moment you call `np.copy()` by mistake you double the RAM footprint.<br>*➖ Many tiny attribute look‑ups:* Python has to deref several objects per system; a single dataclass holding all firm credit fields is marginally faster. |
| **Testability** | *➕ Very explicit contracts*: a unit test can construct the exact component and ignore everything else.<br>\*➕ Fakes are tiny — hand‑crafted arrays with only a few fields.                                       | *➖ Harder invariants:* cross‑system invariants (e.g. `credit_supply` ≥ 0 **and** queue cleared) now span multiple components; property‑based tests need to stitch them together each time.                                                                                                                                                                                                                                                                            |
| **Library UX**  | *➕ Shallow public surface:* down‑stream users import only the systems they care about.<br>*➕ No accidental mutation:* a user cannot poke `banks.contract_rate` because it lives only in the temporary component. | *➖ Discoverability:* a user looking for “the credit demand array” must know it lives in **three** different components.<br>*➖ Version churn:* every time a system wants one extra field you create/rename a dataclass, creating noisy API diffs.                                                                                                                                                                                                                      |

**Verdict:**
The “DTO per system” style is totally viable **inside** an internal module, but
for a public library I’d recommend a *hybrid*:

* **Domain‑oriented primary components** – one `FirmFinance`, one `BankFinance`, …
  holding the authoritative arrays.
* **Scratch/alias objects** created *inside* the system as *light wrappers* that
  expose only the needed views.  (They are never exported, so API stability is
  preserved.)

That gives you the \**performance* of thin structs (they’re just
`__slots__` objects pointing at the same NumPy buffers) **and** the
**ergonomics** of a stable public datamodel.
