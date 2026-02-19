"""
===========================================
Credit Market Investigation: Root Cause
===========================================

Deep diagnostic investigation of why the BAM baseline credit market becomes
inactive after a brief burn-in period.

**Hypothesis**: Structural profit guarantee (markup × sell-through > 1.0)
combined with 90% profit retention (delta=0.10) causes net worth to accumulate
until NW >> WB, making credit_demand = max(0, WB - NW) permanently zero.

**Method**: Custom event hooks capture per-firm, per-period credit market data.
Eight focused figures map the exact transition from credit-dependent to
self-financing, track new entrant cohorts, and quantify the steady-state
NW/WB attractor.

Output: 8 multi-panel figures saved to ``diagnostics/output/credit_investigation/``.

Run with::

    python diagnostics/credit_market_investigation.py

For non-interactive (CI/headless) usage::

    MPLBACKEND=Agg python diagnostics/credit_market_investigation.py
"""

# %%
# Imports
# -------

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import bamengine as bam
from bamengine import event

# %%
# Constants
# ---------

N_PERIODS = 200
SEED = 42
OUTPUT_DIR = Path(__file__).parent / "output" / "credit_investigation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
C_NW = "#2E86AB"  # Blue - Net worth
C_WB = "#A23B72"  # Magenta - Wage bill
C_CD = "#E74C3C"  # Red - Credit demand
C_CS = "#6A994E"  # Green - Credit supply
C_PROFIT = "#F18F01"  # Orange - Profits
C_BANK = "#0EA5E9"  # Cyan - Bank side
C_NEW = "#8B5CF6"  # Purple - New entrants
C_SELF = "#374151"  # Dark gray - Self-financing

# %%
# Module-Level Data Storage
# -------------------------
#
# Simulation uses __slots__, so we store per-period snapshots here.

# Per-period, per-firm arrays (appended each period)
_nw_history: list[np.ndarray] = []  # net_worth after update
_wb_history: list[np.ndarray] = []  # wage_bill at credit demand time
_tf_history: list[np.ndarray] = []  # total_funds at credit demand time
_cd_history: list[np.ndarray] = []  # credit_demand
_gp_history: list[np.ndarray] = []  # gross_profit
_np_history: list[np.ndarray] = []  # net_profit
_rp_history: list[np.ndarray] = []  # retained_profit
_prod_history: list[np.ndarray] = []  # production
_price_history: list[np.ndarray] = []  # price
_inv_history: list[np.ndarray] = []  # inventory (post-sales)
_labor_history: list[np.ndarray] = []  # current_labor

# Per-period, per-bank arrays
_equity_history: list[np.ndarray] = []  # bank equity
_cs_history: list[np.ndarray] = []  # credit supply
_loans_outstanding: list[int] = []  # LoanBook.size after matching

# Per-period scalars
_n_borrowers: list[int] = []  # firms with credit_demand > 0
_total_cd: list[float] = []  # aggregate credit demand
_total_cs: list[float] = []  # aggregate credit supply
_total_lending: list[float] = []  # actual loans granted this period

# Bankruptcy tracking: (period, firm_index) tuples
_bankruptcies: list[tuple[int, int]] = []
_current_period = 0

# %%
# Data Collection Hooks
# ---------------------


@event(name="capture_credit_demand", after="firms_decide_credit_demand")
class CaptureCreditDemand:
    """Capture credit demand, wage bill, and total funds right after demand decision."""

    def execute(self, sim):
        bor = sim.get_role("Borrower")
        _wb_history.append(bor.wage_bill.copy())
        _tf_history.append(bor.total_funds.copy())
        _cd_history.append(bor.credit_demand.copy())
        _n_borrowers.append(int(np.sum(bor.credit_demand > 1e-10)))
        _total_cd.append(float(bor.credit_demand.sum()))


@event(name="capture_credit_supply", after="banks_decide_credit_supply")
class CaptureCreditSupply:
    """Capture credit supply from banks."""

    def execute(self, sim):
        lend = sim.get_role("Lender")
        _cs_history.append(lend.credit_supply.copy())
        _total_cs.append(float(lend.credit_supply.sum()))


@event(name="capture_loans", after="firms_fire_workers")
class CaptureLoans:
    """Capture loan book state after all credit matching rounds complete."""

    def execute(self, sim):
        lb = sim.get_relationship("LoanBook")
        _loans_outstanding.append(lb.size)
        # Total lending = sum of all loan principals this period
        if lb.size > 0:
            _total_lending.append(float(lb.principal[: lb.size].sum()))
        else:
            _total_lending.append(0.0)
        # Also capture bank equity after lending
        lend = sim.get_role("Lender")
        _equity_history.append(lend.equity_base.copy())


@event(name="capture_production", after="firms_run_production")
class CaptureProduction:
    """Capture production and labor data."""

    def execute(self, sim):
        prod = sim.get_role("Producer")
        emp = sim.get_role("Employer")
        _prod_history.append(prod.production.copy())
        _price_history.append(prod.price.copy())
        _labor_history.append(emp.current_labor.copy())


@event(name="capture_revenue", after="firms_pay_dividends")
class CaptureRevenue:
    """Capture profit components after dividend distribution."""

    def execute(self, sim):
        bor = sim.get_role("Borrower")
        _gp_history.append(bor.gross_profit.copy())
        _np_history.append(bor.net_profit.copy())
        _rp_history.append(bor.retained_profit.copy())
        prod = sim.get_role("Producer")
        _inv_history.append(prod.inventory.copy())


@event(name="capture_net_worth", after="firms_update_net_worth")
class CaptureNetWorth:
    """Capture net worth after bankruptcy-phase update."""

    def execute(self, sim):
        bor = sim.get_role("Borrower")
        _nw_history.append(bor.net_worth.copy())


@event(name="capture_bankruptcies", after="mark_bankrupt_firms")
class CaptureBankruptcies:
    """Record which firms went bankrupt and when."""

    def execute(self, sim):
        global _current_period
        exiting = sim.ec.exiting_firms
        for idx in exiting:
            _bankruptcies.append((_current_period, int(idx)))
        _current_period += 1


# %%
# Helper Functions
# ----------------


def style_ax(ax, title, xlabel="Period", ylabel=""):
    """Apply consistent styling to an axis."""
    ax.set_title(title, fontsize=11, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)


def save_figure(fig, fig_num, name):
    """Save figure and show."""
    filename = f"credit_{fig_num:02d}_{name}.png"
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved {filename}")


# %%
# Run Simulation
# --------------

print("=" * 60)
print("Credit Market Investigation")
print("=" * 60)

sim = bam.Simulation.init(seed=SEED, logging={"default_level": "WARNING"})
sim.use_events(
    CaptureCreditSupply,
    CaptureCreditDemand,
    CaptureLoans,
    CaptureProduction,
    CaptureRevenue,
    CaptureNetWorth,
    CaptureBankruptcies,
)

n_firms = sim.n_firms
n_banks = sim.n_banks
config = sim.config

print(f"Config: {n_firms} firms, {sim.n_households} households, {n_banks} banks")
print(f"  delta={config.delta}, r_bar={config.r_bar}, v={config.v}")
print(f"  labor_productivity={config.labor_productivity}")
print(f"  new_firm_size_factor={config.new_firm_size_factor}")
print(f"Running {N_PERIODS} periods with seed={SEED}...")

results = sim.run(n_periods=N_PERIODS, collect=True)
actual_periods = results.metadata["n_periods"]
print(
    f"Completed: {actual_periods} periods in {results.metadata['runtime_seconds']:.2f}s"
)

if actual_periods < N_PERIODS:
    print(f"WARNING: Simulation ended early at period {actual_periods}")
    N_PERIODS = actual_periods

# %%
# Convert to 2D arrays
# --------------------

nw = np.array(_nw_history)  # (n_periods, n_firms)
wb = np.array(_wb_history)
tf = np.array(_tf_history)
cd = np.array(_cd_history)
gp = np.array(_gp_history)
np_arr = np.array(_np_history)
rp = np.array(_rp_history)
prod = np.array(_prod_history)
prices = np.array(_price_history)
inv = np.array(_inv_history)
labor = np.array(_labor_history)
equity = np.array(_equity_history)
cs = np.array(_cs_history)

t = np.arange(N_PERIODS)

# %%
# Compute Key Derived Metrics
# ---------------------------

# NW/WB ratio — only meaningful for firms with employees (WB > 0)
# Note: nw is captured at end-of-period (after firms_update_net_worth),
# while wb/tf are captured at credit-market time (after firms_decide_credit_demand).
# For self-financing analysis, use tf (total_funds at credit time) vs wb.
has_employees = wb > 1e-10
safe_wb = np.where(has_employees, wb, 1.0)
nw_wb_ratio = np.where(has_employees, nw / safe_wb, np.nan)
tf_wb_ratio = np.where(has_employees, tf / safe_wb, np.nan)

# Fraction of firms with employees that are self-financing at credit time (TF >= WB)
n_with_employees = np.sum(has_employees, axis=1).astype(float)
n_self_financing_emp = np.sum(has_employees & (tf >= wb), axis=1).astype(float)
safe_n_emp = np.where(n_with_employees > 0, n_with_employees, 1.0)
frac_self_financing = np.where(
    n_with_employees > 0, n_self_financing_emp / safe_n_emp, 1.0
)

# Fraction of firms that are credit-dependent (among all firms)
frac_borrowing = np.mean(cd > 1e-10, axis=1)

# Revenue per firm: price × (production - inventory_remaining)
# We approximate: units_sold ≈ production - post-sales inventory
units_sold = prod - inv
revenue = prices * np.maximum(units_sold, 0)

# Sell-through ratio
safe_prod = np.where(prod > 1e-10, prod, 1e-10)
sell_through = np.where(prod > 1e-10, np.maximum(units_sold, 0) / safe_prod, 0.0)

# %%
# Console Summary
# ---------------

print("\n" + "=" * 60)
print("ANALYTICAL SUMMARY")
print("=" * 60)

# Transition timeline — skip initial periods where few firms have employees
# Find first period where at least 50% of firms have WB > 0 (labor market active)
labor_active = n_with_employees >= n_firms * 0.5
first_active = int(np.argmax(labor_active)) if np.any(labor_active) else 0

# Only compute milestones from the labor-market-active phase onward
frac_post_active = frac_self_financing[first_active:]
pct_50 = (
    int(np.argmax(frac_post_active >= 0.50)) + first_active
    if np.any(frac_post_active >= 0.50)
    else -1
)
pct_90 = (
    int(np.argmax(frac_post_active >= 0.90)) + first_active
    if np.any(frac_post_active >= 0.90)
    else -1
)
pct_99 = (
    int(np.argmax(frac_post_active >= 0.99)) + first_active
    if np.any(frac_post_active >= 0.99)
    else -1
)

print("\nLabor Market Activation:")
print(f"  First period with 50%+ firms employed: {first_active}")
print(
    f"  Firms with employees at t={first_active}: "
    f"{int(n_with_employees[first_active])}/{n_firms}"
)

print("\nTransition to Self-Financing (among firms with employees):")
print(f"  50% self-financing at period: {pct_50}")
print(f"  90% self-financing at period: {pct_90}")
print(f"  99% self-financing at period: {pct_99}")
print(f"  Final: {frac_self_financing[-1] * 100:.1f}% self-financing")

# Steady-state NW/WB (excluding firms with no employees)
late = slice(max(N_PERIODS - 50, 0), N_PERIODS)
median_nw_wb_late = np.nanmedian(nw_wb_ratio[late])
mean_nw_wb_late = np.nanmean(nw_wb_ratio[late])
print("\nSteady-State NW/WB Ratio (last 50 periods):")
print(f"  Median: {median_nw_wb_late:.1f}x")
print(f"  Mean: {mean_nw_wb_late:.1f}x")

# Structural profit analysis
mean_markup = np.mean(
    prices[late] / np.where(wb[late] > 0, wb[late] / safe_prod[late], 1.0)
)
mean_sell_through = np.mean(sell_through[late])
mean_retain = 1.0 - config.delta
structural_multiplier = mean_sell_through * mean_retain
print("\nStructural Profit Guarantee:")
print(f"  Mean sell-through: {mean_sell_through:.3f}")
print(f"  Retention rate: {mean_retain:.2f}")
print(f"  Sell-through × retention: {structural_multiplier:.3f}")

# Credit activity
print("\nCredit Activity:")
print(f"  Total borrowers (period 0): {_n_borrowers[0]}")
print(f"  Total borrowers (final): {_n_borrowers[-1]}")
print(f"  Total bankruptcies: {len(_bankruptcies)}")
print(
    f"  Periods with any borrowing: {sum(1 for n in _n_borrowers if n > 0)}/{N_PERIODS}"
)
print(
    f"  Periods with any loans: {sum(1 for n in _loans_outstanding if n > 0)}/{N_PERIODS}"
)

print("\nGenerating 8 diagnostic figures...")

# %%
# Figure 1: NW/WB Ratio Evolution
# --------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Figure 1: NW/WB Ratio Evolution — The Self-Financing Attractor",
    fontsize=13,
    fontweight="bold",
    y=1.0,
)

# (0,0) Aggregate NW vs WB
ax = axes[0, 0]
ax.plot(t, np.mean(nw, axis=1), color=C_NW, linewidth=1.5, label="Mean Net Worth")
ax.plot(t, np.mean(wb, axis=1), color=C_WB, linewidth=1.5, label="Mean Wage Bill")
ax.legend(fontsize=8)
ax.set_yscale("log")
style_ax(ax, "Mean Net Worth vs Wage Bill (log)", ylabel="Amount")

# (0,1) NW/WB ratio percentiles
ax = axes[0, 1]
pcts = [10, 25, 50, 75, 90]
colors_pct = plt.cm.Blues(np.linspace(0.3, 0.9, len(pcts)))
for i, p in enumerate(pcts):
    vals = np.nanpercentile(nw_wb_ratio, p, axis=1)
    ax.plot(t, vals, color=colors_pct[i], linewidth=1.2, label=f"p{p}")
ax.axhline(1.0, color="red", linewidth=1, linestyle="--", alpha=0.7, label="NW=WB")
ax.legend(fontsize=7, ncol=2)
ax.set_yscale("log")
style_ax(ax, "NW/WB Ratio Percentiles (log)", ylabel="Ratio")

# (1,0) NW/WB ratio median with IQR band
ax = axes[1, 0]
p25 = np.nanpercentile(nw_wb_ratio, 25, axis=1)
p50 = np.nanpercentile(nw_wb_ratio, 50, axis=1)
p75 = np.nanpercentile(nw_wb_ratio, 75, axis=1)
ax.fill_between(t, p25, p75, color=C_NW, alpha=0.2, label="IQR")
ax.plot(t, p50, color=C_NW, linewidth=1.5, label="Median")
ax.axhline(1.0, color="red", linewidth=1, linestyle="--", alpha=0.7)
ax.legend(fontsize=8)
style_ax(ax, "NW/WB Ratio — Median + IQR", ylabel="Ratio")

# (1,1) Distribution of NW/WB at key periods
ax = axes[1, 1]
snapshot_periods = [0, 5, 10, 20, 50, N_PERIODS - 1]
snapshot_periods = [p for p in snapshot_periods if p < N_PERIODS]
colors_snap = plt.cm.viridis(np.linspace(0.15, 0.85, len(snapshot_periods)))
for i, p in enumerate(snapshot_periods):
    vals = nw_wb_ratio[p]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        continue
    vals = vals[vals < np.percentile(vals, 99)]
    if len(vals) > 3:
        ax.hist(
            vals,
            bins=20,
            alpha=0.4,
            color=colors_snap[i],
            label=f"t={p}",
            density=True,
            edgecolor="none",
        )
ax.axvline(1.0, color="red", linewidth=1, linestyle="--", alpha=0.7)
ax.legend(fontsize=7)
style_ax(
    ax, "NW/WB Distribution at Key Periods", xlabel="NW/WB Ratio", ylabel="Density"
)

plt.tight_layout()
save_figure(fig, 1, "nw_wb_ratio")

# %%
# Figure 2: Credit Demand Decomposition
# --------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Figure 2: Credit Demand Decomposition — Who Borrows and Why?",
    fontsize=13,
    fontweight="bold",
    y=1.0,
)

# Build per-period borrower classification
# Classify each firm-period: "new_entrant" (post-bankruptcy, NW reset),
# "post_loss" (NW dropped since last period), "self_financing" (no need)
bankrupt_set = set()
for period, idx in _bankruptcies:
    bankrupt_set.add((period, idx))

n_new_entrant_borrowers = np.zeros(N_PERIODS)
n_post_loss_borrowers = np.zeros(N_PERIODS)
n_mature_borrowers = np.zeros(N_PERIODS)
n_self_financing = np.zeros(N_PERIODS)

# Track which firms were recently spawned (within 3 periods of bankruptcy)
recent_spawn = {}  # firm_idx -> period_spawned

for period in range(N_PERIODS):
    # Update spawn tracking from bankruptcies
    for bp, bi in _bankruptcies:
        if bp == period:
            recent_spawn[bi] = period

    for firm in range(n_firms):
        if cd[period, firm] > 1e-10:
            # This firm is borrowing
            if firm in recent_spawn and (period - recent_spawn[firm]) <= 3:
                n_new_entrant_borrowers[period] += 1
            elif period > 0 and nw[period, firm] < nw[max(period - 1, 0), firm] * 0.95:
                n_post_loss_borrowers[period] += 1
            else:
                n_mature_borrowers[period] += 1
        else:
            n_self_financing[period] += 1

# (0,0) Stacked area: borrower types
ax = axes[0, 0]
ax.stackplot(
    t,
    n_new_entrant_borrowers,
    n_post_loss_borrowers,
    n_mature_borrowers,
    labels=["New Entrants", "Post-Loss", "Mature"],
    colors=[C_NEW, C_CD, C_WB],
    alpha=0.7,
)
ax.legend(fontsize=8, loc="upper right")
style_ax(ax, "Borrower Count by Type", ylabel="# Firms Borrowing")

# (0,1) Self-financing vs borrowing
ax = axes[0, 1]
ax.fill_between(t, 0, n_self_financing, color=C_SELF, alpha=0.5, label="Self-Financing")
total_borrowing = n_new_entrant_borrowers + n_post_loss_borrowers + n_mature_borrowers
ax.fill_between(
    t,
    n_self_financing,
    n_self_financing + total_borrowing,
    color=C_CD,
    alpha=0.5,
    label="Borrowing",
)
ax.legend(fontsize=8)
style_ax(ax, "Self-Financing vs Borrowing Firms", ylabel="# Firms")

# (1,0) Total credit demand over time
ax = axes[1, 0]
ax.plot(t, _total_cd, color=C_CD, linewidth=1.5, label="Total Demand")
ax.plot(t, _total_cs, color=C_CS, linewidth=1.5, label="Total Supply", alpha=0.7)
ax.legend(fontsize=8)
style_ax(ax, "Aggregate Credit Demand vs Supply", ylabel="Amount")

# (1,1) Credit demand per borrower (among those who borrow)
ax = axes[1, 1]
mean_cd_borrower = []
for period in range(N_PERIODS):
    borrowers = cd[period] > 1e-10
    if np.any(borrowers):
        mean_cd_borrower.append(np.mean(cd[period, borrowers]))
    else:
        mean_cd_borrower.append(0.0)
ax.plot(t, mean_cd_borrower, color=C_CD, linewidth=1.5)
style_ax(ax, "Mean Credit Demand (borrowers only)", ylabel="Amount per borrower")

plt.tight_layout()
save_figure(fig, 2, "credit_demand_decomp")

# %%
# Figure 3: Transition Timeline
# -----------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Figure 3: Transition Timeline — From Credit-Dependent to Self-Financing",
    fontsize=13,
    fontweight="bold",
    y=1.0,
)

# (0,0) Fraction self-financing over time with milestones
ax = axes[0, 0]
ax.plot(t, frac_self_financing * 100, color=C_SELF, linewidth=2)
ax.axhline(50, color="gray", linewidth=0.5, linestyle=":")
ax.axhline(90, color="gray", linewidth=0.5, linestyle=":")
ax.axhline(99, color="gray", linewidth=0.5, linestyle=":")
if pct_50 >= 0:
    ax.axvline(pct_50, color=C_WB, linewidth=1, linestyle="--", alpha=0.7)
    ax.annotate(f"50% @ t={pct_50}", (pct_50, 52), fontsize=8, color=C_WB)
if pct_90 >= 0:
    ax.axvline(pct_90, color=C_NW, linewidth=1, linestyle="--", alpha=0.7)
    ax.annotate(f"90% @ t={pct_90}", (pct_90, 92), fontsize=8, color=C_NW)
if pct_99 >= 0:
    ax.axvline(pct_99, color=C_PROFIT, linewidth=1, linestyle="--", alpha=0.7)
    ax.annotate(f"99% @ t={pct_99}", (pct_99, 80), fontsize=8, color=C_PROFIT)
ax.set_ylim(0, 105)
style_ax(ax, "% Firms Self-Financing", ylabel="%")

# (0,1) Number of active borrowers
ax = axes[0, 1]
ax.bar(t, _n_borrowers, color=C_CD, alpha=0.7, width=1.0)
style_ax(ax, "Active Borrowers per Period", ylabel="# Firms")

# (1,0) Total lending volume
ax = axes[1, 0]
ax.bar(t, _total_lending, color=C_BANK, alpha=0.7, width=1.0)
style_ax(ax, "Total Lending Volume per Period", ylabel="Amount")

# (1,1) Active loans in LoanBook
ax = axes[1, 1]
ax.bar(t, _loans_outstanding, color=C_CS, alpha=0.7, width=1.0)
style_ax(ax, "Active Loans in LoanBook", ylabel="# Loans")

plt.tight_layout()
save_figure(fig, 3, "transition_timeline")

# %%
# Figure 4: Per-Firm Credit Lifecycle
# ------------------------------------

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle(
    "Figure 4: Per-Firm Credit Lifecycle — Individual Firm Trajectories",
    fontsize=13,
    fontweight="bold",
    y=0.998,
)

# Select 6 representative firms:
# - 2 that start with highest credit demand
# - 2 that have mid-range NW at period 0
# - 2 that go bankrupt at some point (if any)
initial_cd_rank = np.argsort(cd[0])[::-1]  # highest CD first
bankrupt_firms = list({idx for _, idx in _bankruptcies})

selected = []
# Top 2 initial borrowers
for f in initial_cd_rank:
    if len(selected) < 2:
        selected.append(f)

# 2 mid-range NW firms (not already selected)
nw_rank = np.argsort(nw[0])
mid = len(nw_rank) // 2
for f in nw_rank[mid - 5 : mid + 5]:
    if f not in selected and len(selected) < 4:
        selected.append(f)

# 2 that go bankrupt (if available)
for f in bankrupt_firms[:5]:
    if f not in selected and len(selected) < 6:
        selected.append(f)

# Fill remainder if needed
for f in range(n_firms):
    if f not in selected and len(selected) < 6:
        selected.append(f)

selected = selected[:6]
firm_colors = plt.cm.Set1(np.linspace(0, 0.8, 6))

# (0,0) Net Worth trajectories
ax = axes[0, 0]
for i, f in enumerate(selected):
    ax.plot(t, nw[:, f], color=firm_colors[i], linewidth=1, label=f"Firm {f}")
ax.legend(fontsize=7, ncol=2)
style_ax(ax, "Net Worth per Firm", ylabel="Amount")

# (0,1) Wage Bill trajectories
ax = axes[0, 1]
for i, f in enumerate(selected):
    ax.plot(t, wb[:, f], color=firm_colors[i], linewidth=1, label=f"Firm {f}")
ax.legend(fontsize=7, ncol=2)
style_ax(ax, "Wage Bill per Firm", ylabel="Amount")

# (1,0) NW/WB ratio trajectories
ax = axes[1, 0]
for i, f in enumerate(selected):
    ratio = nw[:, f] / np.where(wb[:, f] > 1e-10, wb[:, f], 1e-10)
    ax.plot(t, ratio, color=firm_colors[i], linewidth=1, label=f"Firm {f}")
ax.axhline(1.0, color="red", linewidth=1, linestyle="--", alpha=0.5, label="NW=WB")
ax.legend(fontsize=7, ncol=2)
ax.set_yscale("log")
style_ax(ax, "NW/WB Ratio per Firm (log)", ylabel="Ratio")

# (1,1) Credit Demand trajectories
ax = axes[1, 1]
for i, f in enumerate(selected):
    ax.plot(t, cd[:, f], color=firm_colors[i], linewidth=1, label=f"Firm {f}")
ax.legend(fontsize=7, ncol=2)
style_ax(ax, "Credit Demand per Firm", ylabel="Amount")

# (2,0) Production trajectories
ax = axes[2, 0]
for i, f in enumerate(selected):
    ax.plot(t, prod[:, f], color=firm_colors[i], linewidth=1, label=f"Firm {f}")
ax.legend(fontsize=7, ncol=2)
style_ax(ax, "Production per Firm", ylabel="Units")

# (2,1) Retained Profit trajectories
ax = axes[2, 1]
for i, f in enumerate(selected):
    ax.plot(t, rp[:, f], color=firm_colors[i], linewidth=1, label=f"Firm {f}")
ax.axhline(0.0, color="gray", linewidth=0.5, linestyle=":")
ax.legend(fontsize=7, ncol=2)
style_ax(ax, "Retained Profit per Firm", ylabel="Amount")

plt.tight_layout()
save_figure(fig, 4, "firm_lifecycle")

# %%
# Figure 5: New Entrant Credit Dependency
# ----------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Figure 5: New Entrant Credit Dependency — Cohort Analysis",
    fontsize=13,
    fontweight="bold",
    y=1.0,
)

# For each bankruptcy, track how many periods until the replacement firm
# becomes self-financing (NW >= WB). Only count from the first period where
# the firm has employees (WB > 0), since firms spawn with WB=0 and are
# trivially "self-financing" before hiring anyone.
time_to_self_financing = []
cohort_nw_wb_curves = []  # (birth_period, firm_idx, [ratio_t0, ratio_t1, ...])

for bp, bi in _bankruptcies:
    # Firm is spawned at period bp+1 (SpawnReplacementFirms runs after MarkBankruptFirms)
    spawn_period = bp + 1
    if spawn_period >= N_PERIODS:
        continue
    # Find first period with employees (WB > 0)
    first_employed = None
    for tp in range(spawn_period, N_PERIODS):
        if wb[tp, bi] > 1e-10:
            first_employed = tp
            break
    if first_employed is None:
        continue  # Never hired — skip
    # Track NW/WB from first employment onward
    ratios = []
    self_fin_period = None
    for tp in range(first_employed, N_PERIODS):
        if wb[tp, bi] > 1e-10:
            r = nw[tp, bi] / wb[tp, bi]
        else:
            r = np.nan
        ratios.append(r)
        if r >= 1.0 and self_fin_period is None:
            self_fin_period = tp - first_employed
    if self_fin_period is not None:
        time_to_self_financing.append(self_fin_period)
    cohort_nw_wb_curves.append((first_employed, bi, ratios))

# (0,0) Histogram of time-to-self-financing
ax = axes[0, 0]
if time_to_self_financing:
    ax.hist(
        time_to_self_financing,
        bins=range(0, max(time_to_self_financing) + 2),
        color=C_NEW,
        alpha=0.7,
        edgecolor="black",
    )
    mean_ttl = np.mean(time_to_self_financing)
    ax.axvline(
        mean_ttl, color=C_CD, linewidth=2, linestyle="--", label=f"Mean: {mean_ttl:.1f}"
    )
    ax.legend(fontsize=8)
else:
    ax.text(
        0.5,
        0.5,
        "No bankruptcies observed",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
style_ax(
    ax,
    "Time to Self-Financing (New Entrants)",
    xlabel="Periods after entry",
    ylabel="Count",
)

# (0,1) NW/WB trajectory of new entrants (overlay)
ax = axes[0, 1]
max_horizon = 30
for _i, (_sp, _fi, ratios) in enumerate(cohort_nw_wb_curves[:20]):  # first 20 cohorts
    horizon = min(len(ratios), max_horizon)
    finite_ratios = [min(r, 100) for r in ratios[:horizon]]
    ax.plot(range(horizon), finite_ratios, alpha=0.4, linewidth=0.8, color=C_NEW)
ax.axhline(1.0, color="red", linewidth=1.5, linestyle="--", label="NW=WB threshold")
ax.legend(fontsize=8)
style_ax(
    ax,
    "New Entrant NW/WB Trajectories",
    xlabel="Periods since entry",
    ylabel="NW/WB Ratio",
)

# (1,0) Cohort entry period distribution
ax = axes[1, 0]
if _bankruptcies:
    entry_periods = [bp + 1 for bp, _ in _bankruptcies if bp + 1 < N_PERIODS]
    ax.hist(
        entry_periods,
        bins=range(0, N_PERIODS + 1),
        color=C_NEW,
        alpha=0.7,
        edgecolor="none",
    )
else:
    ax.text(
        0.5,
        0.5,
        "No bankruptcies observed",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
style_ax(ax, "New Firm Entry Timing", xlabel="Period of Entry", ylabel="Count")

# (1,1) NW at spawn vs survivor mean NW
ax = axes[1, 1]
if _bankruptcies:
    spawn_nw = []
    survivor_mean_nw = []
    for bp, bi in _bankruptcies:
        sp = bp + 1
        if sp < N_PERIODS:
            spawn_nw.append(nw[sp, bi])
            # Survivor mean = mean of all firms that period (approximation)
            survivor_mean_nw.append(np.mean(nw[sp]))
    ax.scatter(survivor_mean_nw, spawn_nw, color=C_NEW, alpha=0.6, s=20)
    if survivor_mean_nw:
        max_val = max(max(survivor_mean_nw), max(spawn_nw)) * 1.1
        ax.plot(
            [0, max_val], [0, max_val], "k--", linewidth=0.5, alpha=0.3, label="1:1"
        )
        factor_line = [v * sim.config.new_firm_size_factor for v in [0, max_val]]
        ax.plot(
            [0, max_val],
            factor_line,
            "r--",
            linewidth=1,
            alpha=0.5,
            label=f"× {sim.config.new_firm_size_factor}",
        )
    ax.legend(fontsize=8)
else:
    ax.text(
        0.5,
        0.5,
        "No bankruptcies observed",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
style_ax(
    ax, "New Firm NW vs Survivor Mean", xlabel="Survivor Mean NW", ylabel="New Firm NW"
)

plt.tight_layout()
save_figure(fig, 5, "new_entrant_cohorts")

# %%
# Figure 6: Profit Accumulation Waterfall
# ----------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Figure 6: Profit Accumulation Waterfall — Why Firms Self-Finance",
    fontsize=13,
    fontweight="bold",
    y=1.0,
)

# Pick a representative mature firm (high production, no bankruptcy)
bankrupt_set_firms = {idx for _, idx in _bankruptcies}
candidates = [f for f in range(n_firms) if f not in bankrupt_set_firms]
if candidates:
    # Pick the one with median final NW
    candidate_nw = [nw[-1, f] for f in candidates]
    median_idx = np.argsort(candidate_nw)[len(candidate_nw) // 2]
    rep_firm = candidates[median_idx]
else:
    rep_firm = 0

# (0,0) Waterfall: Revenue components for representative firm
ax = axes[0, 0]
firm_revenue = revenue[:, rep_firm]
firm_wb_cost = wb[:, rep_firm]
firm_interest = gp[:, rep_firm] - np_arr[:, rep_firm]  # interest = GP - NP
firm_dividends = np_arr[:, rep_firm] - rp[:, rep_firm]  # dividends = NP - RP
firm_retained = rp[:, rep_firm]

ax.fill_between(t, 0, firm_revenue, color=C_PROFIT, alpha=0.3, label="Revenue")
ax.plot(t, firm_wb_cost, color=C_WB, linewidth=1, label="Wage Bill")
ax.plot(t, firm_revenue, color=C_PROFIT, linewidth=1)
ax.plot(t, firm_retained, color=C_NW, linewidth=1, label="Retained Profit")
ax.legend(fontsize=8)
style_ax(ax, f"Revenue Breakdown — Firm {rep_firm}", ylabel="Amount")

# (0,1) Cumulative NW growth for representative firm
ax = axes[0, 1]
ax.plot(t, nw[:, rep_firm], color=C_NW, linewidth=1.5, label="Net Worth")
ax.plot(t, wb[:, rep_firm], color=C_WB, linewidth=1.5, label="Wage Bill")
cum_retained = np.cumsum(rp[:, rep_firm])
ax.plot(
    t,
    cum_retained + nw[0, rep_firm],
    color=C_PROFIT,
    linewidth=1,
    linestyle="--",
    label="Initial NW + Σ(Retained)",
)
ax.legend(fontsize=8)
style_ax(ax, f"NW Growth — Firm {rep_firm}", ylabel="Amount")

# (1,0) Aggregate: mean retained profit / mean WB ratio (the accumulation rate)
ax = axes[1, 0]
mean_rp = np.mean(rp, axis=1)
mean_wb_agg = np.mean(wb, axis=1)
safe_wb_agg = np.where(mean_wb_agg > 1e-10, mean_wb_agg, 1e-10)
accum_rate = mean_rp / safe_wb_agg
ax.plot(t, accum_rate * 100, color=C_PROFIT, linewidth=1.5)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
style_ax(ax, "Accumulation Rate (Retained Profit / Wage Bill)", ylabel="%")

# (1,1) Aggregate profit decomposition
ax = axes[1, 1]
total_rev = np.sum(revenue, axis=1)
total_wb_cost = np.sum(wb, axis=1)
total_interest = np.sum(gp - np_arr, axis=1)
total_div = np.sum(np_arr - rp, axis=1)
total_retained = np.sum(rp, axis=1)

# Stacked bar is too busy; use lines
ax.plot(t, total_rev, color=C_PROFIT, linewidth=1.2, label="Revenue")
ax.plot(t, total_wb_cost, color=C_WB, linewidth=1.2, label="Wages")
ax.plot(t, total_interest, color=C_CD, linewidth=1.2, label="Interest")
ax.plot(t, total_div, color=C_NEW, linewidth=1.2, label="Dividends")
ax.plot(t, total_retained, color=C_NW, linewidth=1.5, label="Retained")
ax.legend(fontsize=8)
style_ax(ax, "Aggregate Profit Decomposition", ylabel="Amount")

plt.tight_layout()
save_figure(fig, 6, "profit_waterfall")

# %%
# Figure 7: Bank-Side View
# -------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Figure 7: Bank-Side View — Unused Lending Capacity",
    fontsize=13,
    fontweight="bold",
    y=1.0,
)

# (0,0) Credit supply vs actual lending
ax = axes[0, 0]
ax.plot(t, _total_cs, color=C_CS, linewidth=1.5, label="Credit Supply")
ax.plot(t, _total_lending, color=C_CD, linewidth=1.5, label="Actual Lending")
ax.fill_between(
    t, _total_lending, _total_cs, color=C_CS, alpha=0.15, label="Unused Capacity"
)
ax.legend(fontsize=8)
style_ax(ax, "Credit Supply vs Actual Lending", ylabel="Amount")

# (0,1) Credit utilization ratio
ax = axes[0, 1]
total_cs_arr = np.array(_total_cs)
total_lending_arr = np.array(_total_lending)
safe_cs = np.where(total_cs_arr > 1e-10, total_cs_arr, 1e-10)
utilization = total_lending_arr / safe_cs
ax.plot(t, utilization * 100, color=C_BANK, linewidth=1.5)
ax.set_ylim(-5, 105)
style_ax(ax, "Credit Utilization Rate", ylabel="% of Supply Used")

# (1,0) Bank equity evolution (per bank)
ax = axes[1, 0]
bank_colors = plt.cm.tab10(np.linspace(0, 1, n_banks))
for b in range(n_banks):
    ax.plot(
        t,
        equity[:, b],
        color=bank_colors[b],
        linewidth=0.8,
        label=f"Bank {b}" if b < 5 else None,
    )
ax.legend(fontsize=7)
style_ax(ax, "Bank Equity per Bank", ylabel="Amount")

# (1,1) Credit supply per bank
ax = axes[1, 1]
for b in range(n_banks):
    ax.plot(
        t,
        cs[:, b],
        color=bank_colors[b],
        linewidth=0.8,
        label=f"Bank {b}" if b < 5 else None,
    )
ax.legend(fontsize=7)
style_ax(ax, "Credit Supply per Bank", ylabel="Amount")

plt.tight_layout()
save_figure(fig, 7, "bank_side")

# %%
# Figure 8: Credit Market Activity Dashboard
# -------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "Figure 8: Credit Market Activity Dashboard", fontsize=13, fontweight="bold", y=1.0
)

# (0,0) Total credit demand per period
ax = axes[0, 0]
ax.bar(t, _total_cd, color=C_CD, alpha=0.7, width=1.0)
style_ax(ax, "Total Credit Demand", ylabel="Amount")

# (0,1) Total credit supply per period
ax = axes[0, 1]
ax.plot(t, _total_cs, color=C_CS, linewidth=1.5)
style_ax(ax, "Total Credit Supply", ylabel="Amount")

# (0,2) Number of active borrowers
ax = axes[0, 2]
ax.bar(t, _n_borrowers, color=C_NEW, alpha=0.7, width=1.0)
ax.set_ylim(0, max(max(_n_borrowers), 1) * 1.1)
style_ax(ax, "# Active Borrowers", ylabel="Count")

# (1,0) Total loans outstanding
ax = axes[1, 0]
ax.bar(t, _loans_outstanding, color=C_BANK, alpha=0.7, width=1.0)
style_ax(ax, "# Loans in LoanBook", ylabel="Count")

# (1,1) Firm bankruptcies per period
ax = axes[1, 1]
bankr_per_period = np.zeros(N_PERIODS)
for bp, _ in _bankruptcies:
    if bp < N_PERIODS:
        bankr_per_period[bp] += 1
ax.bar(t, bankr_per_period, color=C_SELF, alpha=0.7, width=1.0)
style_ax(ax, "Firm Bankruptcies per Period", ylabel="Count")

# (1,2) Aggregate NW growth vs initial
ax = axes[1, 2]
total_nw = np.sum(nw, axis=1)
total_nw_normalized = total_nw / max(total_nw[0], 1e-10)
ax.plot(t, total_nw_normalized, color=C_NW, linewidth=1.5)
ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
style_ax(ax, "Total Net Worth (normalized to t=0)", ylabel="× initial")

plt.tight_layout()
save_figure(fig, 8, "activity_dashboard")

# %%
# Final Summary
# -------------

print("\n" + "=" * 60)
print("INVESTIGATION COMPLETE")
print("=" * 60)
print(f"8 figures saved to {OUTPUT_DIR}/")
print()
print("Key Findings:")
print(
    f"  1. Self-financing transition: 50% @ t={pct_50}, "
    f"90% @ t={pct_90}, 99% @ t={pct_99}"
)
print(f"  2. Steady-state NW/WB ratio: {median_nw_wb_late:.1f}x (median)")
print(
    f"  3. Total bankruptcies: {len(_bankruptcies)} "
    f"(~{len(_bankruptcies) / N_PERIODS:.1f}/period)"
)
if time_to_self_financing:
    print(
        f"  4. New entrant time to self-finance: "
        f"{np.mean(time_to_self_financing):.1f} periods (mean)"
    )
else:
    print("  4. No new entrant data (zero bankruptcies)")
print(
    f"  5. Residual credit activity: "
    f"{sum(1 for n in _n_borrowers if n > 0)} of {N_PERIODS} periods "
    f"have any borrowing"
)
print(
    f"  6. Accumulation rate (late): "
    f"{np.mean(accum_rate[max(N_PERIODS - 50, 0) :]) * 100:.1f}% of WB/period"
)
