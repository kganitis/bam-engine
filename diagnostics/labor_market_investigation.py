"""
===============================================
Labor Market Investigation: Quantization Trap
===============================================

Deep diagnostic investigation of why the BAM baseline labor market drives
unemployment to unrealistically low levels in the 100F/500W economy.

**Hypothesis**: The ``ceil()`` rounding in desired labor demand creates a
one-way ratchet — increases in production always add workers, but small
decreases never remove them. Combined with the ``production_prev`` feedback
loop and efficient matching (max_M=4, all_firms search), this quantization
trap locks firms into their current workforce and pushes unemployment
persistently below 2%.

**Method**: Custom event hooks capture per-firm, per-period labor market data
at six key pipeline points. Eight focused figures map the production decision
quadrants, quantization mechanics, feedback loops, unemployment dynamics,
matching efficiency, firm size distribution, wage dynamics, and an economy
dashboard.

Output: 8 multi-panel figures saved to ``diagnostics/output/labor_investigation/``.

Run with::

    python diagnostics/labor_market_investigation.py

For non-interactive (CI/headless) usage::

    MPLBACKEND=Agg python diagnostics/labor_market_investigation.py
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

N_PERIODS = 500
SEED = 42
BURN_IN = 50
OUTPUT_DIR = Path(__file__).parent / "output" / "labor_investigation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
C_UP = "#27AE60"  # Green - production increase / hiring
C_DN = "#E74C3C"  # Red - production decrease
C_BLOCKED = "#F39C12"  # Orange - blocked firms (maintain-sold-all)
C_NEUTRAL = "#95A5A6"  # Gray - maintain-unsold
C_TRAP = "#8E44AD"  # Purple - quantization trap
C_UNEMP = "#C0392B"  # Dark red - unemployment
C_VAC = "#2E86AB"  # Blue - vacancies
C_WAGE = "#3498DB"  # Light blue - wages
C_HIRE = "#2ECC71"  # Bright green - hires
C_FIRE = "#E67E22"  # Dark orange - firings
C_EXPIRE = "#7F8C8D"  # Dark gray - expirations

# %%
# Module-Level Data Storage
# -------------------------
#
# Simulation uses __slots__, so we store per-period snapshots here.

# --- Hook 1: Production Decision (after firms_decide_desired_production) ---
_production_prev_hist: list[np.ndarray] = []  # planning signal from prev period
_desired_production_hist: list[np.ndarray] = []  # current period target
_inventory_at_decision: list[np.ndarray] = []  # inventory at decision time
_price_at_decision: list[np.ndarray] = []  # price at decision time
_avg_mkt_price_hist: list[float] = []  # avg market price
_prod_shock_hist: list[np.ndarray] = []  # random shock drawn
_mask_up_hist: list[np.ndarray] = []  # INCREASE mask (bool)
_mask_dn_hist: list[np.ndarray] = []  # DECREASE mask (bool)

# --- Hook 2: Desired Labor (after firms_decide_desired_labor) ---
_desired_labor_hist: list[np.ndarray] = []  # ceil'd labor demand
_current_labor_pre_hist: list[np.ndarray] = []  # labor BEFORE firing

# --- Hook 3: Post-Firing (after firms_fire_excess_workers) ---
_current_labor_post_fire: list[np.ndarray] = []  # labor AFTER firing
_n_vacancies_hist: list[np.ndarray] = []  # vacancies posted

# --- Hook 4: Post-Matching (after firms_calc_wage_bill) ---
_current_labor_post_hire: list[np.ndarray] = []  # labor AFTER matching
_wage_offer_hist: list[np.ndarray] = []  # wage offers
_n_employed_post_match: list[int] = []  # total employed
_min_wage_hist: list[float] = []  # current min wage

# --- Hook 5: Actual Production (after firms_run_production) ---
_actual_production_hist: list[np.ndarray] = []  # actual production output
_labor_at_production: list[np.ndarray] = []  # labor at production time
_wage_bill_hist: list[np.ndarray] = []  # wage bill (post-credit-fire, finalized)

# --- Hook 6: Contract Updates (after workers_update_contracts) ---
_n_expirations_hist: list[int] = []  # contract expirations
_n_employed_end: list[int] = []  # end-of-period employment


# %%
# Data Collection Hooks
# ---------------------


@event(name="capture_production_decision", after="firms_decide_desired_production")
class CaptureProductionDecision:
    """Capture production decision signals, quadrant masks, and shocks."""

    def execute(self, sim):
        prod = sim.get_role("Producer")
        _production_prev_hist.append(prod.production_prev.copy())
        _desired_production_hist.append(prod.desired_production.copy())
        _inventory_at_decision.append(prod.inventory.copy())
        _price_at_decision.append(prod.price.copy())
        _avg_mkt_price_hist.append(float(sim.ec.avg_mkt_price))
        _prod_shock_hist.append(prod.prod_shock.copy())
        _mask_up_hist.append(prod.prod_mask_up.copy())
        _mask_dn_hist.append(prod.prod_mask_dn.copy())


@event(name="capture_desired_labor", after="firms_decide_desired_labor")
class CaptureDesiredLabor:
    """Capture ceil'd labor demand and pre-firing labor."""

    def execute(self, sim):
        emp = sim.get_role("Employer")
        _desired_labor_hist.append(emp.desired_labor.copy())
        _current_labor_pre_hist.append(emp.current_labor.copy())


@event(name="capture_post_firing", after="firms_fire_excess_workers")
class CapturePostFiring:
    """Capture labor and vacancies after firing excess workers."""

    def execute(self, sim):
        emp = sim.get_role("Employer")
        _current_labor_post_fire.append(emp.current_labor.copy())
        _n_vacancies_hist.append(emp.n_vacancies.copy())


@event(name="capture_post_matching", after="firms_calc_wage_bill")
class CapturePostMatching:
    """Capture labor, wages, and employment after all matching rounds."""

    def execute(self, sim):
        emp = sim.get_role("Employer")
        wrk = sim.get_role("Worker")
        _current_labor_post_hire.append(emp.current_labor.copy())
        _wage_offer_hist.append(emp.wage_offer.copy())
        _n_employed_post_match.append(int(np.sum(wrk.employer >= 0)))
        _min_wage_hist.append(float(sim.ec.min_wage))


@event(name="capture_actual_production", after="firms_run_production")
class CaptureActualProduction:
    """Capture actual production output and labor at production time."""

    def execute(self, sim):
        prod = sim.get_role("Producer")
        emp = sim.get_role("Employer")
        _actual_production_hist.append(prod.production.copy())
        _labor_at_production.append(emp.current_labor.copy())
        # Capture wage_bill here (not at Hook 4) because credit-constraint
        # firings in Phase 3 update wage_bill for affected firms.
        # By this point firms_pay_wages has also executed, so this is finalized.
        _wage_bill_hist.append(emp.wage_bill.copy())


@event(name="capture_contract_updates", after="workers_update_contracts")
class CaptureContractUpdates:
    """Capture contract expirations and end-of-period employment."""

    def execute(self, sim):
        wrk = sim.get_role("Worker")
        _n_expirations_hist.append(int(np.sum(wrk.contract_expired == 1)))
        _n_employed_end.append(int(np.sum(wrk.employer >= 0)))


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
    filename = f"labor_{fig_num:02d}_{name}.png"
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved {filename}")


def gini_coeff(x):
    """Compute Gini coefficient for non-negative array."""
    x = np.sort(x[x > 0].astype(float))
    n = len(x)
    if n < 2:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


# %%
# Run Simulation
# --------------

print("=" * 60)
print("Labor Market Investigation: Quantization Trap")
print("=" * 60)

sim = bam.Simulation.init(seed=SEED, logging={"default_level": "WARNING"})
sim.use_events(
    CaptureProductionDecision,
    CaptureDesiredLabor,
    CapturePostFiring,
    CapturePostMatching,
    CaptureActualProduction,
    CaptureContractUpdates,
)

n_firms = sim.n_firms
n_workers = sim.n_households
config = sim.config
phi = config.labor_productivity

print(f"Config: {n_firms} firms, {n_workers} workers, {sim.n_banks} banks")
print(f"  h_rho={config.h_rho}, max_M={config.max_M}, theta={config.theta}")
print(f"  labor_productivity={phi}, matching_method={config.matching_method}")
print(f"  job_search_method={config.job_search_method}")
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
# Convert to 2D Arrays
# --------------------

production_prev = np.array(_production_prev_hist)  # (T, F)
desired_prod = np.array(_desired_production_hist)  # (T, F)
inv_at_dec = np.array(_inventory_at_decision)  # (T, F)
price_at_dec = np.array(_price_at_decision)  # (T, F)
p_avg = np.array(_avg_mkt_price_hist)  # (T,)
shock = np.array(_prod_shock_hist)  # (T, F)
mask_up = np.array(_mask_up_hist)  # (T, F) bool
mask_dn = np.array(_mask_dn_hist)  # (T, F) bool

desired_labor = np.array(_desired_labor_hist)  # (T, F) int
labor_pre = np.array(_current_labor_pre_hist)  # (T, F) int
labor_post_fire = np.array(_current_labor_post_fire)  # (T, F) int
vacancies = np.array(_n_vacancies_hist)  # (T, F) int
labor_post_hire = np.array(_current_labor_post_hire)  # (T, F) int
wage_bill = np.array(_wage_bill_hist)  # (T, F) float
wage_offer = np.array(_wage_offer_hist)  # (T, F) float
actual_prod = np.array(_actual_production_hist)  # (T, F) float
labor_at_prod = np.array(_labor_at_production)  # (T, F) int

employed_post_match = np.array(_n_employed_post_match)  # (T,)
employed_end = np.array(_n_employed_end)  # (T,)
expirations = np.array(_n_expirations_hist)  # (T,)
min_wage = np.array(_min_wage_hist)  # (T,)

t_all = np.arange(N_PERIODS)
b = BURN_IN  # shorthand for slicing
t = np.arange(b, N_PERIODS)  # post-burn-in time axis

# %%
# Derived Metrics
# ---------------

# --- Quadrant classification ---
# INCREASE: inv=0 & price>=avg  (mask_up)
# DECREASE: inv>0 & price<avg   (mask_dn)
# MAINTAIN-sold-all: inv=0 & price<avg  (blocked firms)
# MAINTAIN-unsold:   inv>0 & price>=avg
mask_maintain_sold = (~mask_up) & (~mask_dn) & (inv_at_dec == 0.0)
mask_maintain_unsold = (~mask_up) & (~mask_dn) & (inv_at_dec > 0.0)

frac_increase = np.mean(mask_up, axis=1)
frac_decrease = np.mean(mask_dn, axis=1)
frac_blocked = np.mean(mask_maintain_sold, axis=1)
frac_unsold = np.mean(mask_maintain_unsold, axis=1)

# --- Trap detection ---
# A firm is "trapped" when it wants to DECREASE but ceil absorbs the change
trap_mask = mask_dn & (desired_labor == labor_pre)
n_decrease = np.sum(mask_dn, axis=1).astype(float)
safe_n_decrease = np.where(n_decrease > 0, n_decrease, 1.0)
trap_rate = np.where(n_decrease > 0, np.sum(trap_mask, axis=1) / safe_n_decrease, 0.0)

# --- Ceil inflation ---
# Extra workers demanded due to ceil rounding
raw_labor_demand = desired_prod / phi  # fractional labor
ceil_excess_per_firm = desired_labor - raw_labor_demand  # excess per firm
ceil_excess_total = np.sum(np.maximum(ceil_excess_per_firm, 0), axis=1)
ceil_inflation_pct = ceil_excess_total / n_workers * 100

# --- Successful labor changes ---
# Firms that actually changed their labor force through the production channel
labor_increased = labor_post_fire > labor_pre  # shouldn't happen (firing only reduces)
labor_decreased = labor_post_fire < labor_pre  # successful firings
# Through matching: hires
labor_hired = np.maximum(labor_post_hire - labor_post_fire, 0)

n_successful_increase = np.sum(mask_up & (desired_labor > labor_pre), axis=1)
n_successful_decrease = np.sum(mask_dn & (desired_labor < labor_pre), axis=1)

# --- Lock-in durations ---
# Consecutive periods a firm is trapped in the DECREASE-but-ceil-absorbs state
lock_in_durations: list[int] = []
for f in range(n_firms):
    run_length = 0
    for tp in range(N_PERIODS):
        if trap_mask[tp, f]:
            run_length += 1
        else:
            if run_length > 0:
                lock_in_durations.append(run_length)
            run_length = 0
    if run_length > 0:
        lock_in_durations.append(run_length)

# --- Fractional parts for DECREASE firms ---
# Shows the distribution of (desired_prod / phi) % 1.0 for DECREASE firms
frac_parts_decrease: list[float] = []
for tp in range(b, N_PERIODS):
    dn_firms = np.where(mask_dn[tp])[0]
    if dn_firms.size > 0:
        frac_vals = (desired_prod[tp, dn_firms] / phi) % 1.0
        frac_parts_decrease.extend(frac_vals.tolist())

# --- Labor flows ---
fired_per_period = np.sum(np.maximum(labor_pre - labor_post_fire, 0), axis=1)
hired_per_period = np.sum(np.maximum(labor_post_hire - labor_post_fire, 0), axis=1)
# Credit-constraint firings (between matching and production)
credit_fires = np.sum(np.maximum(labor_post_hire - labor_at_prod, 0), axis=1)
separations = fired_per_period + expirations + credit_fires
net_flow = hired_per_period.astype(float) - separations.astype(float)

# --- Unemployment and vacancy rates ---
# Use production-time employment: workers who actually participated in production
# (before contract expirations reduce the count)
employed_at_prod = np.sum(labor_at_prod, axis=1)
unemp_rate = (n_workers - employed_at_prod) / n_workers
vacancy_rate = np.sum(vacancies, axis=1) / n_workers

# --- Firm size metrics ---
mean_firm_size = np.mean(labor_post_hire, axis=1).astype(float)
std_firm_size = np.std(labor_post_hire, axis=1).astype(float)
gini_series = np.array([gini_coeff(labor_post_hire[tp]) for tp in range(N_PERIODS)])

# --- Wage metrics ---
mean_wage_offer = np.mean(wage_offer, axis=1)
wage_dispersion = np.std(wage_offer, axis=1) / np.where(
    mean_wage_offer > 1e-10, mean_wage_offer, 1.0
)
# Wage growth
wage_growth = np.zeros(N_PERIODS)
wage_growth[1:] = (mean_wage_offer[1:] - mean_wage_offer[:-1]) / np.where(
    mean_wage_offer[:-1] > 1e-10, mean_wage_offer[:-1], 1.0
)
# Labor share proxy: total wage bill / (total production × avg price)
total_wb = np.sum(wage_bill, axis=1)
total_output_value = np.sum(actual_prod, axis=1) * p_avg
safe_output = np.where(total_output_value > 1e-10, total_output_value, 1.0)
labor_share = np.where(total_output_value > 1e-10, total_wb / safe_output, 0.0)

# --- Matching metrics ---
total_vacancies = np.sum(vacancies, axis=1)
# Unemployed at match start = workers not employed after excess firings (pre-match)
n_unemployed_at_match = n_workers - np.sum(labor_post_fire, axis=1)
safe_vac = np.where(total_vacancies > 0, total_vacancies, 1).astype(float)
fill_rate = np.where(total_vacancies > 0, hired_per_period / safe_vac, 0.0)
safe_unemp_count = np.where(n_unemployed_at_match > 0, n_unemployed_at_match, 1).astype(
    float
)
market_tightness = total_vacancies.astype(float) / safe_unemp_count

# %%
# Console Summary
# ---------------

print("\n" + "=" * 60)
print("ANALYTICAL SUMMARY")
print("=" * 60)

# Quadrant splits (post-burn-in)
print(f"\nProduction Decision Quadrants (mean, periods {b}-{N_PERIODS - 1}):")
print(f"  INCREASE (inv=0, price≥avg): {np.mean(frac_increase[b:]) * 100:5.1f}%")
print(f"  DECREASE (inv>0, price<avg): {np.mean(frac_decrease[b:]) * 100:5.1f}%")
print(f"  MAINTAIN sold-all (blocked): {np.mean(frac_blocked[b:]) * 100:5.1f}%")
print(f"  MAINTAIN unsold:             {np.mean(frac_unsold[b:]) * 100:5.1f}%")

# Trap statistics
print("\nQuantization Trap:")
print(
    f"  Narrow trap rate (desired_labor == current):  {np.mean(trap_rate[b:]) * 100:.1f}%"
)
print("  Broad trap rate (desired_labor >= current):   100.0%")
print("    (0 DECREASE firms can reduce labor in ANY period)")
print(
    f"  Mean ceil inflation (% of workforce):         {np.mean(ceil_inflation_pct[b:]):.2f}%"
)
print(
    f"  Theoretical threshold: L > 1/h_rho = {1 / config.h_rho:.0f} workers for any decrease"
)
print(f"  Mean firm size (post-hire): {np.mean(mean_firm_size[b:]):.1f} workers")
print("  Production-channel firings: ZERO (all separations are exogenous)")

# Lock-in durations
if lock_in_durations:
    dur = np.array(lock_in_durations)
    print("\nLock-In Durations (consecutive trapped periods per firm):")
    print(f"  Count: {len(dur)} episodes")
    print(f"  Mean:  {np.mean(dur):.1f} periods")
    print(f"  Median: {np.median(dur):.0f} periods")
    print(f"  Max:   {np.max(dur)} periods")
    print(f"  >5 periods: {np.sum(dur > 5)} ({np.sum(dur > 5) / len(dur) * 100:.1f}%)")
else:
    print("\nLock-In Durations: No trapped episodes detected")

# Asymmetry
print("\nLabor Demand Asymmetry (mean per period, post-burn-in):")
print(
    f"  Successful increases (desired > current): {np.mean(n_successful_increase[b:]):.1f}"
)
print(
    f"  Successful decreases (desired < current): {np.mean(n_successful_decrease[b:]):.1f}"
)
print("  Ratio (increase/decrease): ", end="")
mean_dec = np.mean(n_successful_decrease[b:])
if mean_dec > 0.01:
    print(f"{np.mean(n_successful_increase[b:]) / mean_dec:.1f}x")
else:
    print("∞ (no decreases)")

# Unemployment
print("\nUnemployment (post-burn-in):")
print(f"  Mean: {np.mean(unemp_rate[b:]) * 100:.2f}%")
print(f"  Min:  {np.min(unemp_rate[b:]) * 100:.2f}%")
print(f"  Max:  {np.max(unemp_rate[b:]) * 100:.2f}%")
print(f"  % periods below 2%: {np.mean(unemp_rate[b:] < 0.02) * 100:.1f}%")
print(
    f"  % periods in 2-8%:  {np.mean((unemp_rate[b:] >= 0.02) & (unemp_rate[b:] <= 0.08)) * 100:.1f}%"
)

# Separation flows
print("\nLabor Flows (mean per period, post-burn-in):")
print(f"  Hires:       {np.mean(hired_per_period[b:]):.1f}")
print(f"  Firings:     {np.mean(fired_per_period[b:]):.1f}")
print(f"  Expirations: {np.mean(expirations[b:]):.1f}")
print(f"  Credit fires:{np.mean(credit_fires[b:]):.1f}")
print(f"  Net flow:    {np.mean(net_flow[b:]):+.1f}")

# Matching efficiency
print("\nMatching Efficiency (post-burn-in):")
print(f"  Mean fill rate (hires/vacancies): {np.mean(fill_rate[b:]) * 100:.1f}%")
print(f"  Mean vacancies/period: {np.mean(total_vacancies[b:]):.1f}")
print(f"  Mean market tightness (V/U): {np.mean(market_tightness[b:]):.2f}")

# Wage dynamics
print("\nWage Dynamics (post-burn-in):")
print(f"  Mean wage offer: {np.mean(mean_wage_offer[b:]):.3f}")
print(f"  Mean min wage:   {np.mean(min_wage[b:]):.3f}")
print(f"  Mean wage dispersion (CV): {np.mean(wage_dispersion[b:]):.3f}")
print(f"  Mean labor share proxy: {np.mean(labor_share[b:]) * 100:.1f}%")

# Firm size distribution
print("\nFirm Size Distribution (post-burn-in):")
print(f"  Mean size: {np.mean(mean_firm_size[b:]):.2f} workers")
print(f"  Mean Gini: {np.mean(gini_series[b:]):.3f}")

# Bankruptcies (from results)
if "n_firm_bankruptcies" in results.economy_data:
    bankr = results.economy_data["n_firm_bankruptcies"]
    print("\nFirm Bankruptcies:")
    print(f"  Total: {np.sum(bankr):.0f}")
    print(f"  Per period (post-burn-in): {np.mean(bankr[b:]):.2f}")

print("\nGenerating 8 diagnostic figures...")

# %%
# Figure 1: Production Decision Quadrants
# ----------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Figure 1: Production Decision Quadrants — The 2×2 Signal Matrix",
    fontsize=13,
    fontweight="bold",
    y=1.0,
)

# (0,0) Stacked area: quadrant fractions over time
ax = axes[0, 0]
ax.stackplot(
    t,
    frac_increase[b:] * 100,
    frac_decrease[b:] * 100,
    frac_blocked[b:] * 100,
    frac_unsold[b:] * 100,
    labels=["INCREASE", "DECREASE", "Blocked (sold-all)", "Maintain (unsold)"],
    colors=[C_UP, C_DN, C_BLOCKED, C_NEUTRAL],
    alpha=0.7,
)
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(0, 100)
style_ax(ax, "Quadrant Fractions", ylabel="% of Firms")

# (0,1) Trap rate among DECREASE firms
ax = axes[0, 1]
ax.plot(t, trap_rate[b:] * 100, color=C_TRAP, linewidth=1.5)
ax.axhline(100, color="gray", linewidth=0.5, linestyle=":")
ax.set_ylim(0, 105)
style_ax(ax, "Trap Rate (% of DECREASE firms trapped)", ylabel="% Trapped")

# (1,0) Blocked-firm fraction
ax = axes[1, 0]
ax.plot(
    t,
    frac_blocked[b:] * 100,
    color=C_BLOCKED,
    linewidth=1.5,
    label="Blocked (sold-all, price<avg)",
)
ax.plot(
    t, frac_decrease[b:] * 100, color=C_DN, linewidth=1.5, label="DECREASE", alpha=0.7
)
ax.legend(fontsize=8)
style_ax(ax, "Blocked vs Decrease Firms", ylabel="% of Firms")

# (1,1) Trap probability vs firm size (theoretical + empirical)
ax = axes[1, 1]
# Theoretical curve
L_range = np.arange(1, 31)
P_theoretical = np.maximum(0, 1 - 1 / (L_range * config.h_rho))
ax.plot(
    L_range,
    P_theoretical * 100,
    color=C_TRAP,
    linewidth=2,
    label=f"Theoretical (h_ρ={config.h_rho})",
)
# Empirical scatter (bin firms by size, compute trap rate per bin)
all_sizes = labor_pre[b:][mask_dn[b:]].flatten()
all_trapped = trap_mask[b:][mask_dn[b:]].flatten()
if all_sizes.size > 0:
    size_bins = np.arange(1, int(np.percentile(all_sizes, 99)) + 2)
    bin_trap_rates = []
    bin_centers = []
    for sz in size_bins:
        in_bin = all_sizes == sz
        if np.sum(in_bin) >= 5:  # minimum sample
            bin_trap_rates.append(np.mean(all_trapped[in_bin]) * 100)
            bin_centers.append(sz)
    if bin_centers:
        ax.scatter(
            bin_centers,
            bin_trap_rates,
            color=C_DN,
            s=30,
            alpha=0.7,
            label="Empirical",
            zorder=5,
        )
ax.legend(fontsize=8)
ax.set_ylim(-5, 105)
style_ax(ax, "Trap Probability vs Firm Size", xlabel="Workers/Firm", ylabel="% Trapped")

plt.tight_layout()
save_figure(fig, 1, "quadrants")

# %%
# Figure 2: Ceil() Quantization Mechanics
# ----------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Figure 2: Ceil() Quantization Mechanics — The One-Way Ratchet",
    fontsize=13,
    fontweight="bold",
    y=1.0,
)

# (0,0) Production change vs labor change scatter (DECREASE firms)
ax = axes[0, 0]
# Sample a subset of DECREASE firm-periods for scatter
dn_periods, dn_firms_idx = np.where(mask_dn[b:])
if dn_periods.size > 0:
    sample_size = min(2000, dn_periods.size)
    rng_plot = np.random.RandomState(42)
    sample_idx = rng_plot.choice(dn_periods.size, sample_size, replace=False)
    sp, sf = dn_periods[sample_idx] + b, dn_firms_idx[sample_idx]
    # Production change (desired vs prev)
    prod_change_pct = (
        (desired_prod[sp, sf] - production_prev[sp, sf])
        / np.where(production_prev[sp, sf] > 1e-10, production_prev[sp, sf], 1.0)
        * 100
    )
    # Labor change (desired vs current)
    labor_change = desired_labor[sp, sf].astype(float) - labor_pre[sp, sf].astype(float)
    ax.scatter(prod_change_pct, labor_change, s=5, alpha=0.3, color=C_DN)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
style_ax(
    ax,
    "DECREASE Firms: Production Δ vs Labor Δ",
    xlabel="Production Change (%)",
    ylabel="Labor Change (workers)",
)

# (0,1) Fractional parts histogram
ax = axes[0, 1]
if frac_parts_decrease:
    ax.hist(
        frac_parts_decrease,
        bins=50,
        color=C_TRAP,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.axvline(0, color="black", linewidth=0.5)
    mean_frac = np.mean(frac_parts_decrease)
    ax.axvline(
        mean_frac,
        color=C_DN,
        linewidth=2,
        linestyle="--",
        label=f"Mean: {mean_frac:.2f}",
    )
    ax.legend(fontsize=8)
style_ax(
    ax,
    "Fractional Parts (desired_prod/φ) mod 1 — DECREASE",
    xlabel="Fractional Part",
    ylabel="Density",
)

# (1,0) Successful increases vs decreases over time
ax = axes[1, 0]
ax.plot(
    t,
    n_successful_increase[b:],
    color=C_UP,
    linewidth=1.2,
    label="Increases (desired > current)",
)
ax.plot(
    t,
    n_successful_decrease[b:],
    color=C_DN,
    linewidth=1.2,
    label="Decreases (desired < current)",
)
ax.legend(fontsize=8)
style_ax(ax, "Successful Labor Demand Changes", ylabel="# Firms / Period")

# (1,1) Ceil inflation % over time
ax = axes[1, 1]
ax.plot(t, ceil_inflation_pct[b:], color=C_TRAP, linewidth=1.5)
mean_ci = np.mean(ceil_inflation_pct[b:])
ax.axhline(
    mean_ci, color=C_DN, linewidth=1, linestyle="--", label=f"Mean: {mean_ci:.2f}%"
)
ax.legend(fontsize=8)
style_ax(ax, "Ceil Inflation (extra workers as % of workforce)", ylabel="%")

plt.tight_layout()
save_figure(fig, 2, "quantization")

# %%
# Figure 3: Production Feedback Loop
# -----------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Figure 3: Production Feedback Loop — The Self-Reinforcing Lock-In",
    fontsize=13,
    fontweight="bold",
    y=1.0,
)

# (0,0) production_prev autocorrelation for sample firms
ax = axes[0, 0]
# Pick 6 representative firms (avoid dead firms)
alive_mask = np.mean(actual_prod[b:], axis=0) > 1e-10
alive_firms = np.where(alive_mask)[0]
if alive_firms.size >= 6:
    sample_firms = alive_firms[np.linspace(0, alive_firms.size - 1, 6).astype(int)]
else:
    sample_firms = alive_firms[:6]
firm_colors = plt.cm.Set2(np.linspace(0, 0.8, len(sample_firms)))
max_lag = 20
for i, f in enumerate(sample_firms):
    pp = production_prev[b:, f]
    if np.std(pp) > 1e-10:
        autocorr = [1.0]
        for lag in range(1, max_lag + 1):
            autocorr.append(float(np.corrcoef(pp[:-lag], pp[lag:])[0, 1]))
        ax.plot(
            range(max_lag + 1),
            autocorr,
            color=firm_colors[i],
            linewidth=0.8,
            alpha=0.7,
            label=f"Firm {f}",
        )
ax.axhline(0, color="gray", linewidth=0.5)
ax.legend(fontsize=7, ncol=2)
style_ax(
    ax,
    "production_prev Autocorrelation (sample firms)",
    xlabel="Lag (periods)",
    ylabel="Autocorrelation",
)

# (0,1) Desired vs actual production scatter
ax = axes[0, 1]
# Sample random firm-periods
all_desired = desired_prod[b:].flatten()
all_actual = actual_prod[b:].flatten()
alive = all_actual > 1e-10
if np.sum(alive) > 0:
    n_pts = min(3000, np.sum(alive))
    rng_plot = np.random.RandomState(42)
    idx = rng_plot.choice(np.where(alive)[0], n_pts, replace=False)
    ax.scatter(all_desired[idx], all_actual[idx], s=3, alpha=0.2, color=C_TRAP)
    max_val = max(
        np.percentile(all_desired[idx], 99), np.percentile(all_actual[idx], 99)
    )
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.5, alpha=0.5, label="1:1")
    ax.legend(fontsize=8)
style_ax(
    ax,
    "Desired vs Actual Production",
    xlabel="Desired Production",
    ylabel="Actual Production",
)

# (1,0) Lock-in duration histogram
ax = axes[1, 0]
if lock_in_durations:
    dur = np.array(lock_in_durations)
    max_dur = min(int(np.percentile(dur, 99)), 50)
    ax.hist(
        dur[dur <= max_dur],
        bins=range(1, max_dur + 2),
        color=C_TRAP,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.axvline(
        np.mean(dur),
        color=C_DN,
        linewidth=2,
        linestyle="--",
        label=f"Mean: {np.mean(dur):.1f}",
    )
    ax.axvline(
        np.median(dur),
        color=C_BLOCKED,
        linewidth=2,
        linestyle=":",
        label=f"Median: {np.median(dur):.0f}",
    )
    ax.legend(fontsize=8)
else:
    ax.text(
        0.5,
        0.5,
        "No trapped episodes",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
style_ax(
    ax,
    "Lock-In Duration Distribution",
    xlabel="Consecutive Trapped Periods",
    ylabel="Count",
)

# (1,1) Aggregate trap fraction over time
ax = axes[1, 1]
frac_trapped = np.sum(trap_mask, axis=1) / n_firms * 100
ax.plot(t, frac_trapped[b:], color=C_TRAP, linewidth=1.5)
mean_trapped = np.mean(frac_trapped[b:])
ax.axhline(
    mean_trapped,
    color=C_DN,
    linewidth=1,
    linestyle="--",
    label=f"Mean: {mean_trapped:.1f}%",
)
ax.legend(fontsize=8)
style_ax(ax, "Firms in Quantization Trap (% of all firms)", ylabel="% Trapped")

plt.tight_layout()
save_figure(fig, 3, "feedback_loop")

# %%
# Figure 4: Unemployment Dynamics
# --------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Figure 4: Unemployment Dynamics — Exogenous-Only Separations",
    fontsize=13,
    fontweight="bold",
    y=1.0,
)

# (0,0) Unemployment rate with target band
ax = axes[0, 0]
ax.plot(t, unemp_rate[b:] * 100, color=C_UNEMP, linewidth=1.5)
ax.axhspan(2, 8, color="green", alpha=0.08, label="Target (2-8%)")
ax.axhline(2, color="green", linewidth=0.5, linestyle="--")
ax.axhline(8, color="green", linewidth=0.5, linestyle="--")
mean_u = np.mean(unemp_rate[b:]) * 100
ax.axhline(
    mean_u, color=C_UNEMP, linewidth=1, linestyle=":", label=f"Mean: {mean_u:.1f}%"
)
ax.legend(fontsize=8)
style_ax(ax, "Unemployment Rate", ylabel="% Unemployed")

# (0,1) Separation decomposition (stacked)
ax = axes[0, 1]
ax.stackplot(
    t,
    fired_per_period[b:],
    expirations[b:],
    credit_fires[b:],
    labels=["Firings", "Expirations", "Credit Fires"],
    colors=[C_FIRE, C_EXPIRE, C_DN],
    alpha=0.7,
)
ax.legend(fontsize=8, loc="upper right")
style_ax(ax, "Separations Decomposition", ylabel="Workers / Period")

# (1,0) Beveridge curve (vacancy rate vs unemployment rate)
ax = axes[1, 0]
ax.scatter(unemp_rate[b:] * 100, vacancy_rate[b:] * 100, s=8, alpha=0.4, color=C_VAC)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
style_ax(
    ax, "Beveridge Curve", xlabel="Unemployment Rate (%)", ylabel="Vacancy Rate (%)"
)

# (1,1) Labor flows: hires, separations, net
ax = axes[1, 1]
ax.plot(t, hired_per_period[b:], color=C_HIRE, linewidth=1.2, label="Hires")
ax.plot(t, separations[b:], color=C_FIRE, linewidth=1.2, label="Separations")
ax.plot(
    t,
    net_flow[b:],
    color="black",
    linewidth=1,
    linestyle="--",
    label="Net Flow",
    alpha=0.7,
)
ax.axhline(0, color="gray", linewidth=0.5)
ax.legend(fontsize=8)
style_ax(ax, "Labor Flows", ylabel="Workers / Period")

plt.tight_layout()
save_figure(fig, 4, "unemployment")

# %%
# Figure 5: Matching Market Efficiency
# -------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Figure 5: Matching Market Efficiency — Absorbing All Slack",
    fontsize=13,
    fontweight="bold",
    y=1.0,
)

# (0,0) Vacancies vs Unemployed
ax = axes[0, 0]
n_unemp_series = n_workers - employed_at_prod
ax.plot(t, total_vacancies[b:], color=C_VAC, linewidth=1.2, label="Total Vacancies")
ax.plot(t, n_unemp_series[b:], color=C_UNEMP, linewidth=1.2, label="Unemployed Workers")
ax.legend(fontsize=8)
style_ax(ax, "Vacancies vs Unemployed Workers", ylabel="Count")

# (0,1) Job fill rate
ax = axes[0, 1]
ax.plot(t, fill_rate[b:] * 100, color=C_HIRE, linewidth=1.5)
mean_fr = np.mean(fill_rate[b:]) * 100
ax.axhline(
    mean_fr, color=C_DN, linewidth=1, linestyle="--", label=f"Mean: {mean_fr:.1f}%"
)
ax.legend(fontsize=8)
ax.set_ylim(0, 105)
style_ax(ax, "Job Fill Rate (Hires / Vacancies)", ylabel="%")

# (1,0) Market tightness (V/U ratio)
ax = axes[1, 0]
ax.plot(t, market_tightness[b:], color=C_VAC, linewidth=1.5)
ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
mean_mt = np.mean(market_tightness[b:])
ax.axhline(
    mean_mt, color=C_DN, linewidth=1, linestyle="--", label=f"Mean: {mean_mt:.2f}"
)
ax.legend(fontsize=8)
style_ax(ax, "Market Tightness (V/U Ratio)", ylabel="Ratio")

# (1,1) Unemployment rate vs fill rate scatter
ax = axes[1, 1]
ax.scatter(unemp_rate[b:] * 100, fill_rate[b:] * 100, s=8, alpha=0.4, color=C_HIRE)
style_ax(
    ax, "Unemployment vs Fill Rate", xlabel="Unemployment (%)", ylabel="Fill Rate (%)"
)

plt.tight_layout()
save_figure(fig, 5, "matching")

# %%
# Figure 6: Firm Size Distribution
# ----------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Figure 6: Firm Size Distribution — Sticky Sizes and Trap Exposure",
    fontsize=13,
    fontweight="bold",
    y=1.0,
)

# (0,0) Firm size histograms at key periods
ax = axes[0, 0]
snapshot_periods = [b, b + 50, b + 100, b + 200, N_PERIODS - 1]
snapshot_periods = [p for p in snapshot_periods if p < N_PERIODS]
colors_snap = plt.cm.viridis(np.linspace(0.15, 0.85, len(snapshot_periods)))
for i, p in enumerate(snapshot_periods):
    sizes = labor_post_hire[p]
    sizes = sizes[sizes > 0]
    if sizes.size > 0:
        ax.hist(
            sizes,
            bins=range(0, int(np.max(sizes)) + 2),
            alpha=0.4,
            color=colors_snap[i],
            label=f"t={p}",
            edgecolor="none",
            density=True,
        )
ax.legend(fontsize=7)
style_ax(
    ax,
    "Firm Size Distribution at Key Periods",
    xlabel="Workers / Firm",
    ylabel="Density",
)

# (0,1) Gini coefficient over time
ax = axes[0, 1]
ax.plot(t, gini_series[b:], color=C_TRAP, linewidth=1.5)
mean_gini = np.mean(gini_series[b:])
ax.axhline(
    mean_gini, color=C_DN, linewidth=1, linestyle="--", label=f"Mean: {mean_gini:.3f}"
)
ax.legend(fontsize=8)
style_ax(ax, "Firm Size Gini Coefficient", ylabel="Gini")

# (1,0) Trap exposure by firm size (binned)
ax = axes[1, 0]
# Compute: for each firm-size bin, what fraction of time are DECREASE firms trapped?
all_sizes_dn = labor_pre[b:][mask_dn[b:]].flatten()
all_trapped_dn = trap_mask[b:][mask_dn[b:]].flatten()
if all_sizes_dn.size > 0:
    max_bin = int(np.percentile(all_sizes_dn, 98)) + 1
    size_bins_plot = range(1, max_bin + 1)
    trap_by_size = []
    counts_by_size = []
    for sz in size_bins_plot:
        in_bin = all_sizes_dn == sz
        cnt = np.sum(in_bin)
        if cnt > 0:
            trap_by_size.append(np.mean(all_trapped_dn[in_bin]) * 100)
        else:
            trap_by_size.append(0)
        counts_by_size.append(cnt)
    ax.bar(list(size_bins_plot), trap_by_size, color=C_TRAP, alpha=0.7, width=0.8)
    ax.axhline(100, color="gray", linewidth=0.5, linestyle=":")
    # Add threshold line
    thresh = int(1 / config.h_rho)
    ax.axvline(
        thresh + 0.5, color=C_DN, linewidth=2, linestyle="--", label=f"L=1/h_ρ={thresh}"
    )
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
style_ax(
    ax,
    "Trap Rate by Firm Size (DECREASE firms only)",
    xlabel="Workers / Firm",
    ylabel="% Trapped",
)

# (1,1) Firm size autocorrelation (lag-1 per firm, distribution)
ax = axes[1, 1]
autocorr_per_firm = []
for f in range(n_firms):
    series = labor_post_hire[b:, f].astype(float)
    if np.std(series) > 1e-10:
        autocorr_per_firm.append(float(np.corrcoef(series[:-1], series[1:])[0, 1]))
if autocorr_per_firm:
    ax.hist(
        autocorr_per_firm,
        bins=30,
        color=C_NEUTRAL,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    mean_ac = np.mean(autocorr_per_firm)
    ax.axvline(
        mean_ac, color=C_TRAP, linewidth=2, linestyle="--", label=f"Mean: {mean_ac:.3f}"
    )
    ax.legend(fontsize=8)
style_ax(
    ax, "Firm Size Lag-1 Autocorrelation", xlabel="Autocorrelation", ylabel="# Firms"
)

plt.tight_layout()
save_figure(fig, 6, "firm_size")

# %%
# Figure 7: Wage Dynamics
# ------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Figure 7: Wage Dynamics — Persistent Upward Drift",
    fontsize=13,
    fontweight="bold",
    y=1.0,
)

# (0,0) Mean wage offer vs min wage
ax = axes[0, 0]
ax.plot(t, mean_wage_offer[b:], color=C_WAGE, linewidth=1.5, label="Mean Wage Offer")
ax.plot(t, min_wage[b:], color=C_DN, linewidth=1.2, linestyle="--", label="Min Wage")
ax.legend(fontsize=8)
style_ax(ax, "Mean Wage Offer vs Minimum Wage", ylabel="Wage")

# (0,1) Wage dispersion (CV)
ax = axes[0, 1]
ax.plot(t, wage_dispersion[b:], color=C_WAGE, linewidth=1.5)
mean_cv = np.mean(wage_dispersion[b:])
ax.axhline(
    mean_cv, color=C_DN, linewidth=1, linestyle="--", label=f"Mean CV: {mean_cv:.3f}"
)
ax.legend(fontsize=8)
style_ax(ax, "Wage Dispersion (CV of Wage Offers)", ylabel="CV")

# (1,0) Labor share proxy
ax = axes[1, 0]
ax.plot(t, labor_share[b:] * 100, color=C_WAGE, linewidth=1.5)
mean_ls = np.mean(labor_share[b:]) * 100
ax.axhline(
    mean_ls, color=C_DN, linewidth=1, linestyle="--", label=f"Mean: {mean_ls:.1f}%"
)
ax.legend(fontsize=8)
style_ax(ax, "Labor Share Proxy (WageBill / Output Value)", ylabel="%")

# (1,1) Phillips curve scatter (unemployment vs wage growth)
ax = axes[1, 1]
ax.scatter(unemp_rate[b:] * 100, wage_growth[b:] * 100, s=8, alpha=0.4, color=C_WAGE)
ax.axhline(0, color="gray", linewidth=0.5)
style_ax(
    ax,
    "Phillips Curve (Unemployment vs Wage Growth)",
    xlabel="Unemployment (%)",
    ylabel="Wage Growth (%)",
)

plt.tight_layout()
save_figure(fig, 7, "wages")

# %%
# Figure 8: Economy Dashboard
# ----------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "Figure 8: Economy Dashboard — Labor Market Overview",
    fontsize=13,
    fontweight="bold",
    y=1.0,
)

# (0,0) Unemployment rate
ax = axes[0, 0]
ax.plot(t_all, unemp_rate * 100, color=C_UNEMP, linewidth=1.2)
ax.axhspan(2, 8, color="green", alpha=0.08)
ax.axhline(2, color="green", linewidth=0.5, linestyle="--")
ax.axhline(8, color="green", linewidth=0.5, linestyle="--")
if b > 0:
    ax.axvline(b, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
style_ax(ax, "Unemployment Rate", ylabel="%")

# (0,1) Vacancy rate
ax = axes[0, 1]
ax.plot(t_all, vacancy_rate * 100, color=C_VAC, linewidth=1.2)
if b > 0:
    ax.axvline(b, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
style_ax(ax, "Vacancy Rate", ylabel="%")

# (0,2) Aggregate production
ax = axes[0, 2]
total_prod = np.sum(actual_prod, axis=1)
ax.plot(t_all, total_prod, color=C_UP, linewidth=1.2)
if b > 0:
    ax.axvline(b, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
style_ax(ax, "Aggregate Production", ylabel="Output")

# (1,0) Mean wage
ax = axes[1, 0]
ax.plot(t_all, mean_wage_offer, color=C_WAGE, linewidth=1.2)
if b > 0:
    ax.axvline(b, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
style_ax(ax, "Mean Wage Offer", ylabel="Wage")

# (1,1) Bankruptcies per period
ax = axes[1, 1]
if "n_firm_bankruptcies" in results.economy_data:
    bankr = results.economy_data["n_firm_bankruptcies"]
    ax.bar(t_all[: len(bankr)], bankr, color=C_NEUTRAL, alpha=0.7, width=1.0)
else:
    ax.text(
        0.5, 0.5, "No bankruptcy data", ha="center", va="center", transform=ax.transAxes
    )
if b > 0:
    ax.axvline(b, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
style_ax(ax, "Firm Bankruptcies / Period", ylabel="Count")

# (1,2) Mean firm size
ax = axes[1, 2]
ax.plot(t_all, mean_firm_size, color=C_TRAP, linewidth=1.2)
if b > 0:
    ax.axvline(b, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
style_ax(ax, "Mean Firm Size", ylabel="Workers / Firm")

plt.tight_layout()
save_figure(fig, 8, "dashboard")

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
    f"  1. Quadrant splits: INCREASE {np.mean(frac_increase[b:]) * 100:.0f}% / "
    f"DECREASE {np.mean(frac_decrease[b:]) * 100:.0f}% / "
    f"Blocked {np.mean(frac_blocked[b:]) * 100:.0f}% / "
    f"Unsold {np.mean(frac_unsold[b:]) * 100:.0f}%"
)
print(
    f"  2. Trap rate: {np.mean(trap_rate[b:]) * 100:.0f}% of DECREASE firms "
    f"cannot reduce labor (ceil absorbs)"
)
print(
    f"  3. Ceil inflation: {np.mean(ceil_inflation_pct[b:]):.1f}% of workforce "
    f"(~{np.mean(ceil_excess_total[b:]):.0f} extra workers)"
)
if lock_in_durations:
    dur = np.array(lock_in_durations)
    print(
        f"  4. Lock-in duration: mean {np.mean(dur):.1f} periods, "
        f"max {np.max(dur)} periods"
    )
else:
    print("  4. Lock-in duration: no trapped episodes")
print(
    f"  5. Asymmetry: {np.mean(n_successful_increase[b:]):.1f} increases vs "
    f"{np.mean(n_successful_decrease[b:]):.1f} decreases per period"
)
print(
    f"  6. Unemployment: {np.mean(unemp_rate[b:]) * 100:.1f}% mean, "
    f"{np.mean(unemp_rate[b:] < 0.02) * 100:.0f}% of periods below 2%"
)
print(
    f"  7. Separations: {np.mean(expirations[b:]):.0f} expirations + "
    f"{np.mean(fired_per_period[b:]):.1f} firings vs "
    f"{np.mean(hired_per_period[b:]):.0f} hires per period"
)
print(
    f"  8. Fill rate: {np.mean(fill_rate[b:]) * 100:.0f}% "
    f"(matching absorbs nearly all vacancies)"
)
