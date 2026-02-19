"""
===================================
Comprehensive Diagnostics Dashboard
===================================

Deep diagnostic visualization of every meaningful variable and relationship
in the BAM baseline model. This script captures:

- All 7 core roles (~30 per-agent fields) with proper capture timing
- Custom CreditMetrics per-bank role via inline event hooks
- Weighted aggregation (production-weighted prices, employment-weighted wages)
- Distribution dynamics (KDE snapshots at 5 time points)
- Macroeconomic curves with time-color gradient and regression lines
- Credit network metrics (HHI, utilization, default losses)
- Stock-flow consistency checks

Output: 13 multi-panel figures saved to ``diagnostics/output/``.

Run with::

    python diagnostics/baseline_diagnostics.py

For non-interactive (CI/headless) usage::

    MPLBACKEND=Agg python diagnostics/baseline_diagnostics.py
"""

# %%
# Imports
# -------

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import bamengine as bam
from bamengine import Float, event, ops, role

# %%
# Constants
# ---------

N_PERIODS = 1000
BURN_IN = 500
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
SNAPSHOT_TIMES = np.linspace(BURN_IN, N_PERIODS - 1, 5, dtype=int)

# Color palette by economic phase
C_PROD = "#2E86AB"  # Blue - Production & Output
C_LABOR = "#A23B72"  # Magenta - Labor Market
C_CREDIT = "#6A994E"  # Green - Credit Market
C_PRICE = "#F18F01"  # Orange - Prices & Inflation
C_GOODS = "#8B5CF6"  # Purple - Goods Market & Consumption
C_REVENUE = "#E74C3C"  # Red - Revenue & Profits
C_NETWORK = "#0EA5E9"  # Cyan - Credit Network
C_BANKR = "#374151"  # Dark gray - Bankruptcy


# %%
# CreditMetrics Role
# ------------------
#
# Per-bank role capturing credit network structure each period.


@role
class CreditMetrics:
    """Per-bank credit network metrics computed each period."""

    n_loans: Float
    total_credit: Float
    market_share: Float
    credit_utilization: Float
    default_losses: Float
    hhi: Float
    avg_borrower_fragility: Float


# %%
# CreditMetrics Event Hooks
# -------------------------
#
# Two hooks inject credit network computation into the pipeline:
# 1. After firms_fire_workers: compute credit structure metrics
# 2. After firms_validate_debt_commitments: compute default losses
#
# Module-level storage (Simulation uses __slots__, no dynamic attributes).
_min_wage_history: list[float] = []
_pre_validation_equity: np.ndarray | None = None


@event(name="compute_credit_metrics", after="firms_fire_workers")
class ComputeCreditMetrics:
    """Compute per-bank credit network metrics from LoanBook internals."""

    def execute(self, sim):
        loans = sim.get_relationship("LoanBook")
        lender = sim.get_role("Lender")
        bor = sim.get_role("Borrower")
        cm = sim.get_role("CreditMetrics")
        n_banks = sim.n_banks

        # Track minimum wage history
        _min_wage_history.append(sim.ec.min_wage)

        s = loans.size
        if s == 0:
            ops.assign(cm.n_loans, 0.0)
            ops.assign(cm.total_credit, 0.0)
            ops.assign(cm.market_share, 0.0)
            ops.assign(cm.credit_utilization, 0.0)
            ops.assign(cm.hhi, 0.0)
            ops.assign(cm.avg_borrower_fragility, 0.0)
        else:
            targets = loans.target_ids[:s]
            principals = loans.principal[:s]

            # Loans per bank
            n_loans_arr = np.bincount(targets, minlength=n_banks).astype(float)
            ops.assign(cm.n_loans, n_loans_arr)

            # Total credit per bank
            total_credit_arr = np.zeros(n_banks)
            np.add.at(total_credit_arr, targets, principals)
            ops.assign(cm.total_credit, total_credit_arr)

            # Market share
            total = total_credit_arr.sum()
            if total > 0:
                ms = total_credit_arr / total
            else:
                ms = np.zeros(n_banks)
            ops.assign(cm.market_share, ms)

            # Credit utilization
            ops.assign(
                cm.credit_utilization,
                ops.divide(total_credit_arr, lender.credit_supply),
            )

            # HHI concentration (broadcast to all banks)
            hhi_val = float(np.sum(ms**2))
            cm.hhi[:] = hhi_val

            # Weighted average borrower fragility per bank
            borrower_frag = bor.projected_fragility[loans.source_ids[:s]]
            weighted_frag = borrower_frag * principals
            sum_wfrag = np.zeros(n_banks)
            np.add.at(sum_wfrag, targets, weighted_frag)
            sum_principal = np.zeros(n_banks)
            np.add.at(sum_principal, targets, principals)
            safe_principal = np.where(sum_principal > 0, sum_principal, 1.0)
            avg_frag = np.where(sum_principal > 0, sum_wfrag / safe_principal, 0.0)
            ops.assign(cm.avg_borrower_fragility, avg_frag)

        # Save pre-validation equity for default loss computation
        global _pre_validation_equity
        _pre_validation_equity = lender.equity_base.copy()


@event(name="compute_default_metrics", after="firms_validate_debt_commitments")
class ComputeDefaultMetrics:
    """Compute per-bank default losses from equity change during validation."""

    def execute(self, sim):
        lender = sim.get_role("Lender")
        cm = sim.get_role("CreditMetrics")
        equity_change = _pre_validation_equity - lender.equity_base
        ops.assign(cm.default_losses, ops.maximum(equity_change, 0.0))


# %%
# Helper Functions
# ----------------


def weighted_stats(values, weights=None, pct_bounds=None):
    """Compute per-period weighted mean, std, min, max.

    Parameters
    ----------
    values : ndarray, shape (n_periods, n_agents)
    weights : ndarray, shape (n_periods, n_agents), optional
    pct_bounds : tuple of (lo_pct, hi_pct), optional
        If given, use percentile bounds instead of absolute min/max.
        E.g., ``(1, 99)`` for 1st/99th percentile.

    Returns
    -------
    mean, std, lo, hi : each shape (n_periods,)
    """
    if weights is None:
        mean = np.mean(values, axis=1)
        std = np.std(values, axis=1)
    else:
        w_sum = np.sum(weights, axis=1, keepdims=True)
        w_safe = np.where(w_sum > 0, w_sum, 1.0)
        mean = np.sum(values * weights, axis=1) / w_safe.ravel()
        diff = values - mean[:, None]
        var = np.sum(weights * diff**2, axis=1) / w_safe.ravel()
        std = np.sqrt(np.maximum(var, 0.0))
    if pct_bounds is not None:
        lo = np.percentile(values, pct_bounds[0], axis=1)
        hi = np.percentile(values, pct_bounds[1], axis=1)
    else:
        lo = np.min(values, axis=1)
        hi = np.max(values, axis=1)
    return mean, std, lo, hi


def plot_band(ax, t, mean, std, lo, hi, color, label=None):
    """Plot mean line + std ribbon + min/max dashed lines."""
    ax.plot(t, mean, color=color, linewidth=1.2, label=label)
    ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)
    ax.plot(t, lo, color=color, linewidth=0.4, linestyle="--", alpha=0.4)
    ax.plot(t, hi, color=color, linewidth=0.4, linestyle="--", alpha=0.4)


def style_ax(ax, title, xlabel="Period", ylabel=""):
    """Apply consistent styling to an axis."""
    ax.set_title(title, fontsize=11, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_kde_snapshots(ax, data_2d, times, title, mask_2d=None):
    """Plot overlaid KDE curves at snapshot times."""
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(times)))
    for i, t in enumerate(times):
        vals = data_2d[t].copy()
        if mask_2d is not None:
            vals = vals[mask_2d[t]]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 3 or np.std(vals) < 1e-12:
            continue
        try:
            kde = gaussian_kde(vals)
            x_grid = np.linspace(vals.min(), vals.max(), 200)
            ax.plot(x_grid, kde(x_grid), color=colors[i], linewidth=1.2, label=f"t={t}")
        except np.linalg.LinAlgError:
            continue
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)


def scatter_regression(ax, x, y, periods, title, xlabel, ylabel):
    """Scatter plot with time-color gradient and OLS regression line."""
    valid = np.isfinite(x) & np.isfinite(y)
    xv, yv, pv = x[valid], y[valid], periods[valid]
    sc = ax.scatter(xv, yv, c=pv, cmap="viridis", s=10, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Period")
    if len(xv) > 2:
        m, b = np.polyfit(xv, yv, 1)
        x_range = np.array([xv.min(), xv.max()])
        ax.plot(x_range, m * x_range + b, "r--", linewidth=1.5, alpha=0.8)
        corr = np.corrcoef(xv, yv)[0, 1]
        title = f"{title} (r={corr:.2f})"
    style_ax(ax, title, xlabel, ylabel)


def save_figure(fig, fig_num, name):
    """Save figure and show."""
    filename = f"baseline_{fig_num:02d}_{name}.png"
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved {filename}")


# %%
# Simulation Setup
# ----------------

sim = bam.Simulation.init(seed=42, logging={"default_level": "INFO"})
sim.use_role(CreditMetrics, n_agents=sim.n_banks)
sim.use_events(ComputeCreditMetrics, ComputeDefaultMetrics)

print("Initialized baseline scenario with:")
print(f"  - {sim.n_firms} firms, {sim.n_households} households, {sim.n_banks} banks")

# %%
# Collection Configuration
# ------------------------
#
# Capture all role fields with canonical timing for each variable.

COLLECT_CONFIG = {
    "Producer": [
        "production",
        "production_prev",
        "inventory",
        "expected_demand",
        "desired_production",
        "labor_productivity",
        "breakeven_price",
        "price",
    ],
    "Employer": [
        "desired_labor",
        "current_labor",
        "wage_offer",
        "wage_bill",
        "n_vacancies",
    ],
    "Worker": ["wage", "employed", "periods_left"],
    "Consumer": ["income", "savings", "income_to_spend", "propensity"],
    "Borrower": [
        "net_worth",
        "total_funds",
        "credit_demand",
        "projected_fragility",
        "gross_profit",
        "net_profit",
        "retained_profit",
    ],
    "Lender": ["equity_base", "credit_supply", "interest_rate"],
    "Shareholder": ["dividends"],
    "CreditMetrics": [
        "n_loans",
        "total_credit",
        "market_share",
        "credit_utilization",
        "default_losses",
        "hhi",
        "avg_borrower_fragility",
    ],
    "Economy": True,
    "capture_timing": {
        # Producer
        "Producer.production": "firms_run_production",
        "Producer.production_prev": "firms_run_production",
        "Producer.inventory": "consumers_finalize_purchases",
        "Producer.expected_demand": "firms_decide_desired_production",
        "Producer.desired_production": "firms_decide_desired_production",
        "Producer.labor_productivity": "firms_run_production",
        "Producer.breakeven_price": "firms_calc_breakeven_price",
        "Producer.price": "firms_adjust_price",
        # Employer
        "Employer.desired_labor": "firms_decide_desired_labor",
        "Employer.current_labor": "firms_run_production",
        "Employer.wage_offer": "firms_decide_wage_offer",
        "Employer.wage_bill": "firms_run_production",
        "Employer.n_vacancies": "firms_decide_vacancies",
        # Worker
        "Worker.wage": "workers_receive_wage",
        "Worker.employed": "firms_run_production",
        "Worker.periods_left": "firms_run_production",
        # Consumer
        "Consumer.income": "workers_receive_wage",
        "Consumer.savings": "consumers_finalize_purchases",
        "Consumer.income_to_spend": "consumers_decide_income_to_spend",
        "Consumer.propensity": "consumers_calc_propensity",
        # Borrower
        "Borrower.net_worth": "firms_update_net_worth",
        "Borrower.total_funds": "compute_credit_metrics",
        "Borrower.credit_demand": "firms_decide_credit_demand",
        "Borrower.projected_fragility": "firms_calc_financial_fragility",
        "Borrower.gross_profit": "firms_collect_revenue",
        "Borrower.net_profit": "firms_validate_debt_commitments",
        "Borrower.retained_profit": "firms_pay_dividends",
        # Lender
        "Lender.equity_base": "firms_validate_debt_commitments",
        "Lender.credit_supply": "banks_decide_credit_supply",
        "Lender.interest_rate": "banks_decide_interest_rate",
        # Shareholder
        "Shareholder.dividends": "firms_pay_dividends",
        # CreditMetrics
        "CreditMetrics.n_loans": "compute_credit_metrics",
        "CreditMetrics.total_credit": "compute_credit_metrics",
        "CreditMetrics.market_share": "compute_credit_metrics",
        "CreditMetrics.credit_utilization": "compute_credit_metrics",
        "CreditMetrics.hhi": "compute_credit_metrics",
        "CreditMetrics.avg_borrower_fragility": "compute_credit_metrics",
        "CreditMetrics.default_losses": "compute_default_metrics",
    },
}

# %%
# Run Simulation
# --------------

results = sim.run(n_periods=N_PERIODS, collect=COLLECT_CONFIG)

actual_periods = results.metadata["n_periods"]
print(f"\nSimulation completed: {actual_periods} periods")
print(f"Runtime: {results.metadata['runtime_seconds']:.2f} seconds")
if actual_periods < N_PERIODS:
    print(f"WARNING: Simulation ended early at period {actual_periods}")
    N_PERIODS = actual_periods
    BURN_IN = min(BURN_IN, actual_periods // 2)
    SNAPSHOT_TIMES = np.linspace(BURN_IN, N_PERIODS - 1, 5, dtype=int)

# %%
# Extract Arrays
# --------------
#
# Pull all per-agent arrays from results. Each is 2D: (n_periods, n_agents).

# Producer (n_periods x n_firms)
production = results.get_array("Producer", "production")
production_prev = results.get_array("Producer", "production_prev")
inventory = results.get_array("Producer", "inventory")
expected_demand = results.get_array("Producer", "expected_demand")
desired_production = results.get_array("Producer", "desired_production")
labor_productivity = results.get_array("Producer", "labor_productivity")
breakeven_price = results.get_array("Producer", "breakeven_price")
price = results.get_array("Producer", "price")

# Employer (n_periods x n_firms)
desired_labor = results.get_array("Employer", "desired_labor")
current_labor = results.get_array("Employer", "current_labor")
wage_offer = results.get_array("Employer", "wage_offer")
wage_bill = results.get_array("Employer", "wage_bill")
n_vacancies = results.get_array("Employer", "n_vacancies")

# Worker (n_periods x n_households)
worker_wage = results.get_array("Worker", "wage")
worker_employed = results.get_array("Worker", "employed")

# Consumer (n_periods x n_households)
consumer_income = results.get_array("Consumer", "income")
consumer_savings = results.get_array("Consumer", "savings")
consumer_its = results.get_array("Consumer", "income_to_spend")
consumer_propensity = results.get_array("Consumer", "propensity")

# Borrower (n_periods x n_firms)
net_worth = results.get_array("Borrower", "net_worth")
total_funds = results.get_array("Borrower", "total_funds")
credit_demand_arr = results.get_array("Borrower", "credit_demand")
projected_fragility = results.get_array("Borrower", "projected_fragility")
gross_profit = results.get_array("Borrower", "gross_profit")
net_profit = results.get_array("Borrower", "net_profit")
retained_profit = results.get_array("Borrower", "retained_profit")

# Lender (n_periods x n_banks)
equity_base = results.get_array("Lender", "equity_base")
credit_supply = results.get_array("Lender", "credit_supply")
interest_rate = results.get_array("Lender", "interest_rate")

# Shareholder (n_periods x n_households)
hh_dividends = results.get_array("Shareholder", "dividends")

# CreditMetrics (n_periods x n_banks)
cm_n_loans = results.get_array("CreditMetrics", "n_loans")
cm_total_credit = results.get_array("CreditMetrics", "total_credit")
cm_market_share = results.get_array("CreditMetrics", "market_share")
cm_credit_util = results.get_array("CreditMetrics", "credit_utilization")
cm_default_losses = results.get_array("CreditMetrics", "default_losses")
cm_hhi = results.get_array("CreditMetrics", "hhi")
cm_avg_bfrag = results.get_array("CreditMetrics", "avg_borrower_fragility")

# Economy data (1D: n_periods)
avg_price = results.economy_data["avg_price"]
inflation = results.economy_data.get("inflation", np.zeros(N_PERIODS))
n_firm_bankr = results.economy_data.get("n_firm_bankruptcies", np.zeros(N_PERIODS))
n_bank_bankr = results.economy_data.get("n_bank_bankruptcies", np.zeros(N_PERIODS))

# %%
# Compute Derived Metrics
# -----------------------

# Aggregate GDP
total_gdp = np.sum(production, axis=1)
log_gdp = np.log(total_gdp + 1e-10)
nominal_gdp = total_gdp * avg_price  # GDP in monetary units
total_inventory = np.sum(inventory, axis=1)

# Unemployment and vacancies
employed_float = worker_employed.astype(float)
unemployment = 1.0 - np.mean(employed_float, axis=1)
total_vacancies = np.sum(n_vacancies, axis=1)
vacancy_rate = total_vacancies / sim.n_households

# Employment-weighted average wage
employed_wages = np.where(worker_employed, worker_wage, 0.0)
employed_count = np.sum(employed_float, axis=1)
avg_employed_wage = np.where(
    employed_count > 0,
    np.sum(employed_wages, axis=1) / employed_count,
    0.0,
)

# Minimum wage history
min_wage = np.array(_min_wage_history)

# Total wage bill
total_wage_bill = np.sum(wage_bill, axis=1)

# Capacity utilization per firm
safe_desired = np.where(desired_production > 0, desired_production, 1.0)
cap_util = np.where(desired_production > 0, production / safe_desired, 0.0)

# Markup per firm
safe_breakeven = np.where(breakeven_price > 0, breakeven_price, 1.0)
markup = np.where(breakeven_price > 0, price / safe_breakeven, 1.0)

# Price dispersion (coefficient of variation)
price_mean_ts = np.mean(price, axis=1)
price_std_ts = np.std(price, axis=1)
price_cov = np.where(price_mean_ts > 0, price_std_ts / price_mean_ts, 0.0)

# Real wage
real_wage = np.where(avg_price > 0, avg_employed_wage / avg_price, 0.0)

# Growth rates (length N_PERIODS - 1)
gdp_growth = np.diff(total_gdp) / np.maximum(total_gdp[:-1], 1e-10)
safe_wage_prev = np.where(avg_employed_wage[:-1] > 0, avg_employed_wage[:-1], 1.0)
wage_inflation = np.where(
    avg_employed_wage[:-1] > 0,
    np.diff(avg_employed_wage) / safe_wage_prev,
    0.0,
)
safe_unemp_prev = np.where(unemployment[:-1] > 1e-10, unemployment[:-1], 1.0)
unemployment_growth = np.where(
    unemployment[:-1] > 1e-10,
    np.diff(unemployment) / safe_unemp_prev,
    0.0,
)

# Credit market aggregates
total_credit_demand = np.sum(credit_demand_arr, axis=1)
total_credit_supply = np.sum(credit_supply, axis=1)
total_credit_outstanding = np.sum(cm_total_credit, axis=1)
total_net_worth = np.sum(net_worth, axis=1)
aggregate_leverage = np.where(
    total_net_worth > 0, total_credit_outstanding / total_net_worth, 0.0
)

# Revenue & profit aggregates
total_dividends = np.sum(hh_dividends, axis=1)
profit_rate = np.where(
    total_net_worth > 0, np.sum(net_profit, axis=1) / total_net_worth, 0.0
)
labor_share = np.where(nominal_gdp > 0, total_wage_bill / nominal_gdp, 0.0)
credit_to_gdp = np.where(nominal_gdp > 0, total_credit_outstanding / nominal_gdp, 0.0)
mean_cap_util = np.mean(cap_util, axis=1)
avg_markup = np.mean(markup, axis=1)

# Consumption aggregates
total_income = np.sum(consumer_income, axis=1)
disposable_income = total_income + total_dividends  # wages + dividends
savings_rate = np.where(
    disposable_income > 0,
    1.0 - np.sum(consumer_its, axis=1) / disposable_income,
    0.0,
)
investment_rate = np.where(nominal_gdp > 0, total_credit_demand / nominal_gdp, 0.0)

# Credit network scalars
hhi_ts = cm_hhi[:, 0]  # Same for all banks
total_active_loans = np.sum(cm_n_loans, axis=1)

# Bankruptcy
cum_firm_bankr = np.cumsum(n_firm_bankr)
rolling_window = 20
kernel = np.ones(rolling_window) / rolling_window
rolling_bankr_rate = np.convolve(n_firm_bankr / sim.n_firms, kernel, mode="same")

# Period axis (post-burn-in)
bi = BURN_IN
t_post = np.arange(bi, N_PERIODS)
t_all = np.arange(N_PERIODS)

print(f"\nKey metrics (post burn-in, periods {bi}-{N_PERIODS}):")
print(f"  Mean unemployment: {np.mean(unemployment[bi:]) * 100:.1f}%")
print(f"  Mean inflation: {np.mean(inflation[bi:]) * 100:.1f}%")
print(f"  Mean GDP: {np.mean(total_gdp[bi:]):.0f}")
print("\nGenerating 13 diagnostic figures...")

# %%
# Figure 1: Production & Output
# -----------------------------

fig, axes = plt.subplots(4, 2, figsize=(14, 20))
fig.suptitle("Figure 1: Production & Output", fontsize=14, y=0.995)

# (0,0) Total GDP - linear
ax = axes[0, 0]
ax.plot(t_post, total_gdp[bi:], color=C_PROD, linewidth=1)
style_ax(ax, "Total GDP (linear)", ylabel="Output")

# (0,1) Total GDP - log
ax = axes[0, 1]
ax.plot(t_post, total_gdp[bi:], color=C_PROD, linewidth=1)
ax.set_yscale("log")
style_ax(ax, "Total GDP (log scale)", ylabel="Output")

# (1,0) Desired production per firm
ax = axes[1, 0]
m, s, lo, hi = weighted_stats(desired_production)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_PROD)
style_ax(ax, "Desired Production per Firm", ylabel="Output")

# (1,1) Actual production per firm
ax = axes[1, 1]
m, s, lo, hi = weighted_stats(production)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_PROD)
style_ax(ax, "Actual Production per Firm", ylabel="Output")

# (2,0) Capacity utilization per firm
ax = axes[2, 0]
m, s, lo, hi = weighted_stats(cap_util)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_PROD)
style_ax(ax, "Capacity Utilization (prod/desired)", ylabel="Ratio")

# (2,1) Total inventory
ax = axes[2, 1]
ax.plot(t_post, total_inventory[bi:], color=C_PROD, linewidth=1)
style_ax(ax, "Total Inventory", ylabel="Units")

# (3,0) Labor productivity (production-weighted)
ax = axes[3, 0]
m, s, lo, hi = weighted_stats(labor_productivity, weights=production)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_PROD)
style_ax(ax, "Labor Productivity (prod-weighted)", ylabel="Output/worker")

# (3,1) Expected demand per firm
ax = axes[3, 1]
m, s, lo, hi = weighted_stats(expected_demand)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_PROD)
style_ax(ax, "Expected Demand per Firm", ylabel="Units")

plt.tight_layout()
save_figure(fig, 1, "production_output")

# %%
# Figure 2: Labor Market
# ----------------------

fig, axes = plt.subplots(4, 2, figsize=(14, 20))
fig.suptitle("Figure 2: Labor Market", fontsize=14, y=0.995)

# (0,0) Unemployment rate
ax = axes[0, 0]
ax.plot(t_post, unemployment[bi:] * 100, color=C_LABOR, linewidth=1)
ax.set_ylim(bottom=0)
style_ax(ax, "Unemployment Rate", ylabel="%")

# (0,1) Vacancy rate
ax = axes[0, 1]
ax.plot(t_post, vacancy_rate[bi:] * 100, color=C_LABOR, linewidth=1)
ax.set_ylim(bottom=0)
style_ax(ax, "Vacancy Rate (vacancies / households)", ylabel="%")

# (1,0) Wage offer per firm
ax = axes[1, 0]
m, s, lo, hi = weighted_stats(wage_offer)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_LABOR)
style_ax(ax, "Wage Offer per Firm", ylabel="Wage")

# (1,1) Average employed wage
ax = axes[1, 1]
ax.plot(t_post, avg_employed_wage[bi:], color=C_LABOR, linewidth=1)
style_ax(ax, "Avg Employed Wage (employment-weighted)", ylabel="Wage")

# (2,0) Minimum wage
ax = axes[2, 0]
ax.plot(t_post, min_wage[bi:], color=C_LABOR, linewidth=1)
style_ax(ax, "Minimum Wage", ylabel="Wage")

# (2,1) Total wage bill
ax = axes[2, 1]
ax.plot(t_post, total_wage_bill[bi:], color=C_LABOR, linewidth=1)
style_ax(ax, "Total Wage Bill", ylabel="Amount")

# (3,0) Current labor per firm
ax = axes[3, 0]
m, s, lo, hi = weighted_stats(current_labor.astype(float))
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_LABOR)
style_ax(ax, "Current Labor per Firm", ylabel="Workers")

# (3,1) Desired labor per firm
ax = axes[3, 1]
m, s, lo, hi = weighted_stats(desired_labor.astype(float))
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_LABOR)
style_ax(ax, "Desired Labor per Firm", ylabel="Workers")

plt.tight_layout()
save_figure(fig, 2, "labor_market")

# %%
# Figure 3: Credit Market
# -----------------------

fig, axes = plt.subplots(4, 2, figsize=(14, 20))
fig.suptitle("Figure 3: Credit Market", fontsize=14, y=0.995)

# (0,0) Total credit demand vs supply
ax = axes[0, 0]
ax.plot(t_post, total_credit_demand[bi:], color=C_CREDIT, linewidth=1, label="Demand")
ax.plot(t_post, total_credit_supply[bi:], color=C_REVENUE, linewidth=1, label="Supply")
ax.legend(fontsize=8)
style_ax(ax, "Credit Demand vs Supply", ylabel="Amount")

# (0,1) Interest rate per bank (credit-supply-weighted)
ax = axes[0, 1]
m, s, lo, hi = weighted_stats(interest_rate, weights=credit_supply)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_CREDIT)
style_ax(ax, "Interest Rate (supply-weighted)", ylabel="Rate")

# (1,0) Financial fragility per firm
ax = axes[1, 0]
m, s, lo, hi = weighted_stats(projected_fragility)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_CREDIT)
style_ax(ax, "Financial Fragility per Firm", ylabel="B/A ratio")

# (1,1) Credit demand per firm
ax = axes[1, 1]
m, s, lo, hi = weighted_stats(credit_demand_arr)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_CREDIT)
style_ax(ax, "Credit Demand per Firm", ylabel="Amount")

# (2,0) Total funds per firm
ax = axes[2, 0]
m, s, lo, hi = weighted_stats(total_funds)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_CREDIT)
style_ax(ax, "Total Funds per Firm", ylabel="Amount")

# (2,1) Net worth per firm (linear)
ax = axes[2, 1]
m, s, lo, hi = weighted_stats(net_worth)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_CREDIT)
style_ax(ax, "Net Worth per Firm (linear)", ylabel="Amount")

# (3,0) Net worth per firm (log scale)
ax = axes[3, 0]
m, _, _, _ = weighted_stats(net_worth)
ax.plot(t_post, m[bi:], color=C_CREDIT, linewidth=1.2)
ax.set_yscale("log")
style_ax(ax, "Net Worth per Firm (log)", ylabel="Amount")

# (3,1) Aggregate leverage
ax = axes[3, 1]
ax.plot(t_post, aggregate_leverage[bi:], color=C_CREDIT, linewidth=1)
style_ax(ax, "Aggregate Leverage (debt/net worth)", ylabel="Ratio")

plt.tight_layout()
save_figure(fig, 3, "credit_market")

# %%
# Figure 4: Prices & Inflation
# ----------------------------

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle("Figure 4: Prices & Inflation", fontsize=14, y=0.998)

# (0,0) Price per firm (production-weighted, percentile bounds to clip zombie outliers)
ax = axes[0, 0]
m, s, lo, hi = weighted_stats(price, weights=production, pct_bounds=(1, 99))
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_PRICE)
style_ax(ax, "Price per Firm (prod-weighted)", ylabel="Price")

# (0,1) Breakeven price per firm (percentile bounds to clip zombie outliers)
ax = axes[0, 1]
m, s, lo, hi = weighted_stats(breakeven_price, pct_bounds=(1, 99))
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_PRICE)
style_ax(ax, "Breakeven Price per Firm", ylabel="Price")

# (1,0) Markup (price / breakeven)
ax = axes[1, 0]
m, s, lo, hi = weighted_stats(markup)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_PRICE)
ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
style_ax(ax, "Markup (price / breakeven)", ylabel="Ratio")

# (1,1) Inflation rate
ax = axes[1, 1]
ax.plot(t_all, inflation * 100, color=C_PRICE, linewidth=1)
ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
style_ax(ax, "Inflation Rate", ylabel="%")

# (2,0) Real wage
ax = axes[2, 0]
ax.plot(t_post, real_wage[bi:], color=C_PRICE, linewidth=1)
style_ax(ax, "Real Wage (nominal / avg price)", ylabel="Real wage")

# (2,1) Price dispersion (CoV)
ax = axes[2, 1]
ax.plot(t_post, price_cov[bi:], color=C_PRICE, linewidth=1)
style_ax(ax, "Price Dispersion (CoV = std/mean)", ylabel="CoV")

plt.tight_layout()
save_figure(fig, 4, "prices_inflation")

# %%
# Figure 5: Goods Market & Consumption
# -------------------------------------

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle("Figure 5: Goods Market & Consumption", fontsize=14, y=0.998)

# (0,0) Consumer propensity
ax = axes[0, 0]
m, s, lo, hi = weighted_stats(consumer_propensity)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_GOODS)
style_ax(ax, "Consumer Propensity", ylabel="Propensity")

# (0,1) Income to spend per household
ax = axes[0, 1]
m, s, lo, hi = weighted_stats(consumer_its)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_GOODS)
style_ax(ax, "Income to Spend per Household", ylabel="Amount")

# (1,0) Consumer savings per household
ax = axes[1, 0]
m, s, lo, hi = weighted_stats(consumer_savings)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_GOODS)
style_ax(ax, "Consumer Savings per Household", ylabel="Amount")

# (1,1) Consumer income per household
ax = axes[1, 1]
m, s, lo, hi = weighted_stats(consumer_income)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_GOODS)
style_ax(ax, "Consumer Income per Household", ylabel="Amount")

# (2,0) Dividends per household
ax = axes[2, 0]
m, s, lo, hi = weighted_stats(hh_dividends)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_GOODS)
style_ax(ax, "Dividends per Household", ylabel="Amount")

# (2,1) Savings rate
ax = axes[2, 1]
ax.plot(t_post, savings_rate[bi:] * 100, color=C_GOODS, linewidth=1)
style_ax(ax, "Savings Rate (1 - spending/disposable income)", ylabel="%")

plt.tight_layout()
save_figure(fig, 5, "goods_consumption")

# %%
# Figure 6: Revenue & Profits
# ----------------------------

fig, axes = plt.subplots(4, 2, figsize=(14, 20))
fig.suptitle("Figure 6: Revenue & Profits", fontsize=14, y=0.995)

# (0,0) Gross profit per firm
ax = axes[0, 0]
m, s, lo, hi = weighted_stats(gross_profit)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_REVENUE)
style_ax(ax, "Gross Profit per Firm", ylabel="Amount")

# (0,1) Net profit per firm
ax = axes[0, 1]
m, s, lo, hi = weighted_stats(net_profit)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_REVENUE)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
style_ax(ax, "Net Profit per Firm", ylabel="Amount")

# (1,0) Retained profit per firm
ax = axes[1, 0]
m, s, lo, hi = weighted_stats(retained_profit)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_REVENUE)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
style_ax(ax, "Retained Profit per Firm", ylabel="Amount")

# (1,1) Total dividends paid
ax = axes[1, 1]
ax.plot(t_post, total_dividends[bi:], color=C_REVENUE, linewidth=1)
style_ax(ax, "Total Dividends Paid", ylabel="Amount")

# (2,0) Profit rate (aggregate)
ax = axes[2, 0]
ax.plot(t_post, profit_rate[bi:] * 100, color=C_REVENUE, linewidth=1)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
style_ax(ax, "Profit Rate (net profit / net worth)", ylabel="%")

# (2,1) Labor share
ax = axes[2, 1]
ax.plot(t_post, labor_share[bi:] * 100, color=C_REVENUE, linewidth=1)
style_ax(ax, "Labor Share (wage bill / nominal GDP)", ylabel="%")

# (3,0) Credit-to-GDP ratio
ax = axes[3, 0]
ax.plot(t_post, credit_to_gdp[bi:], color=C_REVENUE, linewidth=1)
style_ax(ax, "Credit-to-GDP Ratio", ylabel="Ratio")

# (3,1) Mean capacity utilization
ax = axes[3, 1]
ax.plot(t_post, mean_cap_util[bi:] * 100, color=C_REVENUE, linewidth=1)
style_ax(ax, "Mean Capacity Utilization", ylabel="%")

plt.tight_layout()
save_figure(fig, 6, "revenue_profits")

# %%
# Figure 7: Credit Network
# ------------------------

fig, axes = plt.subplots(4, 2, figsize=(14, 20))
fig.suptitle("Figure 7: Credit Network", fontsize=14, y=0.995)

# (0,0) HHI credit concentration
ax = axes[0, 0]
ax.plot(t_post, hhi_ts[bi:], color=C_NETWORK, linewidth=1)
style_ax(ax, "HHI Credit Concentration", ylabel="HHI")

# (0,1) Total active loans
ax = axes[0, 1]
ax.plot(t_post, total_active_loans[bi:], color=C_NETWORK, linewidth=1)
style_ax(ax, "Total Active Loans", ylabel="Count")

# (1,0) Total credit outstanding
ax = axes[1, 0]
ax.plot(t_post, total_credit_outstanding[bi:], color=C_NETWORK, linewidth=1)
style_ax(ax, "Total Credit Outstanding", ylabel="Amount")

# (1,1) Credit utilization per bank
ax = axes[1, 1]
m, s, lo, hi = weighted_stats(cm_credit_util)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_NETWORK)
style_ax(ax, "Credit Utilization per Bank", ylabel="Ratio")

# (2,0) Bank equity per bank
ax = axes[2, 0]
m, s, lo, hi = weighted_stats(equity_base)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_NETWORK)
style_ax(ax, "Bank Equity per Bank", ylabel="Amount")

# (2,1) Bank credit supply per bank
ax = axes[2, 1]
m, s, lo, hi = weighted_stats(credit_supply)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_NETWORK)
style_ax(ax, "Credit Supply per Bank", ylabel="Amount")

# (3,0) Market share per bank
ax = axes[3, 0]
m, s, lo, hi = weighted_stats(cm_market_share)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_NETWORK)
style_ax(ax, "Market Share per Bank", ylabel="Share")

# (3,1) Default losses per bank
ax = axes[3, 1]
m, s, lo, hi = weighted_stats(cm_default_losses)
plot_band(ax, t_post, m[bi:], s[bi:], lo[bi:], hi[bi:], C_NETWORK)
style_ax(ax, "Default Losses per Bank", ylabel="Amount")

plt.tight_layout()
save_figure(fig, 7, "credit_network")

# %%
# Figure 8: Bankruptcy & Entry
# ----------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Figure 8: Bankruptcy & Entry", fontsize=14, y=1.0)

# (0,0) Firm bankruptcies per period
ax = axes[0, 0]
ax.plot(t_post, n_firm_bankr[bi:], color=C_BANKR, linewidth=0.8)
style_ax(ax, "Firm Bankruptcies per Period", ylabel="Count")

# (0,1) Bank bankruptcies per period
ax = axes[0, 1]
ax.plot(t_post, n_bank_bankr[bi:], color=C_BANKR, linewidth=0.8)
style_ax(ax, "Bank Bankruptcies per Period", ylabel="Count")

# (1,0) Cumulative firm bankruptcies
ax = axes[1, 0]
ax.plot(t_all, cum_firm_bankr, color=C_BANKR, linewidth=1)
style_ax(ax, "Cumulative Firm Bankruptcies", ylabel="Total")

# (1,1) Bankruptcy rate (rolling average)
ax = axes[1, 1]
ax.plot(t_post, rolling_bankr_rate[bi:] * 100, color=C_BANKR, linewidth=1)
style_ax(ax, f"Firm Bankruptcy Rate ({rolling_window}-period rolling avg)", ylabel="%")

plt.tight_layout()
save_figure(fig, 8, "bankruptcy_entry")

# %%
# Figure 9: Macroeconomic Curves
# ------------------------------

fig, axes = plt.subplots(4, 2, figsize=(14, 20))
fig.suptitle("Figure 9: Macroeconomic Curves", fontsize=14, y=0.995)

periods_post = np.arange(bi, N_PERIODS)

# (0,0) Phillips curve
scatter_regression(
    axes[0, 0],
    unemployment[bi:],
    wage_inflation[bi - 1 :],
    periods_post,
    "Phillips Curve",
    "Unemployment Rate",
    "Wage Inflation Rate",
)

# (0,1) Okun curve
scatter_regression(
    axes[0, 1],
    unemployment_growth[bi - 1 :],
    gdp_growth[bi - 1 :],
    periods_post,
    "Okun Curve",
    "Unemployment Growth Rate",
    "GDP Growth Rate",
)

# (1,0) Beveridge curve
scatter_regression(
    axes[1, 0],
    unemployment[bi:],
    vacancy_rate[bi:],
    periods_post,
    "Beveridge Curve",
    "Unemployment Rate",
    "Vacancy Rate",
)

# (1,1) GDP growth distribution
ax = axes[1, 1]
ax.hist(gdp_growth[bi:], bins=30, edgecolor="black", alpha=0.7, color=C_PROD)
style_ax(ax, "GDP Growth Rate Distribution", xlabel="Growth Rate", ylabel="Frequency")

# (2,0) Firm size histogram (final period)
ax = axes[2, 0]
final_production = production[-1]
ax.hist(final_production, bins=15, edgecolor="black", alpha=0.7, color=C_PROD)
style_ax(
    ax, "Firm Size Distribution (final period)", xlabel="Production", ylabel="Freq"
)

# (2,1) Firm size rank-size (log-log)
ax = axes[2, 1]
sorted_prod = np.sort(final_production)[::-1]
rank = np.arange(1, len(sorted_prod) + 1)
ax.scatter(rank, sorted_prod, s=20, color=C_PROD, alpha=0.8)
ax.set_xscale("log")
ax.set_yscale("log")
style_ax(ax, "Rank-Size Plot (log-log)", xlabel="Rank", ylabel="Production")

# (3,0) Inflation distribution
ax = axes[3, 0]
ax.hist(inflation[bi:] * 100, bins=30, edgecolor="black", alpha=0.7, color=C_PRICE)
style_ax(ax, "Inflation Rate Distribution", xlabel="Inflation (%)", ylabel="Frequency")

# (3,1) Unemployment distribution
ax = axes[3, 1]
ax.hist(unemployment[bi:] * 100, bins=30, edgecolor="black", alpha=0.7, color=C_LABOR)
style_ax(ax, "Unemployment Rate Distribution", xlabel="Unemployment (%)", ylabel="Freq")

plt.tight_layout()
save_figure(fig, 9, "macro_curves")

# %%
# Figure 10: Derived Macro Ratios
# --------------------------------

fig, axes = plt.subplots(4, 2, figsize=(14, 20))
fig.suptitle("Figure 10: Derived Macro Ratios", fontsize=14, y=0.995)

panels = [
    (axes[0, 0], real_wage[bi:], "Real Wage", "Wage/price", C_PRICE),
    (axes[0, 1], labor_share[bi:] * 100, "Labor Share", "%", C_LABOR),
    (axes[1, 0], profit_rate[bi:] * 100, "Profit Rate", "%", C_REVENUE),
    (axes[1, 1], aggregate_leverage[bi:], "Aggregate Leverage", "Debt/NW", C_CREDIT),
    (axes[2, 0], avg_markup[bi:], "Average Markup", "Price/breakeven", C_PRICE),
    (axes[2, 1], credit_to_gdp[bi:], "Credit-to-GDP Ratio", "Ratio", C_CREDIT),
    (axes[3, 0], savings_rate[bi:] * 100, "Savings Rate", "%", C_GOODS),
    (
        axes[3, 1],
        investment_rate[bi:],
        "Investment Rate (credit/GDP)",
        "Ratio",
        C_CREDIT,
    ),
]

for ax, data, title, ylabel, color in panels:
    ax.plot(t_post, data, color=color, linewidth=1)
    if "Rate" in title or "Share" in title:
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    style_ax(ax, title, ylabel=ylabel)

plt.tight_layout()
save_figure(fig, 10, "derived_ratios")

# %%
# Figure 11: Distribution Dynamics -- Firms
# ------------------------------------------

fig, axes = plt.subplots(4, 2, figsize=(14, 20))
fig.suptitle("Figure 11: Distribution Dynamics \u2014 Firms", fontsize=14, y=0.995)

firm_kde_panels = [
    (axes[0, 0], production, "Production"),
    (axes[0, 1], price, "Price"),
    (axes[1, 0], net_worth, "Net Worth"),
    (axes[1, 1], labor_productivity, "Labor Productivity"),
    (axes[2, 0], wage_offer, "Wage Offer"),
    (axes[2, 1], projected_fragility, "Financial Fragility"),
    (axes[3, 0], desired_production, "Desired Production"),
    (axes[3, 1], inventory, "Inventory"),
]

for ax, data, title in firm_kde_panels:
    plot_kde_snapshots(ax, data, SNAPSHOT_TIMES, title)

plt.tight_layout()
save_figure(fig, 11, "dist_firms")

# %%
# Figure 12: Distribution Dynamics -- Households & Banks
# -------------------------------------------------------

fig, axes = plt.subplots(4, 2, figsize=(14, 20))
fig.suptitle(
    "Figure 12: Distribution Dynamics \u2014 Households & Banks", fontsize=14, y=0.995
)

# Households
plot_kde_snapshots(
    axes[0, 0],
    worker_wage,
    SNAPSHOT_TIMES,
    "Wage (employed only)",
    mask_2d=worker_employed,
)
plot_kde_snapshots(axes[0, 1], consumer_savings, SNAPSHOT_TIMES, "Savings")
plot_kde_snapshots(axes[1, 0], consumer_propensity, SNAPSHOT_TIMES, "Propensity")
plot_kde_snapshots(axes[1, 1], consumer_income, SNAPSHOT_TIMES, "Income")

# Banks
plot_kde_snapshots(axes[2, 0], equity_base, SNAPSHOT_TIMES, "Bank Equity")
plot_kde_snapshots(axes[2, 1], interest_rate, SNAPSHOT_TIMES, "Interest Rate")
plot_kde_snapshots(axes[3, 0], credit_supply, SNAPSHOT_TIMES, "Credit Supply")

# Firm-level gross profit (extra panel)
plot_kde_snapshots(axes[3, 1], gross_profit, SNAPSHOT_TIMES, "Gross Profit (firms)")

plt.tight_layout()
save_figure(fig, 12, "dist_households_banks")

# %%
# Figure 13: Stock-Flow Consistency
# -----------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Figure 13: Stock-Flow Consistency Checks", fontsize=14, y=1.0)

# (0,0) Total system wealth
total_wealth = (
    np.sum(net_worth, axis=1)
    + np.sum(equity_base, axis=1)
    + np.sum(consumer_savings, axis=1)
)
ax = axes[0, 0]
ax.plot(t_post, total_wealth[bi:], color=C_PROD, linewidth=1)
style_ax(ax, "Total System Wealth (NW + equity + savings)", ylabel="Amount")

# (0,1) Wage flow residual
wage_received = np.sum(np.where(worker_employed, worker_wage, 0.0), axis=1)
wage_residual = total_wage_bill - wage_received
ax = axes[0, 1]
ax.plot(t_post, wage_residual[bi:], color=C_LABOR, linewidth=1)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
style_ax(ax, "Wage Flow Residual (bill - received)", ylabel="Residual")

# (1,0) Dividend flow residual
firm_div_total = np.sum(net_profit, axis=1) - np.sum(retained_profit, axis=1)
hh_div_total = np.sum(hh_dividends, axis=1)
div_residual = firm_div_total - hh_div_total
ax = axes[1, 0]
ax.plot(t_post, div_residual[bi:], color=C_REVENUE, linewidth=1)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
style_ax(ax, "Dividend Flow Residual (firms - households)", ylabel="Residual")

# (1,1) All residuals overlaid
ax = axes[1, 1]
ax.plot(t_post, wage_residual[bi:], color=C_LABOR, linewidth=1, label="Wage residual")
ax.plot(
    t_post, div_residual[bi:], color=C_REVENUE, linewidth=1, label="Dividend residual"
)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.legend(fontsize=8)
style_ax(ax, "All Flow Residuals", ylabel="Residual")

plt.tight_layout()
save_figure(fig, 13, "stock_flow")

# %%
# Summary
# -------

print(f"\nDiagnostics complete! {13} figures saved to {OUTPUT_DIR}/")
print(f"  Wage residual (mean abs): {np.mean(np.abs(wage_residual[bi:])):.4f}")
print(f"  Dividend residual (mean abs): {np.mean(np.abs(div_residual[bi:])):.4f}")
