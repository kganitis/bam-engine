"""
Beta Sweep: Effect of consumption propensity exponent on credit market.

Tests whether higher β (steeper propensity curve) affects firm profitability
and credit market activity.

Run with::

    MPLBACKEND=Agg python diagnostics/test_beta_sweep.py
"""

import numpy as np

import bamengine as bam

# ── Configuration ──────────────────────────────────────────────────────────

N_PERIODS = 1000
BURN_IN = 500
SEED = 42
BETA_VALUES = [1.0, 1.5, 2.5, 4.0, 6.0, 8.0, 10.0, 15.0]

# ── Collect config ─────────────────────────────────────────────────────────

COLLECT = {
    "Producer": ["production", "inventory", "price"],
    "Borrower": [
        "net_worth",
        "credit_demand",
        "gross_profit",
        "net_profit",
        "retained_profit",
    ],
    "Employer": ["wage_bill"],
    "Lender": ["equity_base", "credit_supply"],
    "Consumer": ["savings", "income", "income_to_spend", "propensity"],
    "Worker": ["wage", "employed"],
    "Economy": True,
    "capture_timing": {
        "Producer.production": "firms_run_production",
        "Producer.inventory": "consumers_finalize_purchases",
        "Producer.price": "firms_plan_price",
        "Borrower.net_worth": "firms_update_net_worth",
        "Borrower.credit_demand": "firms_decide_credit_demand",
        "Borrower.gross_profit": "firms_collect_revenue",
        "Borrower.net_profit": "firms_validate_debt_commitments",
        "Borrower.retained_profit": "firms_pay_dividends",
        "Employer.wage_bill": "firms_run_production",
        "Lender.equity_base": "firms_validate_debt_commitments",
        "Lender.credit_supply": "banks_decide_credit_supply",
        "Consumer.savings": "consumers_finalize_purchases",
        "Consumer.income": "workers_receive_wage",
        "Consumer.income_to_spend": "consumers_decide_income_to_spend",
        "Consumer.propensity": "consumers_calc_propensity",
        "Worker.wage": "workers_receive_wage",
        "Worker.employed": "firms_run_production",
    },
}


# ── Run sweep ──────────────────────────────────────────────────────────────


def run_scenario(beta_val):
    """Run one scenario and return metrics dict."""
    sim = bam.Simulation.init(
        seed=SEED, beta=beta_val, logging={"default_level": "WARNING"}
    )
    results = sim.run(n_periods=N_PERIODS, collect=COLLECT)

    actual = results.metadata["n_periods"]
    bi = min(BURN_IN, actual // 2)
    s = slice(bi, actual)

    # Extract arrays
    nw = results.get_array("Borrower", "net_worth")
    cd = results.get_array("Borrower", "credit_demand")
    wb = results.get_array("Employer", "wage_bill")
    gp = results.get_array("Borrower", "gross_profit")
    np_arr = results.get_array("Borrower", "net_profit")
    rp = results.get_array("Borrower", "retained_profit")
    prod = results.get_array("Producer", "production")
    inv = results.get_array("Producer", "inventory")
    price = results.get_array("Producer", "price")
    eq = results.get_array("Lender", "equity_base")
    inc = results.get_array("Consumer", "income")
    its = results.get_array("Consumer", "income_to_spend")
    prop = results.get_array("Consumer", "propensity")
    employed = results.get_array("Worker", "employed")

    inflation = results.economy_data.get("inflation", np.zeros(actual))

    # Unemployment
    unemp = 1.0 - np.mean(employed.astype(float), axis=1)

    # NW/WB ratio (firms with employees only)
    has_emp = wb > 1e-10
    safe_wb = np.where(has_emp, wb, 1.0)
    nw_wb = np.where(has_emp, nw / safe_wb, np.nan)
    nw_wb_final = np.nanmedian(nw_wb[-1])

    # Credit metrics
    n_borrowers = np.sum(cd[s] > 1e-10, axis=1)  # per period

    # Sell-through
    units_sold = prod - inv
    safe_prod = np.where(prod > 1e-10, prod, 1e-10)
    sell_through = np.where(prod > 1e-10, np.maximum(units_sold, 0) / safe_prod, 0.0)

    # Markup
    breakeven_approx = wb / safe_prod  # wage_bill / production ≈ unit cost
    safe_be = np.where(breakeven_approx > 1e-10, breakeven_approx, 1.0)
    markup = np.where(breakeven_approx > 1e-10, price / safe_be, 1.0)

    # Aggregate profit metrics
    neg_gp_frac = np.mean(gp[s] < -1e-10)

    # Savings rate: 1 - spending / disposable_income
    total_inc = np.sum(inc[s], axis=1)
    total_its = np.sum(its[s], axis=1)
    total_div = np.sum(np_arr[s] - rp[s], axis=1)  # dividends
    disp_inc = total_inc + np.maximum(total_div, 0)
    safe_disp = np.where(disp_inc > 1e-10, disp_inc, 1e-10)
    sav_rate = np.where(disp_inc > 1e-10, 1.0 - total_its / safe_disp, 0.0)

    # Leverage
    total_nw = np.sum(nw[s], axis=1)
    # Approximate credit outstanding from demand (not exact but proportional)
    total_credit = np.sum(cd[s], axis=1)  # credit demand ≈ lending (most gets filled)
    safe_nw = np.where(total_nw > 1e-10, total_nw, 1e-10)
    leverage = total_credit / safe_nw

    # Mean propensity
    mean_prop = np.mean(prop[s])

    # Demand/supply ratio
    total_budget = np.sum(its, axis=1)
    total_prod_value = np.sum(prod * price, axis=1)
    safe_pv = np.where(total_prod_value > 1e-10, total_prod_value, 1e-10)
    demand_supply = np.where(total_prod_value > 1e-10, total_budget / safe_pv, 0.0)

    return {
        "beta": beta_val,
        "collapsed": actual < N_PERIODS,
        "unemp": np.mean(unemp[s]) * 100,
        "infl": np.mean(inflation[s]) * 100,
        "gdp": np.mean(np.sum(prod[s], axis=1)),
        "nw_wb": nw_wb_final,
        "borrowers": np.mean(n_borrowers),
        "leverage": np.mean(leverage),
        "markup": np.mean(markup[s]),
        "sell_thr": np.mean(sell_through[s]) * 100,
        "neg_gp": neg_gp_frac * 100,
        "sav_rate": np.mean(sav_rate) * 100,
        "bank_eq": np.mean(eq[s]),
        "mean_prop": mean_prop,
        "dem_sup": np.mean(demand_supply[s]),
        "price_end": float(results.economy_data["avg_price"][-1]),
    }


# ── Main ───────────────────────────────────────────────────────────────────

print("=" * 90)
print("Beta Sweep: Effect of Consumption Propensity Exponent on Credit Market")
print("=" * 90)
print(f"Config: {N_PERIODS} periods, burn-in={BURN_IN}, seed={SEED}")
print(f"Testing beta = {BETA_VALUES}")
print()

# First show the propensity function at key savings ratios
print("Propensity function c = 1/(1 + tanh(SA/SA_avg)^β):")
print(f"{'β':>6s}", end="")
ratios = [0.5, 1.0, 1.5, 2.0, 3.0]
for r in ratios:
    print(f"  SA/avg={r}", end="")
print()
for b in BETA_VALUES:
    print(f"{b:6.1f}", end="")
    for r in ratios:
        c = 1.0 / (1.0 + np.tanh(r) ** b)
        print(f"  {c:9.3f}", end="")
    print()
print()

# Run all scenarios
all_results = []
for beta_val in BETA_VALUES:
    print(f"Running beta={beta_val:.1f}...", end=" ", flush=True)
    m = run_scenario(beta_val)
    status = "COLLAPSED" if m["collapsed"] else "OK"
    print(
        f"{status} — unemp={m['unemp']:.1f}%, NW/WB={m['nw_wb']:.1f}x, "
        f"borrowers={m['borrowers']:.1f}, sell={m['sell_thr']:.1f}%"
    )
    all_results.append(m)

# ── Print comparison table ─────────────────────────────────────────────────

print()
print("=" * 90)
print("COMPARISON TABLE")
print("=" * 90)

headers = [
    "β",
    "Unemp%",
    "Infl%",
    "GDP",
    "NW/WB",
    "Borr/p",
    "Levg",
    "Markup",
    "Sell%",
    "NegGP%",
    "SavR%",
    "BkEq",
    "Prop",
    "D/S",
    "Price",
]
fmt = [
    "{:.1f}",
    "{:.1f}",
    "{:.1f}",
    "{:.0f}",
    "{:.1f}",
    "{:.1f}",
    "{:.4f}",
    "{:.2f}",
    "{:.1f}",
    "{:.1f}",
    "{:.1f}",
    "{:.1f}",
    "{:.3f}",
    "{:.3f}",
    "{:.1f}",
]
keys = [
    "beta",
    "unemp",
    "infl",
    "gdp",
    "nw_wb",
    "borrowers",
    "leverage",
    "markup",
    "sell_thr",
    "neg_gp",
    "sav_rate",
    "bank_eq",
    "mean_prop",
    "dem_sup",
    "price_end",
]

# Header
print(" ".join(f"{h:>8s}" for h in headers))
print("-" * (9 * len(headers)))

# Rows
baseline_idx = BETA_VALUES.index(2.5)
for i, m in enumerate(all_results):
    row = []
    for k, f in zip(keys, fmt, strict=True):
        val = m[k]
        row.append(f.format(val))
    marker = " ◀ baseline" if i == baseline_idx else ""
    print(" ".join(f"{v:>8s}" for v in row) + marker)

# ── Analysis ───────────────────────────────────────────────────────────────

print()
print("=" * 90)
print("ANALYSIS")
print("=" * 90)

bl = all_results[baseline_idx]
print(
    f"\nBaseline (β={bl['beta']}): NW/WB={bl['nw_wb']:.1f}x, "
    f"sell-through={bl['sell_thr']:.1f}%, propensity={bl['mean_prop']:.3f}"
)

for m in all_results:
    if m["beta"] == bl["beta"]:
        continue
    dnw = m["nw_wb"] - bl["nw_wb"]
    dsell = m["sell_thr"] - bl["sell_thr"]
    dprop = m["mean_prop"] - bl["mean_prop"]
    print(
        f"\nβ={m['beta']:5.1f}: NW/WB={m['nw_wb']:5.1f}x ({dnw:+.1f}), "
        f"sell={m['sell_thr']:.1f}% ({dsell:+.1f}pp), "
        f"prop={m['mean_prop']:.3f} ({dprop:+.3f}), "
        f"borrowers={m['borrowers']:.1f}, "
        f"neg_GP={m['neg_gp']:.1f}%"
    )
