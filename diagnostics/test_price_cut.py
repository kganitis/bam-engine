"""
Test: price_cut_allow_increase=False effect on credit market.

When True (default), firms trying to cut prices can still end up with a price
increase if the breakeven floor exceeds the attempted cut price. Setting to
False prevents this, allowing genuine downward price flexibility.

Run with::

    MPLBACKEND=Agg python diagnostics/test_price_cut.py
"""

import numpy as np

import bamengine as bam

N_PERIODS = 1000
BURN_IN = 500
SEED = 42

COLLECT = {
    "Producer": ["production", "inventory", "price", "breakeven_price"],
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
        "Producer.breakeven_price": "firms_plan_breakeven_price",
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

SCENARIOS = [
    {"name": "BASELINE", "kwargs": {}},
    {"name": "no_price_inc", "kwargs": {"price_cut_allow_increase": False}},
]


def compute_metrics(name, results, sim):
    actual = results.metadata["n_periods"]
    bi = min(BURN_IN, actual // 2)
    s = slice(bi, actual)

    nw = results.get_array("Borrower", "net_worth")
    cd = results.get_array("Borrower", "credit_demand")
    wb = results.get_array("Employer", "wage_bill")
    gp = results.get_array("Borrower", "gross_profit")
    np_arr = results.get_array("Borrower", "net_profit")
    rp = results.get_array("Borrower", "retained_profit")
    prod = results.get_array("Producer", "production")
    inv = results.get_array("Producer", "inventory")
    price = results.get_array("Producer", "price")
    bep = results.get_array("Producer", "breakeven_price")
    eq = results.get_array("Lender", "equity_base")
    inc = results.get_array("Consumer", "income")
    its = results.get_array("Consumer", "income_to_spend")
    employed = results.get_array("Worker", "employed")
    inflation = results.economy_data.get("inflation", np.zeros(actual))
    n_firm_bankr = results.economy_data.get("n_firm_bankruptcies", np.zeros(actual))
    n_bank_bankr = results.economy_data.get("n_bank_bankruptcies", np.zeros(actual))

    unemp = 1.0 - np.mean(employed.astype(float), axis=1)

    # NW/WB ratio
    has_emp = wb > 1e-10
    safe_wb = np.where(has_emp, wb, 1.0)
    nw_wb = np.where(has_emp, nw / safe_wb, np.nan)

    # Sell-through
    units_sold = prod - inv
    safe_prod = np.where(prod > 1e-10, prod, 1e-10)
    sell_thr = np.where(prod > 1e-10, np.maximum(units_sold, 0) / safe_prod, 0.0)

    # Markup
    safe_be = np.where(bep > 1e-10, bep, 1.0)
    markup = np.where(bep > 1e-10, price / safe_be, 1.0)

    # Revenue >= WB fraction
    revenue = price * np.maximum(units_sold, 0)
    rev_ge_wb = np.mean(revenue[s] >= wb[s] - 1e-10)

    # Price floored by breakeven (price increase when trying to cut)
    at_floor = np.abs(price - bep) < 1e-6  # firms exactly at breakeven floor
    pct_at_floor = np.mean(at_floor[s]) * 100

    # Savings rate
    total_inc = np.sum(inc[s], axis=1)
    total_its = np.sum(its[s], axis=1)
    total_div = np.sum(np_arr[s] - rp[s], axis=1)
    disp_inc = total_inc + np.maximum(total_div, 0)
    safe_disp = np.where(disp_inc > 1e-10, disp_inc, 1e-10)
    sav_rate = np.where(disp_inc > 1e-10, 1.0 - total_its / safe_disp, 0.0)

    # Demand/supply ratio
    total_budget = np.sum(its, axis=1)
    total_prod_value = np.sum(prod * price, axis=1)
    safe_pv = np.where(total_prod_value > 1e-10, total_prod_value, 1e-10)
    dem_sup = np.where(total_prod_value > 1e-10, total_budget / safe_pv, 0.0)

    # Labor share
    total_wb = np.sum(wb, axis=1)
    nominal_gdp = np.sum(prod, axis=1) * results.economy_data["avg_price"]
    safe_ngdp = np.where(nominal_gdp > 1e-10, nominal_gdp, 1e-10)
    labor_share = np.where(nominal_gdp > 1e-10, total_wb / safe_ngdp, 0.0)

    # Employed wage
    emp_wages = np.where(employed, results.get_array("Worker", "wage"), 0.0)
    emp_count = np.sum(employed.astype(float), axis=1)
    safe_ec = np.where(emp_count > 0, emp_count, 1.0)
    avg_wage = np.where(emp_count > 0, np.sum(emp_wages, axis=1) / safe_ec, 0.0)
    real_wage = np.where(
        results.economy_data["avg_price"] > 0,
        avg_wage / results.economy_data["avg_price"],
        0.0,
    )

    return {
        "name": name,
        "collapsed": actual < N_PERIODS,
        "unemp": np.mean(unemp[s]) * 100,
        "infl": np.mean(inflation[s]) * 100,
        "gdp": np.mean(np.sum(prod[s], axis=1)),
        "nw_wb": np.nanmedian(nw_wb[-1]),
        "borrowers": np.mean(np.sum(cd[s] > 1e-10, axis=1)),
        "leverage": np.mean(
            np.sum(cd[s], axis=1) / np.maximum(np.sum(nw[s], axis=1), 1e-10)
        ),
        "markup": np.mean(markup[s]),
        "sell_thr": np.mean(sell_thr[s]) * 100,
        "neg_gp": np.mean(gp[s] < -1e-10) * 100,
        "rev_ge_wb": rev_ge_wb * 100,
        "pct_floor": pct_at_floor,
        "sav_rate": np.mean(sav_rate) * 100,
        "bank_eq": np.mean(eq[s]),
        "dem_sup": np.mean(dem_sup[s]),
        "lab_share": np.mean(labor_share[s]) * 100,
        "real_wage": np.mean(real_wage[s]),
        "firm_bk": np.mean(n_firm_bankr[s]),
        "bank_bk": np.mean(n_bank_bankr[s]),
        "price_end": float(results.economy_data["avg_price"][-1]),
    }


print("=" * 90)
print("Test: price_cut_allow_increase=False")
print("=" * 90)
print(f"Config: {N_PERIODS} periods, burn-in={BURN_IN}, seed={SEED}\n")

all_results = []
for sc in SCENARIOS:
    print(f"Running {sc['name']}...", end=" ", flush=True)
    sim = bam.Simulation.init(
        seed=SEED, logging={"default_level": "WARNING"}, **sc["kwargs"]
    )
    results = sim.run(n_periods=N_PERIODS, collect=COLLECT)
    m = compute_metrics(sc["name"], results, sim)
    print(
        f"OK — unemp={m['unemp']:.1f}%, NW/WB={m['nw_wb']:.1f}x, "
        f"sell={m['sell_thr']:.1f}%, markup={m['markup']:.2f}x"
    )
    all_results.append(m)

# ── Comparison table ───────────────────────────────────────────────────────

print("\n" + "=" * 90)
bl, ex = all_results[0], all_results[1]

rows = [
    ("Unemployment %", "unemp", ".1f"),
    ("Inflation %", "infl", ".1f"),
    ("GDP (real)", "gdp", ".0f"),
    ("NW/WB ratio", "nw_wb", ".1f"),
    ("Borrowers/period", "borrowers", ".1f"),
    ("Leverage", "leverage", ".4f"),
    ("Markup (P/breakeven)", "markup", ".2f"),
    ("Sell-through %", "sell_thr", ".1f"),
    ("Neg gross profit %", "neg_gp", ".1f"),
    ("Revenue >= WB %", "rev_ge_wb", ".1f"),
    ("Firms at breakeven floor %", "pct_floor", ".1f"),
    ("Savings rate %", "sav_rate", ".1f"),
    ("Bank equity (mean)", "bank_eq", ".1f"),
    ("Demand/Supply ratio", "dem_sup", ".3f"),
    ("Labor share %", "lab_share", ".1f"),
    ("Real wage", "real_wage", ".3f"),
    ("Firm bankruptcies/p", "firm_bk", ".2f"),
    ("Bank bankruptcies/p", "bank_bk", ".2f"),
    ("Price (final)", "price_end", ".1f"),
]

print(f"{'Metric':<30s} {'BASELINE':>12s} {'no_price_inc':>12s} {'Change':>12s}")
print("-" * 66)
for label, key, fmt in rows:
    bv = bl[key]
    ev = ex[key]
    delta = ev - bv
    sign = "+" if delta > 0 else ""
    print(
        f"{label:<30s} {format(bv, fmt):>12s} {format(ev, fmt):>12s} {sign}{format(delta, fmt):>11s}"
    )

print("\n" + "=" * 90)
print("INTERPRETATION")
print("=" * 90)
print("""
price_cut_allow_increase controls whether the breakeven floor can push
a firm's price UP when it's trying to CUT prices (due to excess inventory).

When True (default): price = max(attempted_cut, breakeven_floor)
  → Even firms with excess supply may see price increases
  → Creates a one-way inflation ratchet

When False: if firm wants to cut, price cannot increase
  → Genuine downward price flexibility
  → Firms with excess supply actually lower prices
  → More price competition → potentially lower markups → less profit
""")
