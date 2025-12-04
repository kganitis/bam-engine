"""
===============
Custom Pipeline
===============

This example demonstrates how to customize the event execution pipeline
using YAML configuration. Custom pipelines let you reorder events,
add custom events, or remove built-in events.

You'll learn to:

- Understand the default pipeline structure
- Create custom pipeline YAML files
- Use special syntax (repetition, interleaving)
- Add custom events to the pipeline
- Load and execute custom pipelines
"""

# %%
# What is the Pipeline?
# ---------------------
#
# The pipeline defines which events execute each period and in what order.
# BAM Engine's default pipeline has 8 phases:
#
# 1. **Planning**: Production targets, pricing
# 2. **Labor Market**: Job search and hiring
# 3. **Credit Market**: Loan applications and provision
# 4. **Production**: Wage payments, production
# 5. **Goods Market**: Shopping and consumption
# 6. **Revenue**: Sales revenue, debt repayment
# 7. **Bankruptcy**: Insolvency detection and exit
# 8. **Entry & Statistics**: New firms, unemployment rate

import tempfile
from pathlib import Path

import bamengine as bam

# Initialize simulation
sim = bam.Simulation.init(n_firms=50, n_households=250, seed=42)

print("Default pipeline has events like:")
print("  - firms_decide_desired_production")
print("  - workers_send_one_round (repeated max_M times)")
print("  - firms_hire_workers (interleaved with above)")
print("  - consumers_shop_one_round (repeated max_Z times)")
print("  - calc_unemployment_rate")

# %%
# Pipeline YAML Syntax
# --------------------
#
# Pipelines are defined in YAML with special syntax:
#
# **Simple event**: ``- event_name``
#
# **Repeated event**: ``- event_name x N``
#   Executes event N times
#
# **Interleaved events**: ``- event1 <-> event2 x N``
#   Alternates between events: [event1, event2, event1, event2, ...]
#
# **Parameter substitution**: ``{max_M}``, ``{max_H}``, ``{max_Z}``
#   Replaced with config values at load time

# Example pipeline YAML structure
example_yaml = """
events:
  # Simple events
  - firms_decide_desired_production
  - firms_calc_breakeven_price
  - firms_adjust_price

  # Repeated event (shopping rounds)
  - consumers_shop_one_round x {max_Z}

  # Interleaved events (job search/hiring)
  - workers_send_one_round <-> firms_hire_workers x {max_M}
"""

print("\nExample pipeline YAML:")
print(example_yaml)

# %%
# Creating a Custom Pipeline
# --------------------------
#
# Create a minimal pipeline for testing that skips unnecessary events.

# Minimal pipeline: just planning, production, and stats
minimal_pipeline = """
# Minimal pipeline for testing
# Skips labor market, credit market, goods market

events:
  # Planning phase
  - firms_decide_desired_production
  - firms_calc_breakeven_price
  - firms_adjust_price
  - update_avg_mkt_price

  # Skip labor/credit/goods markets for speed

  # Production (simplified)
  - firms_run_production

  # End of period stats
  - calc_unemployment_rate
"""

# Write to temp file
config_dir = Path(tempfile.mkdtemp())
pipeline_path = config_dir / "minimal_pipeline.yml"
pipeline_path.write_text(minimal_pipeline)

print(f"Created minimal pipeline at: {pipeline_path}")

# Load and run with custom pipeline
sim_minimal = bam.Simulation.init(
    n_firms=50,
    n_households=250,
    seed=42,
    pipeline_path=str(pipeline_path),
)

print("\nRunning with minimal pipeline...")
sim_minimal.run(n_periods=10)
print("Completed 10 periods")

# %%
# Removing Events
# ---------------
#
# Create a pipeline without dividends (all profits retained).

no_dividends_pipeline = """
# Pipeline without dividend payments
# Firms retain all profits

events:
  # Planning
  - firms_decide_desired_production
  - firms_calc_breakeven_price
  - firms_adjust_price
  - update_avg_mkt_price
  - calc_annual_inflation_rate
  - firms_decide_desired_labor
  - firms_decide_vacancies

  # Labor market
  - adjust_minimum_wage
  - firms_decide_wage_offer
  - workers_decide_firms_to_apply
  - workers_send_one_round <-> firms_hire_workers x {max_M}
  - firms_calc_wage_bill

  # Credit market
  - banks_decide_credit_supply
  - banks_decide_interest_rate
  - firms_decide_credit_demand
  - firms_calc_credit_metrics
  - firms_prepare_loan_applications
  - firms_send_one_loan_app <-> banks_provide_loans x {max_H}
  - firms_fire_workers

  # Production
  - firms_pay_wages
  - workers_receive_wage
  - firms_run_production
  - workers_update_contracts

  # Goods market
  - consumers_calc_propensity
  - consumers_decide_income_to_spend
  - consumers_decide_firms_to_visit
  - consumers_shop_one_round x {max_Z}
  - consumers_finalize_purchases

  # Revenue (NO DIVIDENDS!)
  - firms_collect_revenue
  - firms_validate_debt_commitments
  # - firms_pay_dividends  # REMOVED!

  # Bankruptcy
  - firms_update_net_worth
  - mark_bankrupt_firms
  - mark_bankrupt_banks

  # Entry
  - spawn_replacement_firms
  - spawn_replacement_banks
  - calc_unemployment_rate
"""

# Write and test
no_div_path = config_dir / "no_dividends_pipeline.yml"
no_div_path.write_text(no_dividends_pipeline)

print("Created pipeline without dividends")

# Compare with and without dividends
import matplotlib.pyplot as plt

# Default pipeline (with dividends)
sim_with_div = bam.Simulation.init(n_firms=100, n_households=500, seed=42)
borr_with_div = sim_with_div.get_role("Borrower")
nw_with_div = []
for _ in range(50):
    sim_with_div.step()
    nw_with_div.append(bam.ops.mean(borr_with_div.net_worth))

# Custom pipeline (no dividends)
sim_no_div = bam.Simulation.init(
    n_firms=100,
    n_households=500,
    seed=42,
    pipeline_path=str(no_div_path),
)
borr_no_div = sim_no_div.get_role("Borrower")
nw_no_div = []
for _ in range(50):
    sim_no_div.step()
    nw_no_div.append(bam.ops.mean(borr_no_div.net_worth))

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(nw_with_div, label="With Dividends (default)", linewidth=2)
ax.plot(nw_no_div, label="No Dividends (retained)", linewidth=2)
ax.set_xlabel("Period")
ax.set_ylabel("Mean Firm Net Worth")
ax.set_title("Effect of Dividend Policy on Firm Net Worth")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nFinal mean net worth:")
print(f"  With dividends: {nw_with_div[-1]:.2f}")
print(f"  No dividends: {nw_no_div[-1]:.2f}")

# %%
# Adding Custom Events
# --------------------
#
# Define custom events and add them to the pipeline.

from bamengine import event, ops


@event
class CollectSalesTax:
    """Collect sales tax from firm revenue."""

    def execute(self, sim):
        prod = sim.get_role("Producer")
        borr = sim.get_role("Borrower")

        # 5% sales tax on revenue
        tax_rate = 0.05
        # Approximate revenue as production * price
        revenue = ops.multiply(prod.production, prod.price)
        tax = ops.multiply(revenue, tax_rate)

        # Reduce net worth by tax
        borr.net_worth[:] = borr.net_worth - tax


@event
class PayUnemploymentBenefits:
    """Pay unemployment benefits to jobless workers."""

    def execute(self, sim):
        wrk = sim.get_role("Worker")
        cons = sim.get_role("Consumer")

        # Benefit = 40% of minimum wage for unemployed
        benefit = sim.ec.min_wage * 0.4
        unemployed = wrk.employer < 0

        # Add to consumer income
        cons.income[:] = cons.income + ops.where(unemployed, benefit, 0.0)


print("\nCustom events defined:")
print("  - collect_sales_tax")
print("  - pay_unemployment_benefits")

# Pipeline with custom events
custom_events_pipeline = """
# Pipeline with custom tax and benefits

events:
  # Planning
  - firms_decide_desired_production
  - firms_calc_breakeven_price
  - firms_adjust_price
  - update_avg_mkt_price
  - calc_annual_inflation_rate
  - firms_decide_desired_labor
  - firms_decide_vacancies

  # Labor market
  - adjust_minimum_wage
  - firms_decide_wage_offer
  - workers_decide_firms_to_apply
  - workers_send_one_round <-> firms_hire_workers x {max_M}
  - firms_calc_wage_bill

  # Credit market
  - banks_decide_credit_supply
  - banks_decide_interest_rate
  - firms_decide_credit_demand
  - firms_calc_credit_metrics
  - firms_prepare_loan_applications
  - firms_send_one_loan_app <-> banks_provide_loans x {max_H}
  - firms_fire_workers

  # Production
  - firms_pay_wages
  - workers_receive_wage
  - firms_run_production
  - workers_update_contracts

  # CUSTOM: Unemployment benefits (after wage receipt)
  - pay_unemployment_benefits

  # Goods market
  - consumers_calc_propensity
  - consumers_decide_income_to_spend
  - consumers_decide_firms_to_visit
  - consumers_shop_one_round x {max_Z}
  - consumers_finalize_purchases

  # Revenue
  - firms_collect_revenue

  # CUSTOM: Sales tax (after revenue collection)
  - collect_sales_tax

  - firms_validate_debt_commitments
  - firms_pay_dividends

  # Bankruptcy
  - firms_update_net_worth
  - mark_bankrupt_firms
  - mark_bankrupt_banks

  # Entry
  - spawn_replacement_firms
  - spawn_replacement_banks
  - calc_unemployment_rate
"""

custom_path = config_dir / "custom_events_pipeline.yml"
custom_path.write_text(custom_events_pipeline)

# Run with custom events
sim_custom = bam.Simulation.init(
    n_firms=100,
    n_households=500,
    seed=42,
    pipeline_path=str(custom_path),
)

print("\nRunning with custom events pipeline...")
sim_custom.run(n_periods=30)
print("Completed 30 periods with tax and benefits")
print(f"Final unemployment: {sim_custom.ec.unemp_rate_history[-1]:.2%}")

# %%
# Interleaved Events Explained
# ----------------------------
#
# The ``<->`` syntax interleaves two events for multi-round matching.
# This is used for market mechanisms with sequential rounds.

interleave_explanation = """
Interleave syntax: event1 <-> event2 x N

For N=4, this expands to:
  1. event1 (round 0)
  2. event2 (round 0)
  3. event1 (round 1)
  4. event2 (round 1)
  5. event1 (round 2)
  6. event2 (round 2)
  7. event1 (round 3)
  8. event2 (round 3)

Used in BAM for:
  - Labor market: workers_send_one_round <-> firms_hire_workers
  - Credit market: firms_send_one_loan_app <-> banks_provide_loans
"""
print(interleave_explanation)

# %%
# Parameter Substitution
# ----------------------
#
# Use ``{param}`` in pipeline YAML to substitute config values.

param_example = """
# Available parameters:
#   {max_M} - Job applications per worker per period
#   {max_H} - Loan applications per firm per period
#   {max_Z} - Shopping rounds per household per period

events:
  # 4 rounds of job search/hiring
  - workers_send_one_round <-> firms_hire_workers x {max_M}

  # 2 rounds of loan application
  - firms_send_one_loan_app <-> banks_provide_loans x {max_H}

  # 2 rounds of shopping
  - consumers_shop_one_round x {max_Z}

# The actual values come from:
#   - defaults.yml (max_M: 4, max_H: 2, max_Z: 2)
#   - Or your custom config
"""
print("Parameter substitution in pipelines:")
print(param_example)

# Test with different friction values
high_friction_pipeline = """
events:
  - firms_decide_desired_production
  - firms_calc_breakeven_price
  - firms_adjust_price
  - update_avg_mkt_price
  - firms_decide_desired_labor
  - firms_decide_vacancies
  - adjust_minimum_wage
  - firms_decide_wage_offer
  - workers_decide_firms_to_apply
  - workers_send_one_round <-> firms_hire_workers x {max_M}
  - firms_calc_wage_bill
  - banks_decide_credit_supply
  - banks_decide_interest_rate
  - firms_decide_credit_demand
  - firms_calc_credit_metrics
  - firms_prepare_loan_applications
  - firms_send_one_loan_app <-> banks_provide_loans x {max_H}
  - firms_fire_workers
  - firms_pay_wages
  - workers_receive_wage
  - firms_run_production
  - workers_update_contracts
  - consumers_calc_propensity
  - consumers_decide_income_to_spend
  - consumers_decide_firms_to_visit
  - consumers_shop_one_round x {max_Z}
  - consumers_finalize_purchases
  - firms_collect_revenue
  - firms_validate_debt_commitments
  - firms_pay_dividends
  - firms_update_net_worth
  - mark_bankrupt_firms
  - mark_bankrupt_banks
  - spawn_replacement_firms
  - spawn_replacement_banks
  - calc_unemployment_rate
"""

friction_path = config_dir / "friction_pipeline.yml"
friction_path.write_text(high_friction_pipeline)

# Low friction (many search rounds)
sim_low_friction = bam.Simulation.init(
    n_firms=100,
    n_households=500,
    max_M=6,
    max_H=4,
    max_Z=4,  # More search rounds
    seed=42,
    pipeline_path=str(friction_path),
)

# High friction (few search rounds)
sim_high_friction = bam.Simulation.init(
    n_firms=100,
    n_households=500,
    max_M=2,
    max_H=1,
    max_Z=1,  # Fewer rounds
    seed=42,
    pipeline_path=str(friction_path),
)

# Run both
n_periods = 50
unemp_low = []
unemp_high = []

for _ in range(n_periods):
    sim_low_friction.step()
    sim_high_friction.step()
    unemp_low.append(sim_low_friction.ec.unemp_rate_history[-1])
    unemp_high.append(sim_high_friction.ec.unemp_rate_history[-1])

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    bam.ops.multiply(bam.ops.asarray(unemp_low), 100),
    label="Low Friction (M=6, H=4, Z=4)",
    linewidth=2,
)
ax.plot(
    bam.ops.multiply(bam.ops.asarray(unemp_high), 100),
    label="High Friction (M=2, H=1, Z=1)",
    linewidth=2,
)
ax.set_xlabel("Period")
ax.set_ylabel("Unemployment Rate (%)")
ax.set_title("Effect of Search Frictions on Unemployment")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Cleanup
# -------

import shutil

shutil.rmtree(config_dir)
print("\nTemp files cleaned up.")

# %%
# Key Takeaways
# -------------
#
# - Pipelines define event execution order
# - Use YAML format with ``events:`` list
# - Special syntax: ``x N`` for repetition, ``<->`` for interleaving
# - Parameter substitution: ``{max_M}``, ``{max_H}``, ``{max_Z}``
# - Remove events by commenting/deleting from YAML
# - Add custom events by defining with ``@event`` and including in YAML
# - Load via ``pipeline_path`` parameter in ``Simulation.init()``
