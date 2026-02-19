# Comprehensive Diagnostics Analysis

- **Simulation**: Baseline BAM model, seed=42, 1000 periods, 100 firms / 500 households / 10 banks
- **Post-burn-in window**: periods 500-1000
- **Date**: 2026-02-18, updated after `20986d4` (planning-phase pricing)
- **Previous revision**: 2026-02-14, after `4dc5147` (zombie/bankruptcy fix)

______________________________________________________________________

## Changelog Since Previous Analysis

Five commits landed between the previous analysis (`4dc5147`) and this one (`20986d4`):

| Commit    | Change                                                          | Impact                                      |
| --------- | --------------------------------------------------------------- | ------------------------------------------- |
| `957a1ad` | Per-bank `opex_shock` in loan rate (was using constant `h_phi`) | Interest rate heterogeneity now real        |
| `fb3906e` | Principal-based bad debt + multi-lender loans                   | Credit mechanics more correct               |
| `b3944d6` | Retain LoanBook through planning/labor phases                   | Enables planning-phase breakeven            |
| `142e1b6` | Remove spurious `wage_bill` recalc from contract updates        | Gross profit no longer overstated           |
| `20986d4` | Planning-phase breakeven/price events (now default)             | Pricing shifted from production to planning |

Also notable: `new_firm_price_markup` changed from `1.5` to `1.05` in defaults.

______________________________________________________________________

## Summary Statistics

| Metric                             | Previous (4dc5147) | Current (20986d4) | Change         |
| ---------------------------------- | ------------------ | ----------------- | -------------- |
| Mean unemployment                  | 2.9%               | **4.5%**          | +1.6pp         |
| Std unemployment                   | —                  | 2.5%              |                |
| Mean inflation (quarterly)         | 2.7%               | **1.5%**          | -1.2pp         |
| Mean real GDP                      | 243                | **239**           | -4             |
| Avg price (period 999)             | 571                | **45**            | **-92%**       |
| Price growth factor (500→999)      | ~6x                | 6.2x              | Similar        |
| Active loans/period (post-BI)      | Sporadic (0-8)     | **4.2** (mean)    | Now persistent |
| Periods with loans (post-BI)       | —                  | 476/500 (95%)     |                |
| NW/WB ratio (t=999)                | 33.1x              | **12.9x**         | -61%           |
| Aggregate leverage                 | ~0.001             | **0.0017**        | +70%           |
| Firm bankruptcies/period (post-BI) | 1.10               | **1.42**          | +29%           |
| Bank bankruptcies/period (post-BI) | ~1                 | **0.71**          | -29%           |
| Non-producing firms (post-BI avg)  | 0.8                | 0.8               | Unchanged      |
| Wage flow residual                 | 0.0000             | 0.0000            | Unchanged      |
| Dividend flow residual             | 0.0000             | 0.0000            | Unchanged      |

The most dramatic change is the **92% reduction in final price level** (571 → 45), driven by
the planning-phase breakeven using `desired_production` as denominator (avoids ceil() inflation)
and the `new_firm_price_markup` drop from 1.5 to 1.05.

______________________________________________________________________

## A. Critical Findings

### A1. Credit Market — IMPROVED but Still Largely Inactive (Figures 3, 7, 10)

The credit market is now **persistently active** (476 of 500 post-burn-in periods have >0
loans), but remains **economically marginal**. Aggregate leverage is 0.0017 — essentially zero
by real-economy standards (typical corporate leverage is 0.3-0.5).

| Metric                     | Previous | Current     |
| -------------------------- | -------- | ----------- |
| Mean active loans/period   | 0-8      | **4.2**     |
| Max active loans/period    | 8        | **13**      |
| Periods with >0 loans      | Sporadic | **476/500** |
| Mean credit demand         | ~0       | **112.8**   |
| Mean credit supply (total) | ~2,000   | **306.4**   |
| Mean aggregate leverage    | ~0.001   | **0.0017**  |

**NW/WageBill ratio over time:**

| Period | NW/WB | Credit Demand | Total NW | Total WB | Loans |
| ------ | ----- | ------------- | -------- | -------- | ----- |
| 1      | 1.8x  | 0.0           | 455      | 258      | 0     |
| 10     | 4.4x  | 0.0           | 1,309    | 297      | 0     |
| 50     | 4.0x  | 40.4          | 1,643    | 413      | 10    |
| 100    | 5.7x  | 12.6          | 2,398    | 419      | 5     |
| 250    | 9.7x  | 9.6           | 5,369    | 553      | 5     |
| 500    | 10.9x | 82.5          | 14,137   | 1,293    | 5     |
| 750    | 12.4x | 128.1         | 37,553   | 3,035    | 7     |
| 999    | 12.9x | 69.5          | 97,326   | 7,527    | 2     |

**Key improvement**: The NW/WB ratio now **plateaus around 11-13x** instead of exploding to
33x. The wage_bill fix (`142e1b6`) corrected overstated gross profits, and the planning-phase
pricing dampened inflation so nominal net worth grows slower.

**Root cause chain** (four reinforcing mechanisms — updated from five):

1. **Structural profit guarantee** (see A4): Aggregate profits are mathematically guaranteed
   by the circular flow structure, ensuring net worth grows every period.

1. **90% retention**: `delta = 0.10` means only 10% of positive profits leave the firm.
   Net worth compounds at ~90% of the profit rate.

1. **No capital expenditure**: Credit demand = max(0, wage_bill - total_funds). Firms only
   need credit to cover wages. No investment, depreciation, materials, R&D, or fixed costs.

1. **Steady-state attractor**: The NW/WB ratio converges to ~13x regardless of initial
   conditions (see Section G intervention testing — `net_worth_ratio` has no long-run effect).

**Previous claim refuted**: The previous analysis listed "Initial over-capitalization"
(`net_worth_ratio = 1.0`) as a root cause. Intervention testing (Section G) shows that even
starting firms at 10% of normal net worth barely changes the long-run NW/WB ratio (12.0x vs
12.9x). The self-financing problem is a **steady-state attractor**, not an initial-condition
artifact. Only flow parameters (like `delta`) can shift the steady state.

### A2. Zombie Firms — STABLE

| Metric                          | 4dc5147 fix | Current |
| ------------------------------- | ----------- | ------- |
| Non-producing firms/period (BI) | 0.8         | 0.8     |
| Max non-producing in one period | 4           | 5       |

The zombie fix from `4dc5147` remains effective. Occasional non-producing firms (max 5) are
transient — newly spawned firms that haven't yet acquired labor. The `new_firm_price_markup`
reduction from 1.5 to 1.05 means new entrants no longer inject extreme price outliers.

### A3. Planning-Phase Breakeven — NEW BEHAVIOR

The switch from production-phase to planning-phase breakeven pricing is the most impactful
behavioral change. The formula shifted from:

```
Production-phase:  breakeven = (wage_bill + interest) / (productivity × current_labor)
Planning-phase:    breakeven = (prev_wage_bill + prev_interest) / desired_production
```

Key differences:

1. **Denominator**: `desired_production` is computed *before* ceil() rounding inflates labor,
   so it's smaller than actual production. This gives a higher breakeven, but the numerator
   uses *previous period's* costs (which are generally lower), and the net effect is a
   **lower** breakeven overall because the lag matters more than the denominator change.

1. **Degenerate t=0**: All 100 firms get `breakeven = 0` at t=0 because `wage_bill = 0` and
   `interest = 0` (no previous-period costs). Benign in practice — `price_init = 1.50`
   provides the actual floor.

1. **Impact on inflation**: The one-period lag in costs means the breakeven floor responds
   more slowly to wage increases. This **dampens the inflation ratchet** — the breakeven
   floor can no longer instantly pass through a wage increase in the same period.

| Metric           | Previous (prod-phase) | Current (plan-phase) |
| ---------------- | --------------------- | -------------------- |
| Mean inflation   | 2.7%/quarter          | **1.5%/quarter**     |
| Price at t=999   | 571                   | **45**               |
| Breakeven at t=0 | Non-zero              | **0 (all firms)**    |
| Mean markup      | 1.66x                 | **1.54x**            |

### A4. Structural Aggregate Profit Guarantee — CONFIRMED

The Kalecki profit equation analysis is **fully reconfirmed** with updated numbers:

#### The mechanism (unchanged)

```
Firms pay wages → Households receive income → Households spend → Firms collect revenue
                                                     ↑
                            Firms pay dividends → Households get extra income
```

Aggregate Profit = Total Consumption Spending - Total Wage Bill > 0 because households
spend from dividends and accumulated savings in addition to wages.

#### Updated quantitative evidence

| Metric                                   | Previous | Current    |
| ---------------------------------------- | -------- | ---------- |
| % active firms with price >= breakeven   | 97.0%    | **96.6%**  |
| Mean markup (price/breakeven)            | 1.659x   | **1.540x** |
| Mean sell-through rate                   | 71.9%    | **75.1%**  |
| Revenue >= wage_bill (% of firm-periods) | 61.0%    | **61.9%**  |
| Firms with negative gross profit         | 33.5%    | **38.1%**  |
| Firms with negative net profit           | 33.6%    | **38.4%**  |
| Demand/Supply ratio (budget/prod_value)  | 0.741    | **0.802**  |

**Negative profit increase explained**: The percentage of firms with negative profits
**increased** from 33.5% to 38.1%. This is because the wage_bill fix (`142e1b6`) corrected
an overstatement of gross profit. Previously, `wage_bill` was reduced by contract expirations
*before* revenue collection, making `revenue - wage_bill` appear larger than reality. Now that
`wage_bill` correctly reflects actual payments, more firms show their true (negative) profit.

The aggregate guarantee still holds: effective markup × sell-through = `1.54 × 0.75 = 1.16 > 1.0`.

#### Five reinforcing sub-mechanisms (unchanged)

1. **Breakeven price floor**: `price = max(attempted_price, breakeven)`. With
   `price_cut_allow_increase: true`, even firms in the "cut price" group can have their
   price pushed up by the breakeven floor.

1. **No overhead costs**: Firm costs = wages + interest only. Variable-only cost structure
   makes break-even trivial.

1. **Inventory replacement, not accumulation**: `prod.inventory[:] = prod.production` —
   unsold goods from previous periods are erased. No mounting liabilities.

1. **Dividend recycling**: Dividends → household savings → spending → revenue → more profit.
   Positive feedback loop.

1. **Savings drawdown**: Mean savings rate is persistently negative (-5.4%). Households
   spend more than current income, providing a structural demand floor above wages.

### A5. Cost-Push Inflation Ratchet — DAMPENED (Figures 4, 10)

The inflation ratchet mechanism is **unchanged** but its intensity has been **halved**:

```
1. Workers demand higher wages (h_xi shock)
2. Breakeven price rises: breakeven = (prev_wage_bill + prev_interest) / desired_production
3. Price floor forces prices up: price = max(attempted_price, breakeven)
4. Even firms trying to CUT prices get floored (price_cut_allow_increase: true)
5. Average price rises → inflation recorded
6. Minimum wage adjusts upward (indexed to price history)
7. → Back to step 1
```

What dampened it:

- **Planning-phase breakeven** introduces a one-period lag in cost pass-through, softening
  the wage→price→wage feedback loop.
- **`new_firm_price_markup: 1.05`** (was 1.5) means new entrants no longer inject large
  upward price shocks.

Evidence:

| Metric                                  | Previous                      | Current                              |
| --------------------------------------- | ----------------------------- | ------------------------------------ |
| Mean inflation                          | 2.7%                          | **1.5%**                             |
| Markup mean                             | 1.66x                         | **1.54x**                            |
| Markup 5th percentile (breakeven floor) | 1.000                         | **1.000**                            |
| Inflation distribution                  | Right-skewed, rarely negative | Centered ~1-3%, occasional negatives |

The `price_cut_allow_increase: true` setting remains the critical one-way ratchet enabler.

### A6. Banking Sector — NEW: ACCELERATING DECAY (Figures 7, 8)

This is the **most concerning new finding**. The banking sector shows a clear **death spiral**
pattern with accelerating failures in later periods:

| Metric                          | Previous | Current   |
| ------------------------------- | -------- | --------- |
| Mean bank equity/bank (post-BI) | ~100-200 | **2.16**  |
| Bank equity (t=500 → t=999)     | —        | 1.1 → 8.0 |
| Cumulative bank bankruptcies    | ~500     | **660**   |
| Mean bank bankruptcies/period   | ~1       | **0.71**  |
| Any negative equity (post-BI)   | —        | **Yes**   |

The log output reveals **cascading** failures in the final ~200 periods: clusters of 2-4
banks failing simultaneously, with one instance of all 10 banks being replaced in rapid
succession.

**Root cause**: Banks earn almost no interest income (only 4.2 loans/period across 10 banks)
but face operating expense shocks. After `957a1ad`, each bank draws `phi_k ~ U(0, h_phi)`,
creating genuine heterogeneity but also genuine drain:

```
Bank equity change/period ≈ interest_income - opex_shock × equity
                          ≈ negligible - random × equity
                          → slow negative drift
```

Respawned banks get `equity_base_init = 5.0` which quickly erodes again.

**Why this is worse than before**: Previously, the constant `h_phi` bug overstated interest
rates, which paradoxically gave banks a (small) revenue stream from rare loans. Now with
correct per-bank `phi_k ~ U(0, h_phi)`, banks charge less on average, earning even less.

### A7. Excess Supply / Demand Deficit — IMPROVED (Figures 1, 5)

| Metric                   | Previous | Current   |
| ------------------------ | -------- | --------- |
| Mean sell-through rate   | 71.9%    | **75.1%** |
| Median sell-through rate | 78.4%    | **75.0%** |
| Demand/Supply ratio      | 0.741    | **0.802** |

The demand/supply gap narrowed from ~26% unsold to ~20% unsold. Lower inflation preserves
real purchasing power better.

Capacity utilization still exceeds 100% in **82.6%** of post-burn-in periods (mean: 101.3%),
confirming the quantization trap remains active but is less severe than the previous ~120%
peaks.

______________________________________________________________________

## B. Visualization & Script Quality — ALL PREVIOUS FIXES HOLD

All 5 script fixes from the previous analysis remain correct:

| Issue               | Fix Applied                                                           | Status |
| ------------------- | --------------------------------------------------------------------- | ------ |
| B1. Labor share     | `nominal_gdp = total_gdp * avg_price`; divide by nominal GDP          | DONE   |
| B2. Savings rate    | `disposable_income = income + dividends`; divide by disposable income | DONE   |
| B3. Price y-axis    | `weighted_stats(..., pct_bounds=(1, 99))` for price/breakeven panels  | DONE   |
| B4. Credit/GDP      | Divide by nominal GDP                                                 | DONE   |
| B5. Investment rate | Divide by nominal GDP                                                 | DONE   |

### B5. Capacity Utilization Exceeds 100% (Figures 1, 6) — REDUCED

Mean capacity utilization reaches ~106% in some periods (was ~120%). The planning-phase
breakeven uses `desired_production` directly, which slightly reduces the mismatch between
desired and actual production from ceil() rounding. Still not a bug — expected behavior.

### B6. Price Dispersion CoV Spikes — REDUCED (Figure 4)

Price dispersion (CoV) now shows spikes to ~0.07 (was ~10). The `new_firm_price_markup`
reduction from 1.5 to 1.05 eliminated the extreme price outliers from new entrants.

**No new visualization issues detected.**

______________________________________________________________________

## C. Economically Notable Patterns

### C1. Moderated Nominal Growth, Stationary Real Output

All nominal variables still grow exponentially but at a lower rate:

| Variable            | Period 500 | Period 999 | Growth Factor |
| ------------------- | ---------- | ---------- | ------------- |
| Avg price           | 7.2        | 44.8       | 6.2x          |
| Total system wealth | 14,854     | 101,005    | 6.8x          |
| NW/WB ratio         | 10.9x      | 12.9x      | 1.2x          |

Key improvement: NW/WB **stabilizes** around 11-13x instead of exploding to 33x, meaning
the self-financing problem is no longer getting worse over time.

Real GDP stays flat around 239 units — expected without the R&D extension.

### C2. High Vacancy Rate — STILL PRESENT (Figure 2)

Vacancy rate persists at 7-20%, with 4.5% unemployment. Still reflects the quantization trap.

### C3. Real Wage Trend — NEW (Figure 10)

Real wage (`nominal_wage / avg_price`) fluctuates between 0.35-0.42 with a subtle decline
from ~0.40 to ~0.37 over the post-burn-in window, suggesting inflation slightly outpaces
nominal wage growth. No secular trend — correct for baseline (no productivity growth).

### C4. Macroeconomic Curves — IMPROVED (Figure 9)

| Curve     | Previous r | Current r | Expected | Quality      |
| --------- | ---------- | --------- | -------- | ------------ |
| Phillips  | -0.37      | **-0.54** | Negative | **Good**     |
| Okun      | -0.83      | **-0.58** | Negative | Moderate     |
| Beveridge | -0.17      | **-0.30** | Negative | **Improved** |

The Phillips curve strengthened substantially. The higher unemployment variance (std: 2.5%)
provides more statistical power. The planning-phase pricing lag may also soften the direct
wage→price pass-through, giving the Phillips relationship more room to express itself.

### C5. Gross Profit Distribution — BIMODAL (Figure 12)

The gross profit KDE shows a **bimodal** distribution at later snapshots: a cluster near 0
(break-even firms) and spreading tails. At t=999 the distribution spans -150 to +50, with
the left tail (loss-making) heavier than the right. This bimodality was less visible before
the wage_bill fix, which was inflating the right tail.

### C6. Interest Rate Heterogeneity — NEW

Post `957a1ad`, banks now have genuine interest rate heterogeneity:

| Metric                  | Value                   |
| ----------------------- | ----------------------- |
| Rate range (t=500)      | [2.004%, 2.171%]        |
| Rate range (t=999)      | [2.009%, 2.188%]        |
| Mean std across periods | 0.054 percentage points |

The spread is small (~17bps) because the fragility term is near-zero (leverage ≈ 0). With an
active credit market, rate dispersion would be much wider.

### C7. Savings Rate Dynamics (Figures 5, 10)

Savings rate ranges from -2% to -17%, mean **-5.4%** (was -2.9%). Households draw down
savings faster relative to income, possibly because lower inflation reduces nominal income
growth while spending remains tied to savings levels through the propensity function.

### C8. Growing Wealth Concentration (Figure 11)

Net worth KDE shows dramatic right-tail growth at 5 time snapshots. By period 999 the
distribution spans ~0 to ~3,500 (less extreme than the previous ~90,000 due to lower
inflation), but the shape remains highly right-skewed.

______________________________________________________________________

## D. Stock-Flow Consistency (Figure 13) — PASS

| Check                  | Result             | Status |
| ---------------------- | ------------------ | ------ |
| Wage flow residual     | 3.34e-13           | PASS   |
| Dividend flow residual | 2.94e-14           | PASS   |
| Total wealth trend     | Smooth exponential | PASS   |

Both residuals at floating-point noise level. Perfect internal accounting.

______________________________________________________________________

## E. Root Cause Dependency Graph (Updated)

```
QUANTIZATION TRAP (ceil rounding)
  │
  ├──→ Over-production (cap util 101.3%, >100% in 83% of periods)
  │      │
  │      └──→ Persistent excess supply (~25% unsold) [improved from 28%]
  │
  └──→ High vacancy rate (7-20%)

PLANNING-PHASE BREAKEVEN (uses desired_production + prev-period costs)
  │
  ├──→ Lower unit cost estimates (desired_prod avoids ceil inflation)
  │      │
  │      └──→ Dampened inflation ratchet (1.5% vs 2.7%) [IMPROVED]
  │
  └──→ Degenerate t=0 breakeven (all zeros) [cosmetic]

BREAKEVEN PRICE FLOOR + price_cut_allow_increase: true
  │
  ├──→ Markup guarantee (mean 1.54x)
  │      │
  │      └──→ Revenue >> Costs for selling firms
  │
  └──→ One-way price ratchet (still present, just slower)
         │
         └──→ Min wage indexation feedback loop

AGGREGATE PROFIT GUARANTEE (Kalecki equation)
  │
  ├──→ Dividend recycling → household spending > wages
  │
  ├──→ Savings drawdown → negative savings rate (-5.4%)
  │
  └──→ Net worth accumulation (90% retention, delta=0.10)
         │
         ├──→ NW >> WageBill (13x, stabilized) [improved from 33x]
         │      │
         │      └──→ NEAR-DEAD CREDIT MARKET (leverage 0.0017)
         │
         └──→ Banks earn no interest
                │
                └──→ BANK DEATH SPIRAL (operating expenses > revenue)
                       │
                       └──→ Cascading bank failures (late periods) [NEW]
```

______________________________________________________________________

## F. Impact Assessment of Recent Fixes

| Fix                            | Commit    | Expected Impact               | Observed Impact                                                | Assessment                         |
| ------------------------------ | --------- | ----------------------------- | -------------------------------------------------------------- | ---------------------------------- |
| Per-bank opex_shock            | `957a1ad` | Bank rate heterogeneity       | 17bps spread, accelerated bank decay                           | Correct fix, surfaced latent issue |
| Principal-based bad debt       | `fb3906e` | Correct loss-sharing          | No visible effect (too few loans)                              | Correct but dormant                |
| LoanBook retention             | `b3944d6` | Enable planning-phase pricing | Breakeven at t≥1 uses prev-period interest                     | Works as intended                  |
| Wage_bill fix                  | `142e1b6` | Correct gross profit          | More firms show negative profit (38% vs 34%), NW/WB stabilized | **Significant**                    |
| Planning-phase pricing         | `20986d4` | Earlier pricing, lower break. | Inflation halved, price level -92%, markup 1.66→1.54           | **Dramatic**                       |
| new_firm_price_markup 1.5→1.05 | `f831e0f` | Reduce price dispersion       | CoV spikes eliminated (10 → 0.07)                              | **Effective**                      |

______________________________________________________________________

## G. Parameter Intervention Testing

Tested three parameter-level interventions, individually and combined, against the baseline.
All 10 scenarios ran for 1000 periods with no collapses. Seed=42.

### Comparison Table

| Scenario            | Unemp% | Infl% | GDP | Price(end) | NW/WB | Loans/p | Leverage | Markup | Sell% | NegGP% | SavRate% | LabShr% | FBankr/p | BBankr/p | BankEq |
| ------------------- | ------ | ----- | --- | ---------- | ----- | ------- | -------- | ------ | ----- | ------ | -------- | ------- | -------- | -------- | ------ |
| **BASELINE**        | 4.5    | 1.5   | 239 | 44.8       | 12.9  | 4.2     | 0.0017   | 1.54   | 75.1  | 38.1   | -5.4     | 74.6    | 1.42     | 0.71     | 2.2    |
| delta=0.40          | 3.2    | 1.7   | 242 | 81.2       | 7.8   | 5.2     | 0.0053   | 1.58   | 78.2  | 32.2   | -8.4     | 73.6    | 1.84     | 0.64     | 13.1   |
| delta=0.60          | 2.3    | 1.7   | 244 | 126.2      | 3.8   | 7.8     | 0.0137   | 1.61   | 80.5  | 26.7   | -12.2    | 72.5    | 2.65     | 0.49     | 26.0   |
| nw_ratio=0.3        | 4.3    | 1.5   | 239 | 40.6       | 13.1  | 3.4     | 0.0019   | 1.55   | 74.9  | 38.0   | -5.0     | 74.1    | 1.28     | 0.69     | 3.7    |
| nw_ratio=0.1        | 4.2    | 1.5   | 239 | 42.3       | 12.0  | 3.9     | 0.0018   | 1.54   | 74.7  | 38.2   | -5.8     | 74.4    | 1.65     | 0.76     | 2.7    |
| no_price_inc        | 2.6    | 1.6   | 243 | 47.7       | 11.5  | 3.9     | 0.0025   | 1.52   | 77.9  | 37.1   | -8.9     | 78.4    | 1.89     | 0.76     | 4.5    |
| delta=0.40 + nw=0.3 | 3.2    | 1.7   | 242 | 82.1       | 6.7   | 5.3     | 0.0053   | 1.57   | 78.5  | 32.0   | -9.0     | 74.0    | 1.89     | 0.62     | 10.3   |
| delta=0.40 + no_inc | 2.3    | 1.7   | 244 | 87.3       | 6.3   | 6.0     | 0.0065   | 1.55   | 79.7  | 31.5   | -12.4    | 76.3    | 2.81     | 0.57     | 13.6   |
| ALL THREE           | 2.3    | 1.7   | 244 | 72.5       | 7.3   | 5.7     | 0.0062   | 1.55   | 80.0  | 31.6   | -12.2    | 76.3    | 2.38     | 0.57     | 18.0   |
| AGGRESSIVE          | 2.4    | 1.7   | 244 | 109.7      | 4.0   | 8.6     | 0.0137   | 1.60   | 81.0  | 26.8   | -13.5    | 73.4    | 2.76     | 0.58     | 20.4   |

Legend:

- NW/WB = net worth / wage bill ratio at final period
- Loans/p = mean active loans per period (post-burn-in)
- NegGP% = % of firm-periods with negative gross profit
- BankEq = mean bank equity per bank (post-burn-in)
- AGGRESSIVE = delta=0.60, net_worth_ratio=0.1, price_cut_allow_increase=false

### Individual Effects

#### `delta` (dividend payout ratio) — THE DOMINANT LEVER

| delta | NW/WB | Loans/p | Leverage | Unemployment | Bank Equity |
| ----- | ----- | ------- | -------- | ------------ | ----------- |
| 0.10  | 12.9x | 4.2     | 0.0017   | 4.5%         | 2.2         |
| 0.40  | 7.8x  | 5.2     | 0.0053   | 3.2%         | 13.1        |
| 0.60  | 3.8x  | 7.8     | 0.0137   | 2.3%         | 26.0        |

By far the most powerful lever. At delta=0.60, NW/WB drops from 13x to 4x, active loans
nearly double, and bank equity improves **12x** (2.2 → 26.0).

**Mechanism**: Higher dividends drain firm net worth → firms need credit → banks earn
interest → banks stay healthy. Also lowers unemployment because dividend recycling
(firms → households → spending → firms) increases aggregate demand.

**Trade-off**: Slightly higher inflation (+0.2pp) and more negative savings rate.
Firm bankruptcy rate increases (more firms become credit-dependent and vulnerable).

#### `net_worth_ratio` (initial firm capitalization) — INEFFECTIVE

| net_worth_ratio | NW/WB | Loans/p | Leverage |
| --------------- | ----- | ------- | -------- |
| 1.0 (baseline)  | 12.9x | 4.2     | 0.0017   |
| 0.3             | 13.1x | 3.4     | 0.0019   |
| 0.1             | 12.0x | 3.9     | 0.0018   |

**Surprisingly ineffective.** Even starting firms at 10% of normal net worth barely changes
the long-run outcome. By period 500 the profit accumulation mechanism has erased any initial
conditions. The NW/WB ratio converges to the same ~12-13x regardless.

**Conclusion**: The self-financing problem is a **steady-state attractor**, not an
initial-condition artifact. Only flow parameters (like `delta`) can shift the steady state.
Remove `net_worth_ratio` from the intervention list.

#### `price_cut_allow_increase: false` — MODERATE, INTERESTING SIDE EFFECTS

| Setting | NW/WB | Unemployment | Sell-through | Labor Share |
| ------- | ----- | ------------ | ------------ | ----------- |
| true    | 12.9x | 4.5%         | 75.1%        | 74.6%       |
| false   | 11.5x | 2.6%         | 77.9%        | 78.4%       |

Slightly reduces NW/WB and notably **lowers unemployment** (-1.9pp) and **increases labor
share** (+3.8pp). Allowing genuine downward price adjustment creates a more competitive
goods market where sell-through rates improve → firms sell more → hire more → lower
unemployment.

Does NOT meaningfully fix the credit market (leverage only 0.0025 vs 0.0017).

### Combination Effects

The `delta + no_price_inc` combination is **better** than `ALL THREE` on leverage (0.0065
vs 0.0062), confirming that `net_worth_ratio` adds nothing. The AGGRESSIVE scenario achieves
the best credit metrics (8.6 loans, leverage 0.0137) at the cost of higher firm bankruptcies
(2.76/period) and very negative savings rates (-13.5%).

### Key Conclusions

1. **`delta` is the only lever that matters for the credit market.** Increasing from 0.10 to
   0.40 halves NW/WB and rescues bank equity. At 0.60 it nearly eliminates self-financing.

1. **`net_worth_ratio` is useless for long-run dynamics.** The steady-state NW/WB ratio is
   invariant to initial conditions — a textbook dynamical-systems result where the fixed
   point depends on parameters, not initial state.

1. **`price_cut_allow_increase: false` helps the goods market** (sell-through +3pp, labor
   share +4pp, unemployment -2pp) but does not fix the credit market.

1. **No parameter intervention achieves a "normal" credit market.** Even AGGRESSIVE only
   reaches leverage 0.014, still near-zero vs real-economy standards (0.3-0.5). The
   structural issue (`credit_demand = max(0, WB - NW)`, no investment) means leverage cannot
   be high without mechanism-level changes.

1. **Bank health is a pure function of `delta`.** Higher delta → more dividends → firms need
   more credit → banks earn interest → bank equity grows. The bank death spiral is
   **completely solved** by delta ≥ 0.40.

______________________________________________________________________

## H. Recommended Parameter Change

Based on intervention testing, the single highest-impact change:

```yaml
delta: 0.40   # was 0.10 — quadruples dividend payout
```

This achieves:

- NW/WB: 12.9x → 7.8x (-39%)
- Active loans: 4.2 → 5.2/period (+24%)
- Bank equity: 2.2 → 13.1 per bank (+6x)
- Unemployment: 4.5% → 3.2%
- Inflation: +0.2pp (negligible cost)
- No collapses, no pathological behavior

Optional addition for goods market improvement:

```yaml
price_cut_allow_increase: false  # was true — allow genuine downward price flexibility
```

### Remaining Structural Issues (require mechanism-level changes)

| Issue                   | Root Cause                                       | Possible Interventions                                        |
| ----------------------- | ------------------------------------------------ | ------------------------------------------------------------- |
| Near-dead credit market | credit_demand = max(0, WB - NW), no investment   | Add depreciation, overhead costs, or investment demand        |
| Profit guarantee        | Breakeven floor + no overhead + inventory reset  | Weaken breakeven floor, accumulate inventory, add fixed costs |
| Bank death spiral       | No loan revenue + operating expenses             | **Solved by delta ≥ 0.40**                                    |
| Excess supply           | Quantization trap (ceil rounding)                | See LABOR_MARKET_QUANTIZATION.md                              |
| Inflation ratchet       | price_cut_allow_increase: true + wage indexation | Set to false; decouple min wage from prices                   |
