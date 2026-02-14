# Reference: Section 3.10.1 — Exploration of the Parameter Space

This document summarizes the expected findings from Section 3.10.1 of
*Macroeconomics from the Bottom-Up* (Delli Gatti et al., 2011), organized by
analysis component. It serves as the ground truth for cross-checking our
simulation results.

Quantitative benchmarks (approximate values read from the book's figures)
are in the companion file `reference_values.yaml`.

Reference figures are in `notes/BAM/figures/robustness/` and
`notes/BAM/figures/dsge-comparison/`.

> **Note on economy size**: The book's default parameters (Table 3.1) are
> **F=100 firms, W=500 workers, B=10 banks** — the same as our defaults.
> Results should therefore be directly comparable with Figure 3.9.

______________________________________________________________________

## 1. Internal Validity (Part 1)

### 1.1 Overall Stability

> The aggregate behaviour emerging from an averaging of outcomes over 20
> alternative random-seed simulations show that the results we have discussed so
> far are significantly robust. The key qualitative time-series features of growth and
> cyclical fluctuations remain unaffected, and the cross-simulation variance
> calculated for typical macroeconomic variables (GDP, productivity, inflation, real
> wage, unemployment, interest rates, bankruptcy rates) is remarkably small.

**What to check:** Cross-simulation CV (coefficient of variation) for key
statistics should be small. The book doesn't give exact thresholds, but
"remarkably small" suggests CV < 0.1–0.2 for level statistics and higher
tolerance for volatility statistics.

### 1.2 Co-Movement Structure (Figure 3.9)

> Figure 3.9 reports the structure of co-movements at four leads and lags, plus
> the contemporaneous one, between the de-trended values of the GDP and of the
> other five variables already considered in Figure 3.7. It largely corroborates our
> previous findings regarding the procyclicality of unemployment, productivity
> and the real wage, as well as the substantial a-ciclicality of the aggregate price
> index and of the real interest rate.

**Reference figure:** `notes/BAM/figures/robustness/3_9.png` (combined),
plus individual panels `3_9a_unemployment.png` through `3_9e_real-wage.png`.

**Expected classifications** (from the figure, using ±0.2 acyclicality band):

| Variable          | Classification                 | Contemporaneous (lag=0)      | Shape                |
| ----------------- | ------------------------------ | ---------------------------- | -------------------- |
| (a) Unemployment  | Countercyclical (all negative) | ≈ −0.60                      | V-shaped trough      |
| (b) Productivity  | Procyclical (all positive)     | ≈ +0.62 (avg), +0.88 (basic) | Inverted-V peak      |
| (c) Price index   | Acyclical/weakly negative      | ≈ −0.08 (avg)                | Rising left-to-right |
| (d) Interest rate | Acyclical (all near zero)      | ≈ −0.02 to 0.00              | Flat, noisy          |
| (e) Real wage     | Procyclical (all positive)     | ≈ +0.65 (avg), +0.85 (basic) | Inverted-V peak      |

> **Textual discrepancy:** The book text says "the procyclicality of unemployment"
> but Figure 3.9a clearly shows all-negative correlations (countercyclical).
> The text likely refers to unemployment's strong *cyclicality* pattern, or the
> authors may have been thinking of employment (Figure 3.7a), which *is*
> procyclical. In any case, follow the figure as ground truth.

**Key qualitative checks:**

- Baseline (`+`) and average (`•`) should have the **same sign** at each lag
- Average is typically **attenuated** (closer to zero) vs. baseline
- Average is **smoother** (less variation across lags)

### 1.3 AR Structure and Impulse Response

> A final remark is in order to highlight the simulation outcome that proves
> to be most challenging, namely the auto-regressive structure of the de-trended
> output and its relative hump-shaped impulse-response pattern. At odds with the
> result shown in Panel (f) of Figure 3.7, when we consider an average over cross-
> section simulations, the movement in the log of detrended GPD can be best
> approximated by an AR(1) structure (with an autoregressive parameter around
> 0.8).

**Reference figure (single-seed IRF):** `notes/BAM/figures/dsge-comparison/3_7f_gdp-cyclical-component-impulse-response-function.png`

**Expected AR structure:**

| Scope             | AR order | phi_1    | IRF shape                                             |
| ----------------- | -------- | -------- | ----------------------------------------------------- |
| Individual seed   | AR(2)    | variable | Hump-shaped (peak at period 1–2, decay to 0 by ~8–10) |
| Cross-sim average | AR(1)    | ≈ 0.8    | Monotone exponential decay (no hump)                  |

**Why the difference:** Averaging individual AR(2) processes with different
coefficients (but similar phi_1) produces an aggregate that looks AR(1)-like.
The hump in the IRF comes from the AR(2) term (phi_2), which cancels out
across seeds while phi_1 averages stably.

### 1.4 Firm Size Distribution

> The distribution of the firms' size (both in terms of sales and net worth)
> calculated in correspondence of the last simulation period is definitely invariant
> in its significant departure from normality and its strong positive skewness.

**What to check:**

- Shapiro-Wilk normality test should **reject** (p < 0.05) across most/all seeds
- Skewness should be **positive** (right-skewed) across all seeds
- Both sales (production) and net worth distributions should show this

### 1.5 Empirical Curves

> Finally, a Phillips curve, an Okun law and a Beveridge curve continue to emerge
> from each simulation and on average.

**Expected signs:**

| Curve           | Variables                          | Expected correlation |
| --------------- | ---------------------------------- | -------------------- |
| Phillips curve  | Unemployment vs. wage inflation    | Negative             |
| Okun's law      | Unemployment growth vs. GDP growth | Negative             |
| Beveridge curve | Unemployment vs. vacancies         | Negative             |

______________________________________________________________________

## 2. Sensitivity Analysis (Part 2)

### 2.1 Overview

> We choose to perform a univariate sensitivity analysis, according to which the
> model outcomes are analyzed with respect to the variation of one parameter at a
> time, whereas all the other parameters of the system remain constant. [...]
> The parameters that prove to be crucial − in that alternative
> parameter values change simulation results significantly − are the ones related to
> the duration of labour contracts, to the number of opportunities any unit is
> allowed to locally explore as it searches for market transactions (*local* markets),
> and to the total size of the economy.

### 2.2 (i) Local Credit Markets — H (max_H)

> As we increase the number of banks each firm can borrow from − in particular,
> as we raise the parameter H from its baseline value from 2 to 3, 4, and 6 − the
> general properties of the model (in terms of output, productivity, unemployment,
> inflation, real wages, bankruptcy rates, and so on) do not manifest any
> significant variation. It must be noted, however, that an increase in H forces the
> cyclical component of the price index to be coincident with the aggregate output,
> while the right tail of the size distribution of firms' net worth becomes more and
> more similar to a Pareto distribution. As the number of potential partners on the
> credit market is reduced to 1, on the contrary, the size distribution looks more
> similar to an exponential.

**Experiment:** `credit_market`, param `max_H`, values [1, 2, 3, 4, 6], baseline=2

**Expected behavior:**

- General macro properties: **stable** across all H values
- Price index co-movement: becomes **coincident** (less leading) as H increases
- Net worth distribution: becomes more **Pareto-like** as H increases, more **exponential** at H=1
- No significant change in unemployment, inflation, GDP growth

### 2.3 (ii) Local Consumption Goods Markets — Z (max_Z)

> As we increase Z from 2 to 3, 4, 5 and 6, competition among firms increases, and
> the function exerted on firms' growth by the preferential attachment mechanism
> becomes less and less effective. In particular, the real wages become lagging,
> their co-movement with output similar to those of the price index and, as it is
> logical, the kurtosis of the firm's size distribution decreases dramatically.
> Moreover, production displays smoother patterns, without sudden booms or
> crashes.

**Experiment:** `goods_market`, param `max_Z`, values [2, 3, 4, 5, 6], baseline=2

**Expected behavior:**

- Real wage: becomes **lagging** (peak shifts to positive lags)
- Firm size kurtosis: **decreases** with higher Z
- Output: becomes **smoother** (lower volatility)
- Big firms less likely to emerge → systemic risk more evenly spread

### 2.4 (iii) Local Labour Markets — M (max_M)

> As far as the former is concerned, we start our sensitivity experiment by
> decreasing the number of allowable applications from 4 to 3 and 2, discovering
> that prices switch from being anti-cyclical and leading to pro-cyclical and
> lagging. Aggregate output shows an higher degree of instability [...]
> This interpretation is indirectly confirmed as we *increase* the number of
> applications (to 5 and 6): tougher competition on the labour market and a higher
> probability to find workers make firms all alike, and their size distribution scales
> much more as an exponential, or even a uniform. In addition, as one can expect,
> competition between firms in hiring workers tends to push the real wage up,
> sometime even above average productivity.

**Experiment:** `labor_applications`, param `max_M`, values [2, 3, 4, 5, 6], baseline=4

**Expected behavior when M decreases (2, 3):**

- Prices: switch from anti-cyclical/leading → **pro-cyclical/lagging**
- Output: **higher instability**
- Firm size distribution: more **Pareto-like** (path-dependency creates "advantaged" firms)

**Expected behavior when M increases (5, 6):**

- Firm size distribution: more **exponential/uniform** (firms become alike)
- Real wages: pushed **up** (sometimes above average productivity)

### 2.5 (iv) Employment Contracts Duration — theta

> In order to control for both a very flexible and a quite rigid labour market, we
> have first decreased it to 6, 4 and 1, to subsequently increase it to 10, 12 and 14.
> [...] While for intermediate values of the parameter the main statistical properties
> of the model do not change significantly, the opposite is true for the extreme
> values, which produce degenerate dynamics.

**Experiment:** `contract_length`, param `theta`, values [1, 4, 6, 8, 10, 12, 14], baseline=8

**Expected behavior — short contracts (theta < 8):**

- Co-movements: become **less pronounced** (except unemployment and wages)
- Output: becomes **smoother**, loses AR(2) structure
- Firm size: distributes more **uniformly**
- Unemployment: **increases** despite flexible labor market (coordination failures)
- theta=1: **economy collapses** in most simulations (fatal market failures)

> When labour contracts last only one period, that is when firms are given full
> freedom of firing, the number of bankruptcies and the unemployment rate reach
> very high values, and in most of the simulations the whole economy collapses,
> signalling the presence of fatal market failures.

**Expected behavior — long contracts (theta > 8):**

- Co-movements: **contrast sharply** with real data
- Dynamics: often **degenerate**
- theta=12+: firms can't fire when fragile → more bankruptcies → macroeconomic breakdown

> A different reasoning applies when the labour market is rigid [...] because of
> long contractual commitments, firms cannot resort to firing when they are
> financially fragile and go bankruptcy more easily. This leads to an overall
> macroeconomic breakdown.

### 2.6 (v) Size and Structure of the Economy

> As the size of the economy is scaled up, the average growth rate and the
> statistical properties expressed in terms of co-movements are very similar to
> their counterparts calculated for the baseline simulation, whereas the time series
> of macroeconomic variables display rather smoother cyclical fluctuations.

**Experiment:** `economy_size`, multi-param, values listed in `experiments.py`

**Expected behavior — proportional scaling (2x, 5x, 10x):**

- Co-movements: **similar** to baseline
- Growth rate: **similar** to baseline
- Volatility: **decreases** with economy size (macroeconomic averaging effect)

**Expected behavior — composition changes:**

> Doubling the number of banks does not exert any significant variation to the
> model's outcomes. When the number of households is increased, in turn, the
> leads-and-lags co-movement analysis shows a scenario quite similar to that of
> the baseline simulation, but time series appear to grow much faster – and with a
> higher volatility - thanks to the enlarged availability of workforce. Conversely,
> an increase of the proportion of firms has the effect of slowing down the average
> rate of growth of the economy.

| Change        | Effect on growth                     | Effect on co-movements | Notes                                                      |
| ------------- | ------------------------------------ | ---------------------- | ---------------------------------------------------------- |
| 2x banks      | **No change**                        | No change              | Banks are not a bottleneck                                 |
| 2x households | **Faster growth**, higher volatility | Similar to baseline    | More workforce available                                   |
| 2x firms      | **Slower growth**                    | Modified               | More competition → more rationing → less profit → less R&D |

______________________________________________________________________

## Figure Reference

| Figure | Description                                 | Path                                                                                          |
| ------ | ------------------------------------------- | --------------------------------------------------------------------------------------------- |
| 3.9    | Co-movements (combined, 5 panels)           | `notes/BAM/figures/robustness/3_9.png`                                                        |
| 3.9a   | Unemployment co-movements                   | `notes/BAM/figures/robustness/3_9a_unemployment.png`                                          |
| 3.9b   | Productivity co-movements                   | `notes/BAM/figures/robustness/3_9b_productivity.png`                                          |
| 3.9c   | Price index co-movements                    | `notes/BAM/figures/robustness/3_9c_price-index.png`                                           |
| 3.9d   | Real interest rate co-movements             | `notes/BAM/figures/robustness/3_9d_real-interest-rate.png`                                    |
| 3.9e   | Real wage co-movements                      | `notes/BAM/figures/robustness/3_9e_real-wage.png`                                             |
| 3.10   | Log-output time series                      | `notes/BAM/figures/robustness/3_10.png`                                                       |
| 3.7f   | GDP cyclical component IRF (AR(2) baseline) | `notes/BAM/figures/dsge-comparison/3_7f_gdp-cyclical-component-impulse-response-function.png` |

______________________________________________________________________

## Source

All quotes are from Chapter 3, Section 3.10.1 of:

Delli Gatti, D., Desiderio, S., Gaffeo, E., Cirillo, P., & Gallegati, M. (2011).
*Macroeconomics from the Bottom-up*. Springer.

Full text of Section 3.10.1 is in `notes/BAM/BOOK_SECTION_3_10.md`.
