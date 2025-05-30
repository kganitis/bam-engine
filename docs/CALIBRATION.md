### Quick-start calibration cheatsheet

| parameter                    | ball-park range                       | suggested default      | units & scale                                       |
|------------------------------| ------------------------------------- |------------------------|-----------------------------------------------------|
| **beta**                     | 0.5 – 4                               | **2**                  | dimension-less exponent                             |
| **delta**                    | 0.10 – 0.50                           | **0.30**               | share of profits paid as dividends                  |
| **v**                        | 0.06 – 0.10                           | **0.08**               | capital-adequacy coefficient ⇒ max leverage = 1 / v |
| **r̄**                       | 0.0025 – 0.0125                       | **0.010**              | quarterly nominal rate (0.25–1.25 % → default = 1 %) |
| **min\_wage**                | 0.8 · price\_init – 1.2 · price\_init | **1 · price\_init**    | nominal wage per period                             |
| **min\_wage\_rev\_period**   | 8 – 24                                | **12**                 | periods between updates                             |
| **net\_worth\_init**         | 25 – 100                              | **50**                 | currency units per firm                             |
| **production\_init**         | 50 – 150                              | **100**                | units of good per firm                              |
| **price\_init**              | 0.8 – 1.2                             | **1.0**                | currency units per good                             |
| **savings\_init**            | 10 – 40                               | **20**                 | currency units per household                        |
| **wage\_offer\_init**        | 1.0 – 1.3 · min\_wage                 | **1.1 · min\_wage**    | nominal wage per period                             |
| **equity\_base\_init**       | 500 – 1500                            | **1000**               | currency units per bank                             |

---

## Mini calibration recipe & tuning logic

Below I walk through *why* the defaults are reasonable, how each parameter “talks” to the others, and the triggers that tell you it is time to recalibrate.

### 1.  **β – consumption-propensity exponent**

* **What it does**
  You map relative savings to a consumption share

  $$
    \pi_j = \frac{1}{1+\tanh\!\bigl(SA_j/SA_{\text{avg}}\bigr)^{\beta}}
  $$

  Larger **β** amplifies heterogeneity: rich agents save noticeably more, poor agents consume almost all income.

* **Why 2?**
  At **β = 2** and $SA_j=SA_{\text{avg}}$, average propensity to consume ≈ 0.63, leaving 37 % for savings—close to empirical household saving in developed economies.

* **Couplings**
  *Higher β ↓ aggregate demand*. If you push β up, partially offset by

  * lowering **savings\_init**, or
  * raising **min\_wage**.
    Rule of thumb: every +1 in β reduces initial aggregate demand \~6 pp; lift min\_wage or cut savings\_init by \~10 % to re-balance.

* **Re-calibrate when** you change:

  * **h\_η** (price shock width) or
  * household population size.

---

### 2. **δ – dividend pay-out ratio**

* **What it does**
  Determines the split between retained earnings and dividends.

* **Why 0.30?**
  Mature listed firms pay ≈ 30 % of profits on average; startups pay 0. Realistic mix for 500 identical firms is in that middle.

* **Couplings**
  Higher **δ** ↓ internal funds → ↑ external borrowing.
  If you raise δ above 0.4, consider

  * boosting **net\_worth\_init** 10–20 % *or*
  * lowering **v** (tighter capital rule) so banks lend less aggressively.

* **Re-calibrate when** you introduce asymmetric firm ages or a shock that skews profits.

---

### 3. **v – bank capital requirement**

* **What it does**
  Maximum leverage = 1/ v. **v = 0.08** ⇒ 12.5× leverage, matching Basel III.

* **Couplings**

  * Lower **v** (looser) → credit boom → inflationary pressure; compensate by nudging **r̄** up 25 bp for every 0.02 drop in **v**.
  * Very low **v** demands higher **equity\_base\_init** to keep banks solvent after shocks $h_\phi$.

* **Re-calibrate when** you change **h\_\phi** or add a systemic-risk policy module.

---

### 4. **r̄ – baseline nominal interest (quarterly)**

* **Why 1 %?**
  Sits between the ZLB and the post-pandemic hikes; leaves room for monetary reactions to shocks.

* **Couplings**

  * Higher r̄ ↑ household interest income → effectively ↓ average π; similar tuning to higher **β**.
  * Also ↑ bank profits → allows slightly lower **equity\_base\_init**.

* **Re-calibrate when** you alter the monetary-policy rule or frequency (period length).

---

### 5–6. **min\_wage & min\_wage\_rev\_period**

* **Why min\_wage = price\_init?**
  Normalises the real wage to 1 unit of the good, making early-period consumption budgeting transparent.

* **Revision every 12 periods**
  With quarterly periods, that is 3 years—close to typical statutory review cycles.
  Shorter cycles (≤8) needed only if **h\_η** or **h\_ξ** are large, leading to rapid real-wage erosion.

* **Couplings**
  If you double **h\_ξ** (wage shock cap) you can leave min\_wage alone but cut **min\_wage\_rev\_period** by half to keep the floor binding.

---

### 7–13. Initial balance-sheet scalars

| parameter                               | Why this default                                                                                              | Key interactions & tuning rules                                                                                          |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **net\_worth\_init = 50**               | Gives firms ≈ ½ year of wage bill buffer (θ = 8) at default wages.                                            | Raise 20 % for higher **δ** or **h\_ρ**.                                                                                 |
| **production\_init = 100**              | Ensures initial aggregate supply (500 × 100) can match demand if 60 % of household income is spent.           | If you scale *n\_firms* use: production\_init ≈ (n\_households × wage\_offer\_init × π\_avg) / (n\_firms × price\_init). |
| **price\_init = 1**                     | Makes nominal and real values identical at t = 0.                                                             | If you change, adjust min\_wage and wage\_offer\_init proportionally.                                                    |
| **savings\_init = 20**                  | Roughly 3 months of wages; prevents immediate credit reliance.                                                | If **β**>3, drop to \~15 to avoid stagnant demand.                                                                       |
| **wage\_offer\_init = 1.1 · min\_wage** | Gives firms room to bid down in slack labour markets while staying above the floor.                           | Tie to min\_wage; keep ratio 1.05–1.3.                                                                                   |
| **equity\_base\_init = 1000**           | With v = 0.08 allows ≈ 12 500 of balance-sheet assets per bank, enough to cover initial firm borrowing needs. | For each –0.01 change in **v**, raise equity\_base\_init by \~150 to preserve the same asset capacity.                   |

---

## When is a *full* re-calibration warranted?

| Trigger                                                   | Why it matters                                 | What to revisit first                                  |
| --------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------ |
| **Changing any shock width h\_ρ, h\_ξ, h\_φ, h\_η**       | Alters volatility → solvency & wage dispersion | v, equity\_base\_init, min\_wage\_rev\_period          |
| **Altering population sizes**                             | Changes scale of flows                         | production\_init, net\_worth\_init, equity\_base\_init |
| **Switching period length (e.g. monthly → quarterly)**    | Re-indexes all flows per period                | r̄ (annualise → divide by 4), min\_wage\_rev\_period   |
| **New behavioural rules (e.g. endogenous credit limits)** | May break old steady-state                     | β, v, savings\_init                                    |
| **Empirical validation step**                             | Comparing to target macro ratios               | Fine-tune β (consumption), δ (payout), v (leverage)    |

---

### Calibrate iteratively, not once-and-for-all

1. **Back-of-envelope check**
   Ensure *initial* goods market roughly clears:

   $$
     n_{\text{households}}\;wage\;π_{\text{avg}} ≈ n_{\text{firms}}\;production\;price
   $$

2. **Dry-run for 50 – 100 periods**
   Watch for: bankruptcy waves, explosive credit, wage free-fall.

3. **Stress-test** with max shocks (use the *caps*, not Gaussians).
   If >10 % of firms or 1 bank fail within 10 periods, shore up net\_worth\_init or raise v.

4. **Lock parameters** and only revisit on the triggers above.

By starting with the defaults and using the coupling rules as “sliders,” you should reach a stable yet lively baseline from which policy and shock experiments will be informative rather than dominated by numerical artefacts.
