Validation Targets
==================

Each scenario's ``targets.yaml`` defines the numeric criteria that determine
whether a simulation run reproduces the macroeconomic phenomena described in
Delli Gatti et al. (2011). This page documents how targets are structured,
where their values come from, and the methodology used to calibrate them.


Target Structure
----------------

A ``targets.yaml`` file contains two sections:

**metrics:** — Standardized targets used by the scoring engine for pass/fail
decisions. Each metric has a ``check_type`` (see :doc:`scoring`) and
corresponding parameters:

.. code-block:: yaml

   metrics:
     unemployment_rate_mean:
       target: 0.065      # Book Figure 3.4b mean = 0.0645
       tolerance: 0.02

     okun_correlation:
       min: -0.98
       max: -0.70

     inflation_hard_ceiling_upper:
       threshold: 0.25    # BOOLEAN: max inflation < 25%

**metadata.visualization:** — Reference values used by ``viz.py`` for plot
annotations and visual comparison bands. These do not affect scoring:

.. code-block:: yaml

   metadata:
     visualization:
       time_series:
         unemployment_rate:
           targets:
             mean_target: 0.065
             normal_min: 0.02
             normal_max: 0.11


Evidence Sources
----------------

Every target value is grounded in one or more of three evidence types:

1. **Extracted book values** — Precise numbers from the author's simulation
   figures, obtained through pixel-level analysis of the original MATLAB plots.
   Example: ``log_gdp_mean: 8.91`` extracted from Figure 3.4a.

2. **Economic theory** — Established empirical regularities from the
   macroeconomics literature. Example: labor share (wage/productivity ratio)
   of 0.60--0.70 from Kaldor's stylized facts.

3. **Multi-seed model behavior** — Statistical properties of the BAM Engine
   model across 100+ seeds. Ranges are set wide enough that 95%+ of seeds
   pass while still being economically meaningful. Example:
   ``phillips_correlation min: -0.50`` gives 2-sigma headroom below the model
   mean of -0.31.

.. list-table:: Economic Anchors
   :header-rows: 1
   :widths: 30 35 35

   * - Concept
     - Expected Value
     - Source
   * - Labor share
     - 0.60--0.70
     - Kaldor's stylized facts
   * - Wages track productivity
     - Strong positive correlation
     - Kaldor's 1st fact
   * - Phillips curve
     - Negative, weak (-0.1 to -0.4)
     - Phillips (1958)
   * - Okun's law
     - Strong negative (-0.7 to -0.95)
     - Okun (1962)
   * - Beveridge curve
     - Negative, variable (-0.1 to -0.8)
     - Beveridge (1944)
   * - Firm size distribution
     - Right-skewed (power law tail)
     - Axtell (2001)
   * - Growth rate distribution
     - Tent-shaped (Laplace)
     - Stanley et al. (1996)


Target Calibration Philosophy
-----------------------------

Targets define a **validity region** where both of the following hold:

1. The book's specific simulation realization falls within the range.
2. Simulations that differ from the book but are *economically sounder* also
   pass.

This means:

- **Center** on the economically expected value, not necessarily the book value.
- **Ranges** must encompass both the book value and the theoretical expectation.
- A model that validates economic theory *better* than the book should still
  pass.

When targets should **not** be changed:

- **Structural gates** — BOOLEAN checks (``threshold``) and wide RANGE checks
  that serve as sanity checks (e.g., ``inflation_hard_ceiling < 25%``). These
  are intentionally loose for multi-seed robustness.
- **Unreliable extraction** — When dense dot overlap makes the extracted value
  unreliable (e.g., Okun's curve where only 97 of ~250 dots could be resolved).
- **Already aligned** — Current target already matches the book value.


Figure Reproduction Pipeline
-----------------------------

Validation targets for the Growth+ scenario (Section 3.9.2) are calibrated
from 16 book figures using a two-phase extraction and update process.

**Phase 1 — Reproduce figure:**

::

   Original PNG  -->  Extract Script  -->  NPZ Data  -->  Reproduce Script  -->  Verify PNG

Each figure has a paired ``extract_<id>.py`` and ``reproduce_<id>.py`` script.
The extract script detects colored pixels in the original MATLAB output image,
converts pixel coordinates to data coordinates, and saves the result as a
NumPy ``.npz`` file. The reproduce script generates a side-by-side comparison
image for visual verification.

**Phase 2 — Update targets:**

::

   NPZ Data  -->  Compute Metrics  -->  Compare  -->  Stability Test  -->  Apply  -->  Validate

The extracted data is used to compute the specific metrics each figure informs.
A 100-seed stability test is run before and after any changes to verify that
pass rates are maintained (>= 95%) and no regressions occur.


Extraction Methods
------------------

Different figure types require different extraction approaches:

**Smooth time series** (Figures 3.4a, 3.4d) — Column scanning: for each
x-pixel column, find blue pixels and take their mean y-position. Works well
because the line is mostly horizontal.

**Volatile time series** (Figures 3.4b, 3.4c, 3.6c--d, 3.7a--b) —
Dual-envelope extraction: at sharp transitions, the plotted line creates a
tall vertical streak of blue pixels. The method records upper (min y-pixel)
and lower (max y-pixel) per column, then selects the envelope value furthest
from the local running mean at transition columns.

**Scatter plots** (Figures 3.5a--c) — Connected component labeling
(``scipy.ndimage.label``) to find dot clusters, with centroid extraction.
For dense overlapping regions (e.g., Okun's curve), distance transform +
local maxima detection (``distance_transform_edt`` + ``maximum_filter``)
recovers additional dot centers.

**Log-rank distribution plots** (Figures 3.6a--b) — Two-color detection
(blue for negative, red for positive growth rates) on a logarithmic y-axis.
Tent shape quality (Laplace R\ :sup:`2`) is computed from linear regression
on log\ :sub:`10`\ (rank) vs growth rate per side.

**Histograms** (Figure 3.5d) — Connected component labeling to find
individual bars; bar height gives count, bar position gives bin center.


Figure-to-Metrics Mapping
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Figure
     - Type
     - Metrics Affected
   * - 3.4a (GDP)
     - Smooth TS
     - ``log_gdp_mean``, ``log_gdp_trend``, ``log_gdp_total_growth``
   * - 3.4b (Unemployment)
     - Volatile TS
     - ``unemployment_rate_mean``, ``unemployment_std``
   * - 3.4c (Inflation)
     - Volatile TS
     - ``inflation_rate_mean``
   * - 3.4d (Productivity + Wage)
     - Dual smooth TS
     - ``productivity_growth``, ``real_wage_growth``, ``productivity_wage_correlation``, ``wage_productivity_ratio_*``, ``productivity_trend``
   * - 3.5a (Phillips curve)
     - Scatter
     - ``phillips_correlation``
   * - 3.5b (Okun curve)
     - Dense scatter
     - ``okun_correlation``
   * - 3.5c (Beveridge curve)
     - Scatter
     - ``beveridge_correlation``, ``vacancy_rate_mean``
   * - 3.5d (Firm size dist.)
     - Histogram
     - ``firm_size_skewness``, ``firm_size_pct_below_*``
   * - 3.6a (Output growth dist.)
     - Log-rank tent
     - ``output_growth_tent_r2``, ``output_growth_positive_frac``
   * - 3.6b (NW growth dist.)
     - Log-rank tent
     - ``networth_growth_tent_r2``
   * - 3.6c (Real interest rate)
     - Volatile TS
     - ``real_interest_rate_*``
   * - 3.6d (Bankruptcies)
     - Volatile TS
     - ``bankruptcies_mean``
   * - 3.7a (Financial fragility)
     - Volatile TS
     - ``financial_fragility_*``
   * - 3.7b (Price ratio)
     - Volatile TS
     - ``price_ratio_*``
   * - 3.7c (Price dispersion)
     - Volatile TS
     - ``price_dispersion_*``
   * - 3.7d (Equity + Sales disp.)
     - Dual volatile TS
     - ``equity_dispersion_*``, ``sales_dispersion_*``


Practical Lessons
-----------------

Several insights emerged from the calibration process:

- **RGBA images require explicit conversion** — Always use
  ``.convert("RGB")`` before pixel analysis. The alpha channel can cause
  boolean mask operations to silently produce empty results.

- **Dense scatter clusters are unresolvable** — When MATLAB dots overlap into
  a continuous mass (e.g., Okun's curve), no morphological technique can
  recover individual dot positions. Accept the limitation and note it for
  affected metrics.

- **Book text values take precedence** — When the book explicitly states a
  number (e.g., Phillips r = -0.19), prefer it over the pixel extraction
  which has scatter noise.

- **Cross-figure data reuse** — Some metrics lack dedicated figures. The
  vacancy rate mean is derived from the Beveridge curve's y-axis (Figure 3.5c).

- **Widening ranges never causes regressions** — Making a RANGE ``min`` more
  negative or ``max`` more positive can only help pass rates. Only tightening
  or shifting centers risks regression.
