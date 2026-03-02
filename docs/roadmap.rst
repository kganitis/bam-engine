Roadmap
=======

BAM Engine is under active development as part of MSc thesis research at the
University of Piraeus. This roadmap outlines known limitations and strategic
goals for the project. No timelines are attached --- priorities may shift as
research progresses.

For bug reports and feature requests, see the
`issue tracker <https://github.com/kganitis/bam-engine/issues>`_.


Known Limitations
-----------------

The following structural issues are documented and under investigation. There
is strong evidence that they share interconnected root causes, so they are best
understood as facets of the same underlying problem rather than independent bugs.

**Labor Market Quantization Trap**
   The ceiling function in the hiring rule (``ceil(desired_production / phi)``)
   creates a one-way ratchet: at small firm sizes, firms can increase their
   workforce but never decrease it. This inflates employment and suppresses
   the unemployment rate. See the labor market section of the
   :doc:`user_guide/bam_model` page for details.

**Credit Market Inactivity (Kalecki Trap)**
   The Kalecki profit identity guarantees that aggregate profits exceed costs,
   causing firms to accumulate net worth faster than they accumulate debt. Once
   the net-worth-to-wage-bill ratio reaches its steady-state attractor (~12x),
   firms self-finance entirely and borrowing demand drops to zero. The dividend
   payout rate (``delta``) is the only effective parameter lever. See the
   credit market section of the :doc:`user_guide/bam_model` page for details.

**Price Dynamics**
   Several price-related metrics deviate from reference targets: low mean
   inflation, low dispersion in firm-level prices, equity, and sales, and a
   high ratio of market price to market-clearing price. These are likely
   downstream consequences of the credit and labor market traps reducing
   competitive pressure.


Strategic Goals
---------------

**Model Accuracy**
   Resolve the structural traps described above, improve price dynamics, and
   achieve a closer match to the simulation results in the reference book
   (Delli Gatti et al., 2011).

**API & Usability**
   Simplify the user-facing API, reduce exposure of ECS internals, and make
   it easier to capture, export, and analyze a wider variety of simulation
   metrics.

**Code Quality & Performance**
   Reduce cyclomatic complexity, improve test coverage and type strictness,
   and explore JIT compilation (Numba) for critical code paths. Refactor
   complex event systems toward single-responsibility design.

**Research Extensions**
   Implement remaining scenarios from the reference book, including DSGE
   methodology comparison (Section 3.9.3). Explore variants from the broader
   CATS family: CC-MABM (credit and capital) and R-MABM (capital and credit
   with R implementation).

**Ecosystem & Community**
   Submit to the `CoMSES Model Library <https://www.comses.net/codebases/>`_,
   continue documentation refinement, and finalize project branding.
