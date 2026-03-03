Roadmap
=======

Originally developed as part of MSc thesis research at the University of
Piraeus, BAM Engine is now a personal project under independent development.
This roadmap outlines known limitations and strategic goals for the project.
No timelines are attached --- priorities may shift as development progresses.

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

**Architecture & API**
   Make the framework more powerful and easier to extend:

   - **Role-based agent identity**

     Today, agents exist only as implicit array indices inside roles,
     and each role is hardcoded to a single agent type (firms, households, or banks).
     The goal is to make agents first-class entities that own their roles,
     so the engine discovers participants by querying for roles
     ("which agents have the Borrower role?") rather than by type.
     This would allow the same role to be shared across different agent types
     and new agent kinds to be introduced without changing the core.

   - **Economy as self-contained context**

     Currently, events receive the entire Simulation object and reach through it
     to access roles, configuration, and economy state. Extracting the economy into
     a standalone context that events receive directly would decouple the model
     definition from the simulation driver, and open the door to running multiple
     economies within a single simulation (e.g., multi-country simulations).

   - **Multi-point data collection**

     The data collector currently captures each metric once per step
     (at the end or after a single designated event). Supporting capture
     at multiple points within the same step would let users observe how
     variables evolve across events within a period.

   - **Easier extensibility**

     Adding new roles and events is straightforward, but extending or customizing
     existing ones (e.g., adding fields to a built-in role) requires workarounds.
     The registration system should support extending existing components without
     breaking discovery.

**Research Extensions**
   Add extensions from the broader CATS family: CC-MABM, a capital and credit extension
   implemented in Julia, and R-MABM, a reinforcement-learning extension of CC-MABM in
   Python and Julia. See :doc:`related_projects` for full details on these projects.
