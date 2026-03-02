Related Projects
================

The BAM (Bottom-Up Adaptive Macroeconomics) model has been implemented in
several programming languages by different research groups. This page lists
known implementations and closely related variants from the CATS (Complex
Adaptive Trivial Systems) model family.


BAMmodel
--------

A Python/NetLogo implementation of the BAM model developed by Platas-López
et al. Provides both a NetLogo version for interactive exploration and a
Python version for batch simulations.

- **Citation:** Platas-López, A., Panadero, J., & Juan, A. A. (2019).
  BAMmodel: An Agent-Based Simulation Model for the Bottom-Up Adaptive
  Macroeconomics Framework. *Frontiers in Artificial Intelligence and
  Applications*, 319, 149--156. DOI:
  `10.3233/FAIA190141 <https://doi.org/10.3233/FAIA190141>`_
- **Repository:** https://github.com/alexplatasl/BAMmodel


R-MABM
------

An R implementation of the MABM (Macroeconomic Agent-Based Model) variant,
which extends the BAM model with capital goods and a richer credit market.
Based on the formalization by Assenza, Delli Gatti, and Grazzini (2015).

- **Citation:** Assenza, T., Delli Gatti, D., & Grazzini, J. (2015).
  Emergent dynamics of a macroeconomic agent based model with capital and
  credit. *Journal of Economic Dynamics and Control*, 50, 5--28. DOI:
  `10.1016/j.jedc.2014.07.001 <https://doi.org/10.1016/j.jedc.2014.07.001>`_
- **Repository:** https://github.com/Brusa99/R-MABM


ABCredit.jl
-----------

A Julia implementation developed at the Banca d'Italia (Bank of Italy) by
some of the same researchers behind R-MABM. Takes advantage of Julia's
performance characteristics for large-scale simulations of the credit-augmented
MABM variant.

- **Repository:** https://github.com/bancaditalia/ABCredit.jl
