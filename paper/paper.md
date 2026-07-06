---
title: 'BAM Engine: A vectorized Python framework for the Bottom-up Adaptive Macroeconomics model'
tags:
  - Python
  - agent-based modeling
  - computational economics
  - macroeconomics
  - complex adaptive systems
  - heterogeneous agents
authors:
  - name: Konstantinos Ganitis
    orcid: 0009-0001-6976-723X
    affiliation: 1
affiliations:
  - name: Department of Informatics, University of Piraeus, Greece
    index: 1
date: 6 July 2026
bibliography: paper.bib
---

# Summary

BAM Engine is an open-source Python framework that implements the Bottom-up
Adaptive Macroeconomics (BAM) model of @delli_gatti_2011, a canonical
agent-based macroeconomic model from the CATS (Complex Adaptive Trivial Systems) family. The
model populates an artificial economy with heterogeneous firms, households, and
banks that interact across three markets: labor, credit, and consumption goods.
Rather than imposing aggregate behavior through representative-agent equations,
the model lets macroeconomic phenomena such as growth, unemployment, inflation,
and endogenous business cycles emerge from the decentralized, boundedly rational
decisions of many individual agents [@delli_gatti_2005_fluctuations;
@tesfatsion_2006_ace].

BAM Engine is validated against quantitative targets from the source text and
benchmarked against four other agent-based modeling frameworks, and it ships
three extensions that demonstrate how the model can be varied and extended.

# Statement of need

Agent-based macroeconomic models are usually disseminated as systems of
equations and prose, frequently without accompanying source code. The BAM model
in particular is specified verbally [@delli_gatti_2011], so every researcher who
studies it must re-implement it from the text and make independent interpretive
choices along the way. Results are therefore hard to reproduce, compare, or build
upon, which is the central obstacle this framework addresses: it provides a
single, inspectable, validated reference implementation that others can run,
audit, and modify.

BAM Engine's intended users are researchers, educators, and students in
agent-based computational economics who need a fast and modifiable reference
implementation: to reproduce the book's results, to experiment with parameters
and model variants, and to run the large ensembles of simulations that
quantitative analysis demands.

# State of the field

General-purpose ABM frameworks ease the engineering burden but impose a
trade-off. Object-oriented toolkits such as Mesa in Python [@kazil_2020_mesa;
@terhoeven_2025_mesa3], NetLogo [@wilensky_1999_netlogo], and Repast
[@north_2013_repast] represent each agent as an object and dispatch behavior
agent by agent. That design is flexible
but scales poorly to the thousands of long runs needed for calibration,
sensitivity, and robustness studies; the first-party benchmark reported below
measures idiomatic per-agent ports of the same model running
an order of magnitude slower than BAM Engine at moderate scale. The mesa-frames project
[@amer_2024_mesaframes] independently diagnoses the same per-agent overhead and
retrofits columnar storage onto Mesa, but it remains a general-purpose,
early-stage library with no economic primitives. Domain-specific macro models
such as EURACE [@deissenberg_2008_eurace] achieve scale, yet they are monolithic
research artifacts (C with MPI) that expose no reusable API or extension system.

For the baseline BAM model specifically, existing implementations are limited: a
NetLogo port (general-purpose, per-agent, not designed for large-scale
experiments) and the Julia-based ABCredit engine wrapped by R-MABM
[@brusatin_2024_rmabm], a reinforcement-learning variant of the CATS credit
model that requires a Julia runtime. To the author's knowledge, BAM Engine is the first
vectorized, pure-Python framework for the baseline BAM model that combines a
reusable, extensible architecture with systematic validation against the original
text.

To ground this comparison in measurement, the repository ships a
cross-framework benchmark (the `comparison/` package) that runs the identical
baseline BAM model on five frameworks: BAM Engine, Mesa [@kazil_2020_mesa],
mesa-frames [@amer_2024_mesaframes], Agents.jl [@datseris_2022_agentsjl], and
a pre-existing third-party NetLogo implementation [@platas_lopez_2020_bam].
Before any timing is counted, each port must pass a behavioral-equivalence
gate: twenty seeds compared against the BAM Engine reference on unemployment,
output, and inflation dynamics, firm-size skewness, and cross-correlation
structure, so the comparison measures the same model rather than five similar
ones. The NetLogo implementation participates as a non-blocking cross-language
reference: it reproduces the baseline levels but diverges on parts of the
co-movement structure. All timings were collected on a single machine
(Apple M4 Pro); jobs are single-threaded and run serially, and the result
snapshots and environment captures are committed with the repository.

At 1,000 firms (6,100 agents in total), BAM Engine simulates a period in
about 1.5 ms: roughly 11x faster than the idiomatic Mesa port, 13x faster
than mesa-frames, and 1.4x faster than the compiled Agents.jl port. At 20,000
firms (122,000 agents), BAM Engine with the optional Numba kernel runs 3.7x
faster than Agents.jl, while the per-agent Python frameworks do not complete
within budget beyond 5,000 firms. The results split by execution model rather
than by language: array-oriented engines reach the population sizes that
ensemble studies require, and BAM Engine is the fastest of the group from 500
firms upward (\autoref{fig:scaling}). The competitor ports are written
competently and profiled to their inherent floors rather than left naive, so
the gaps reflect framework design, not implementation effort.

![Steady-state per-period wall time against total agent count, log-log. Mesa,
mesa-frames, and NetLogo hit the per-job time budget at smaller populations
than the array-oriented engines (Apple M4 Pro).\label{fig:scaling}](scaling.png){ width="75%" }

# Software design

The framework is built on an Entity-Component-System (ECS) architecture that
follows data-oriented design principles. Agent state is not held in per-agent
objects; instead, each behavioral facet (a "role") is a structure of parallel
NumPy arrays [@harris_2020_numpy], and each economic process (an "event") is a
stateless system that transforms those arrays in place. Events run in an
explicit, YAML-configurable pipeline organized into the model's economic phases.
This layout replaces per-agent Python loops with vectorized array operations,
and it lets researchers reorder, reconfigure, and extend the model (adding new
roles, events, relationships, and market mechanisms) without modifying the
engine core. Three extensions ship with the framework and demonstrate the
mechanism: R&D-driven productivity growth [@russo_2007_industrial], buffer-stock
consumption [@carroll_1997_buffer], and profit taxation. The package depends only on NumPy and PyYAML at runtime, is
fully type-annotated, tested under continuous integration across operating
systems and Python versions, and documented with a tutorial-style user guide and
runnable examples.

Two design trade-offs are worth stating. First, vectorization is applied
selectively: the labor and credit markets use batched array matching with a
sparse per-row sampler, while the goods market intentionally remains a
sequential loop because its purchase-by-purchase state updates make
vectorized approximations behaviorally incorrect; an optional Numba kernel
(`pip install bamengine[fast]`) compiles that loop instead, producing
bit-identical results roughly twice as fast at scale. Second, the event
pipeline is ordered explicitly in YAML rather than derived from a dependency
graph: economic causality in the BAM model is an assumption to be inspected
and varied, not an implementation detail to be inferred.

# Research impact statement

BAM Engine is validated against quantitative targets extracted from
@delli_gatti_2011 using a two-layer scheme that pairs discrete PASS/WARN/FAIL
status checks with continuous scores in the range 0 to 1, the latter feeding an
automated calibration pipeline. Across 1,000 random seeds of 1,000 periods each,
the baseline scenario passes for 98.8% of seeds, the R&D growth scenario for
97.2%, and the buffer-stock scenario for 98.8% (results committed with release
v0.10.0). The implementation reproduces
canonical stylized facts of the model, including a strong Okun's-law relationship
(mean correlation of about -0.87), Phillips and Beveridge curves,
right-skewed firm-size distributions, and financial fragility that co-moves with
output in a Minsky-type manner. A robustness suite provides multi-seed
internal-validity checks, univariate sensitivity analysis, and structural
experiments.

Because the implementation is transparent, fast, and reproducible, it has also
surfaced structural properties of the BAM model that the source text does not
discuss. Two are documented as hypotheses: a labor-market "quantization trap," in
which integer rounding of labor demand produces a one-way ratchet that suppresses
production-driven layoffs, and a credit-market "Kalecki trap," in which the
profit identity drives firm net worth toward a steady-state attractor that nearly
eliminates borrowing. These observations illustrate the mechanism-level analysis
that a faithful, open reference implementation makes possible, and they motivate
the model variants explored through the extension system.

# AI usage disclosure

BAM Engine was developed with the assistance of generative AI: Anthropic Claude
models (Claude Sonnet and Opus families, versions 3.7 through 4.x), used through
the Claude Code development tool (2025-2026), assisted with code generation,
refactoring, test authoring, documentation writing, and drafting of this paper,
under the author's direction and continuous supervision. The author framed every
problem, made all core design and architectural decisions (the
entity-component-system layout, the event pipeline, the validation methodology
and targets, the benchmark design and its behavioural-equivalence gate),
reviewed and validated every change against the source text and the committed
test and validation suites, and takes full responsibility for the correctness,
originality, and licensing of the software and of this manuscript.

# Acknowledgements

BAM Engine was developed as part of MSc thesis research at the University of
Piraeus.

# References
