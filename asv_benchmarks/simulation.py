"""ASV benchmarks for BAM Engine simulation performance.

These benchmarks track performance across commits with machine-specific
baselines, solving the CI runner variability issue.
"""

import bamengine as bam


def _run(sim, n_periods):
    """Run simulation without result collection (backward compatible).

    The ``collect`` parameter was added in v0.1.2. For earlier versions,
    fall back to plain ``run()`` which had no collection by default.
    """
    try:
        sim.run(n_periods=n_periods, collect=False)
    except TypeError:
        sim.run(n_periods=n_periods)


def _try_get_event(sim, name):
    """Look up a pipeline event, returning None if missing."""
    try:
        return sim.get_event(name)
    except KeyError:
        return None


def _require_event(event, name):
    """Raise NotImplementedError if event was not resolved.

    ASV treats NotImplementedError as "skip this benchmark for this version".
    """
    if event is None:
        raise NotImplementedError(f"Event '{name}' not available in this version")


class SimulationSuite:
    """Benchmark suite for full simulation runs."""

    # ASV timeout (seconds) - large/1000 periods needs ~80-90s
    timeout = 120

    # Configuration parameters
    params = ["small", "medium", "large"]
    param_names = ["config"]

    # Configuration details
    configs = {
        "small": {"n_firms": 100, "n_households": 500, "n_banks": 10},
        "medium": {"n_firms": 200, "n_households": 1000, "n_banks": 10},
        "large": {"n_firms": 500, "n_households": 2500, "n_banks": 10},
    }

    def setup(self, config):
        """Setup simulation before each benchmark run."""
        cfg = self.configs[config]
        self.sim = bam.Simulation.init(
            n_firms=cfg["n_firms"],
            n_households=cfg["n_households"],
            n_banks=cfg["n_banks"],
            seed=42,
            log_level="ERROR",
        )

    def time_simulation_100_periods(self, config):
        """Benchmark 100 simulation periods."""
        _run(self.sim, 100)

    def time_simulation_1000_periods(self, config):
        """Benchmark 1000 simulation periods (full baseline)."""
        _run(self.sim, 1000)


class PipelineSuite:
    """Benchmark suite for pipeline operations."""

    def setup(self):
        """Setup minimal simulation."""
        self.sim = bam.Simulation.init(
            n_firms=100,
            n_households=500,
            n_banks=10,
            seed=42,
            log_level="ERROR",
        )

    def time_single_step(self):
        """Benchmark a single simulation step (all 37 events)."""
        self.sim.step()


class MemorySuite:
    """Benchmark memory usage."""

    params = [100, 200, 500]
    param_names = ["n_firms"]

    def peakmem_simulation_init(self, n_firms):
        """Peak memory during simulation initialization."""
        sim = bam.Simulation.init(
            n_firms=n_firms,
            n_households=n_firms * 5,
            n_banks=max(2, n_firms // 10),
            seed=42,
            log_level="ERROR",
        )
        return sim

    def peakmem_simulation_100_periods(self, n_firms):
        """Peak memory during 100-period simulation."""
        sim = bam.Simulation.init(
            n_firms=n_firms,
            n_households=n_firms * 5,
            n_banks=max(2, n_firms // 10),
            seed=42,
            log_level="ERROR",
        )
        _run(sim, 100)


class CriticalEventSuite:
    """Benchmark critical path events (goods/labor/credit markets).

    These events are the primary bottlenecks identified in profiling.
    Individual benchmarks are skipped for versions where the event
    does not exist (e.g., market round events added in v0.6.0).
    """

    timeout = 60

    def setup(self):
        """Setup simulation in steady state.

        Events are resolved here so the lookup cost is excluded from
        timed methods. Missing events are stored as None and their
        benchmarks raise NotImplementedError (ASV skip).
        """
        self.sim = bam.Simulation.init(
            n_firms=100,
            n_households=500,
            n_banks=10,
            seed=42,
            log_level="ERROR",
        )
        _run(self.sim, 10)
        self._consumers_decide = _try_get_event(
            self.sim, "consumers_decide_firms_to_visit"
        )
        self._goods_round = _try_get_event(self.sim, "goods_market_round")
        self._workers_decide = _try_get_event(self.sim, "workers_decide_firms_to_apply")
        self._labor_round = _try_get_event(self.sim, "labor_market_round")
        self._credit_round = _try_get_event(self.sim, "credit_market_round")

    # Goods market events
    def time_consumers_decide_firms_to_visit(self):
        """Benchmark firm selection for shopping."""
        _require_event(self._consumers_decide, "consumers_decide_firms_to_visit")
        self._consumers_decide.execute(self.sim)

    def time_goods_market_round(self):
        """Benchmark goods market matching (one round)."""
        _require_event(self._goods_round, "goods_market_round")
        self._goods_round.execute(self.sim)

    # Labor market events
    def time_workers_decide_firms_to_apply(self):
        """Benchmark firm selection for job applications."""
        _require_event(self._workers_decide, "workers_decide_firms_to_apply")
        self._workers_decide.execute(self.sim)

    def time_labor_market_round(self):
        """Benchmark labor market matching (one round)."""
        _require_event(self._labor_round, "labor_market_round")
        self._labor_round.execute(self.sim)

    # Credit market events
    def time_credit_market_round(self):
        """Benchmark credit market matching (one round)."""
        _require_event(self._credit_round, "credit_market_round")
        self._credit_round.execute(self.sim)


class InitSuite:
    """Benchmark initialization costs across scales."""

    timeout = 60
    params = [100, 200, 500, 1000]
    param_names = ["n_firms"]

    def time_simulation_init(self, n_firms):
        """Time simulation initialization."""
        bam.Simulation.init(
            n_firms=n_firms,
            n_households=n_firms * 5,
            n_banks=max(2, n_firms // 10),
            seed=42,
            log_level="ERROR",
        )


class LoanBookSuite:
    """Benchmark sparse relationship operations.

    The LoanBook uses COO sparse format for memory-efficient loan storage.
    These benchmarks track the performance of common operations.
    """

    timeout = 30

    def setup(self):
        """Setup LoanBook with pre-populated loans."""
        import numpy as np

        from bamengine.relationships import LoanBook

        self.np = np
        self.loans = LoanBook()
        self.rng = np.random.default_rng(42)
        # Pre-populate with 500 loans (50 per bank × 10 banks)
        for bank in range(10):
            n = 50
            self.loans.append_loans_for_lender(
                lender_idx=np.intp(bank),
                borrower_indices=self.rng.integers(0, 100, n, dtype=np.int64),
                amount=self.rng.uniform(100, 1000, n),
                rate=self.rng.uniform(0.02, 0.05, n),
            )

    def time_append_loans(self):
        """Time appending 100 loans."""
        self.loans.append_loans_for_lender(
            lender_idx=self.np.intp(0),
            borrower_indices=self.rng.integers(0, 100, 100, dtype=self.np.int64),
            amount=self.rng.uniform(100, 1000, 100),
            rate=self.rng.uniform(0.02, 0.05, 100),
        )

    def time_debt_per_borrower(self):
        """Time debt aggregation by borrower."""
        self.loans.debt_per_borrower(n_borrowers=100)

    def time_purge_borrowers(self):
        """Time purging 10 bankrupt borrowers."""
        bankrupt = self.rng.choice(100, 10, replace=False).astype(self.np.int64)
        self.loans.purge_borrowers(bankrupt)


class ScalingSuite:
    """Track performance scaling with agent count.

    Verifies that performance scales sub-linearly with agent count
    due to NumPy vectorization efficiency.
    """

    timeout = 120
    params = [50, 100, 200, 400]
    param_names = ["n_firms"]

    def setup(self, n_firms):
        """Setup simulation for given scale."""
        self.sim = bam.Simulation.init(
            n_firms=n_firms,
            n_households=n_firms * 5,
            n_banks=max(2, n_firms // 10),
            seed=42,
            log_level="ERROR",
        )

    def time_100_periods(self, n_firms):
        """Track how 100-period runtime scales with agent count."""
        _run(self.sim, 100)

    def time_single_step(self, n_firms):
        """Track how single-step runtime scales."""
        self.sim.step()
